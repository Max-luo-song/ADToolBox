"""Read-only image and numeric tools. No cause selection or behavior rules."""
import base64
import hashlib
import json
import math
from pathlib import Path

POSITIONS={'CAM_L0':(0,0),'CAM_F0':(0,1),'CAM_R0':(0,2),'CAM_L1':(1,0),
           'CAM_R1':(1,2),'CAM_L2':(2,0),'CAM_B0':(2,1),'CAM_R2':(2,2)}


def save(path,value):
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def steps(value):
    if not isinstance(value,list) or not value or any(type(i) is not int or not 0<=i<12 for i in value):
        raise ValueError('steps must be a nonempty list of integers 0..11')
    return sorted(set(value))


class Evidence:
    def __init__(self,project,cfg,row,frames,out):
        self.project,self.cfg,self.row,self.frames,self.out=project,cfg,row,frames,out
        self.root=Path(cfg['result_root']);self.images={};self.data=None;self.numeric_source=None

    def prepare_images(self):
        import cv2
        from PIL import Image,ImageDraw
        directory=self.out/'images';directory.mkdir(exist_ok=True)
        cap=cv2.VideoCapture(str(self.root/self.row['bev_video']))
        try:
            if abs(cap.get(cv2.CAP_PROP_FPS)-2)>1e-4:raise ValueError('Expected 2fps corrected BEV')
            for i,frame in enumerate(self.frames):
                board=None;tile_height=None
                for camera,(r,c) in POSITIONS.items():
                    base=(self.root/'WE_output/openscene_format/sensor_blobs').resolve()
                    source=(base/frame['cams'][camera]['data_path']).resolve()
                    if not source.is_relative_to(base):raise ValueError('Image outside archive')
                    with Image.open(source) as raw:
                        im=raw.convert('RGB');im.thumbnail((640,640))
                    self._image(camera,i,im,directory)
                    if tile_height is None:tile_height=round(im.height*320/im.width)
                    tile=im.copy();tile.thumbnail((320,tile_height))
                    if board is None:board=Image.new('RGB',(960,tile_height*3),'#191919')
                    x,y=c*320,r*tile_height;board.paste(tile,(x,y))
                    draw=ImageDraw.Draw(board);draw.rectangle((x,y,x+100,y+18),fill='black');draw.text((x+4,y+4),camera,fill='white')
                self._image('overview',i,board,directory)
                ok,bgr=cap.read()
                if not ok:raise ValueError('Missing corrected BEV frame')
                bev=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));bev.thumbnail((640,640))
                self._image('bev',i,bev,directory)
        finally:cap.release()
        save(self.out/'images.json',self.images)

    def _image(self,camera,step,im,directory):
        key=f'I_{camera}_{step:02d}';path=directory/(key+'.jpg');im.save(path,quality=80)
        self.images[key]={'id':key,'camera':camera,'step':step,'video_t_s':step*.5,'control_t_s':(step-3)*.5,
                          'file':str(path),'sha256':sha(path),'size':list(im.size)}

    def select_images(self,requests):
        if not isinstance(requests,list) or not requests:raise ValueError('Nonempty requests list required')
        ids=[]
        for request in requests:
            if not isinstance(request,dict):raise ValueError('Each image request must be an object')
            camera=request.get('camera')
            if camera not in (*POSITIONS,'bev'):raise ValueError('camera must be an actual camera name or bev')
            for step in steps(request.get('steps')):
                key=f'I_{camera}_{step:02d}'
                if key not in ids:ids.append(key)
        return [self.images[k] for k in ids]

    def content(self,images):
        result=[]
        for im in images:
            result.extend([{'type':'text','text':json.dumps(im,ensure_ascii=False)},
                {'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(Path(im['file']).read_bytes()).decode()}}])
        return result

    def numbers(self):
        if self.data is not None:return self.data
        cached=self.project/self.cfg.get('numeric_cache_dir','runs/mechanism_all_v2/scenes')/self.row['token']
        valid=(cached/'inputs.json').is_file() and (cached/'numeric_evidence.json').is_file()
        if valid:
            inputs=json.loads((cached/'inputs.json').read_text());fingerprint=inputs['source_fingerprints']
            metadata=(self.root/self.row['metadata']).resolve()
            valid=inputs.get('token')==self.row['token'] and fingerprint.get(str(metadata))==sha(metadata)
            for name,expected in fingerprint.items():
                if name=='map_file_stats':
                    valid &= all(Path(p).is_file() and [Path(p).stat().st_size,Path(p).stat().st_mtime_ns]==v for p,v in expected.items())
                else:
                    path=Path(name)
                    if not path.exists() and path.name=='analyze_scene.py':
                        path=self.project/'archive/workflow_v2/scripts/analyze_scene.py'
                    valid &= path.is_file() and sha(path)==expected
        if valid:
            source=cached/'numeric_evidence.json'
        else:
            from numeric import numeric
            numeric(Path(self.cfg['simulation_root']),self.root,self.frames,self.row['token'],self.out)
            source=self.out/'numeric_evidence.json'
        self.data=json.loads(source.read_text())
        self.numeric_source={'file':str(source),'sha256':sha(source),'reused':bool(valid),
            'scope':'Simulator recorded annotations, not model perception. Geometry is supplemental; official scores remain correct.'}
        return self.data

    def query(self,request):
        if not isinstance(request,dict):raise ValueError('Each query must be an object')
        kind=request.get('kind');selected=steps(request.get('steps'))
        if kind not in ('ego','actors','boundary','plans'):raise ValueError('kind must be ego, actors, boundary or plans')
        data=self.numbers();frames={f['step']:f for f in data['frames']};rows=[]
        if kind=='actors':
            token=request.get('track_token');limit=request.get('nearest',3)
            if type(limit) is not int or not 1<=limit<=20:raise ValueError('nearest must be 1..20; use track_token for an identified actor')
            all_rows={(a['track_token'],a['step']):a for a in data['actor_measurements']}
            for step in selected:
                actors=[a for a in data['actor_measurements'] if a['step']==step and (token is None or a['track_token']==token)]
                for a in sorted(actors,key=lambda a:a['distance_m'])[:limit if token is None else len(actors)]:
                    row=dict(a);previous=all_rows.get((a['track_token'],step-1))
                    row['lateral_axis_gap_m']=None;row['longitudinal_axis_gap_m']=None
                    if len(self.frames)>step:
                        anns=self.frames[step]['anns'];tokens=anns.get('track_tokens',anns.get('instance_tokens',[]))
                        boxes={str(t):b for t,b in zip(tokens,anns['gt_boxes'])}
                        box=boxes.get(a['track_token'])
                        if box is not None:
                            half_x=(abs(math.cos(box[6]))*box[3]+abs(math.sin(box[6]))*box[4])/2
                            half_y=(abs(math.sin(box[6]))*box[3]+abs(math.cos(box[6]))*box[4])/2
                            vehicle=data['vehicle_parameters'];front=vehicle['rear_axle_to_center']+vehicle['length']/2;rear=front-vehicle['length']
                            row['lateral_axis_gap_m']=float(abs(box[1])-half_y-vehicle['width']/2)
                            row['longitudinal_axis_gap_m']=float(max(box[0]-half_x-front,rear-box[0]-half_x))
                    row['world_speed_fd_mps']=None;row['world_longitudinal_speed_fd_mps']=None;row['world_speed_change_fd_mps2']=None
                    if previous:
                        vx=(a['center_x_m']-previous['center_x_m'])/.5;vy=(a['center_y_m']-previous['center_y_m'])/.5
                        row['world_speed_fd_mps']=math.hypot(vx,vy)
                        row['world_longitudinal_speed_fd_mps']=vx*math.cos(frames[step]['heading'])+vy*math.sin(frames[step]['heading'])
                        earlier=all_rows.get((a['track_token'],step-2))
                        if earlier:
                            previous_speed=math.hypot(previous['center_x_m']-earlier['center_x_m'],previous['center_y_m']-earlier['center_y_m'])/.5
                            row['world_speed_change_fd_mps2']=(row['world_speed_fd_mps']-previous_speed)/.5
                    rows.append(row)
        elif kind=='plans':
            keys=('record_step','input_frame','plan_idx','executed','planning_horizon_s','points_rear_global')
            rows=[{k:p[k] for k in keys} for p in data['plans'] if p['input_frame'] in selected]
        else:
            keys=('speed_mps','accel_forward_mps2','rear_x','rear_y','heading') if kind=='ego' else ('min_corner_margin_m','outside_corners')
            rows=[{'step':s,**{k:frames[s][k] for k in keys}} for s in selected]
        scope={'ego':'自车记录运动状态，不等于刹车/油门控制指令',
               'actors':'对象几何和运动；轴投影间距为负只说明该轴重叠，不独自证明接触或官方对象身份',
               'boundary':'仅针对可行驶区域边界，不是侧车/墙体间距；不能用DAC=1或正边界余量单独否定侧向空间类候选',
               'plans':'所选规划轨迹，不提供候选概率或内部选择原因'}[kind]
        return {'request':request,'rows':rows,'source':self.numeric_source,'sampling_seconds':.5,'evidence_scope':scope,
                'units':'positions/gaps=m, speed=m/s, acceleration=m/s^2, heading=rad, intersection=m^2',
                'limits':'No rows means unavailable, not zero. Actor world velocity is past-to-current finite difference; relative gap rate is not actor speed. Plan input_frame=s precedes execution at s+1; last unused plan is not executed.'}
