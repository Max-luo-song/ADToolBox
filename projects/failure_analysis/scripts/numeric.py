"""Compute recorded geometry and trajectories; no behavioral diagnosis or VLM calls."""
import csv,json,sys
import numpy as np
from shapely.geometry import Polygon,Point
from shapely.ops import unary_union

def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def rot(h):return np.array([[np.cos(h),-np.sin(h)],[np.sin(h),np.cos(h)]])

def corners(center,length,width,heading):
    return np.array([[length/2,width/2],[length/2,-width/2],[-length/2,-width/2],[-length/2,width/2]])@rot(heading).T+center

def read_rows(path):
    with path.open() as f:return list(csv.DictReader(f))

def map_union(we,frames):
    sys.path.insert(0,str(we/'projects/SimEngine'))
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer as Layer
    from nuplan.common.actor_state.state_representation import Point2D
    api=get_maps_api(str(we/'data/raw/nuplan/dataset/maps'),'nuplan-maps-v1.0',frames[3]['map_location'])
    m=np.asarray(frames[3]['ego2global']);h=np.arctan2(m[1,0],m[0,0]);c=m[:2,3]+np.array([np.cos(h),np.sin(h)])*1.461
    objects=api.get_proximal_map_objects(Point2D(*c),100,[Layer.ROADBLOCK,Layer.ROADBLOCK_CONNECTOR,Layer.INTERSECTION,Layer.CARPARK_AREA])
    polys=[]
    for layer in (Layer.ROADBLOCK,Layer.ROADBLOCK_CONNECTOR):
        for road in objects[layer]:
            if layer==Layer.ROADBLOCK:polys.append(road.polygon)
            polys.extend(x.polygon for x in road.interior_edges)
    for layer in (Layer.INTERSECTION,Layer.CARPARK_AREA):polys.extend(x.polygon for x in objects[layer])
    if not polys:raise ValueError('No official drivable polygons')
    return unary_union(polys)

def numeric(we,root,frames,token,out):
    # These are the same Pacifica rear-axle dimensions used by MetricManager.
    sys.path.insert(0,str(we/'projects/SimEngine'))
    from worldengine.components.agents.vehicle_model.pacifica_vehicle import get_pacifica_parameters
    params=get_pacifica_parameters();shift=(params.front_length-params.rear_length)/2
    poses=[]
    for f in frames:
        m=np.asarray(f['ego2global']);poses.append((m[:2,3],float(np.arctan2(m[1,0],m[0,0]))))
    area=map_union(we,frames)
    all_frames=[];actor_rows=[]
    for i,(f,(rear,h)) in enumerate(zip(frames,poses)):
        center=rear+np.array([np.cos(h),np.sin(h)])*shift
        xy=corners(center,params.length,params.width,h);ego=Polygon(xy)
        margins=[Point(p).distance(area.boundary)*(1 if area.contains(Point(p)) else -1) for p in xy]
        anns=f['anns'];boxes=np.asarray(anns['gt_boxes']).reshape(-1,7)
        tracks=anns.get('track_tokens',anns['instance_tokens']);names=anns['gt_names']
        nearest=None;overlaps=[]
        for j,b in enumerate(boxes):
            c=b[:2]@rot(h).T+rear
            actor=Polygon(corners(c,float(b[3]),float(b[4]),float(b[6])+h))
            distance=ego.distance(actor);intersection=ego.intersection(actor).area
            rec={'step':i,'track_token':str(tracks[j]),'class':str(names[j]),'distance_m':float(distance),
                 'intersection_area_m2':float(intersection),'relative_x_m':float(b[0]),'relative_y_m':float(b[1]),
                 'center_x_m':float(c[0]),'center_y_m':float(c[1])}
            actor_rows.append(rec)
            if nearest is None or distance<nearest['distance_m']:nearest=rec
            if intersection>1e-6:overlaps.append(rec)
        state=np.asarray(f['ego_dynamic_state'],dtype=float)
        rec={'step':i,'control_time_s':(i-3)*.5,'rear_x':float(rear[0]),'rear_y':float(rear[1]),'heading':h,
             'speed_mps':float(np.linalg.norm(state[:2])),'accel_forward_mps2':float(state[2]),
             'corner_margin_m':list(map(float,margins)),'min_corner_margin_m':float(min(margins)),
             'outside_corners':[name for name,m in zip(['front_left','front_right','rear_right','rear_left'],margins) if m<0],
             'nearest_actor':nearest,'overlap_candidates':overlaps}
        all_frames.append(rec)
    vocab=np.load(we/'data/alg_engine/test_8192_kmeans.npy')
    selected=[r for r in read_rows(root/'plan_traj/plan_idx.csv') if r['prefix']==token.rsplit('-',1)[-1]]
    plans=[]
    for r in selected:
        step=int(r['step']);idx=int(r['plan_idx']);base=step-1
        if base>=len(poses):continue
        rear,h=poses[base];raw=vocab[idx];xy=raw[:,:2]@rot(h).T+rear
        rec={'record_step':step,'input_frame':base,'plan_idx':idx,'executed':step<len(poses),
             'first_0_5s_displacement_m':float(np.linalg.norm(raw[4,:2])),
             'planning_horizon_s':4.0,'points_rear_global':xy.tolist(),
             'execution_position_error_m':float(np.linalg.norm(xy[4]-poses[step][0])) if step<len(poses) else None}
        plans.append(rec)
    used=[p for p in plans if p['executed']]
    if not used:raise ValueError('No executed plans available for supplemental comparison')
    control=all_frames[3:]
    first_out=next((r['step'] for r in control if r['outside_corners']),None)
    first_overlap=next((r['step'] for r in control if r['overlap_candidates']),None)
    track_groups={}
    for r in actor_rows:track_groups.setdefault(r['track_token'],[]).append(r)
    for seq in track_groups.values():
        for a,b in zip(seq,seq[1:]):
            if b['step']-a['step']==1:b['relative_longitudinal_velocity_fd_mps']=(b['relative_x_m']-a['relative_x_m'])/.5
    findings={'first_sampled_corner_outside_step':first_out,'first_sampled_overlap_candidate_step':first_overlap,
              'min_corner_margin_m':min(r['min_corner_margin_m'] for r in control),
              'max_execution_position_error_m':max(p['execution_position_error_m'] for p in used),
              'control_distance_m':float(sum(np.linalg.norm(poses[i][0]-poses[i-1][0]) for i in range(4,len(poses)))),
              'minimum_actor_distance_m':min((r['distance_m'] for r in actor_rows if r['step']>=3),default=None),
              'interpretation':'Polygon overlap is a contact candidate, not an official at-fault classification. 0.5s sampling can miss contacts. Execution matching does not explain why the model selected a plan.',
              'map_method':'nuPlan official polygons and MetricManager 100m query at control start; corner containment; numerical boundary equality requires scorer re-evaluation.'}
    dump(out/'numeric_evidence.json',{'findings':findings,'frames':all_frames,'plans':plans,'actor_measurements':actor_rows,
         'vehicle_parameters':{'length':params.length,'width':params.width,'rear_axle_to_center':shift}})
    for name,rows in [('actors.csv',actor_rows),('execution.csv',[{k:v for k,v in p.items() if k!='points_rear_global'} for p in plans])]:
        with (out/name).open('w',newline='') as f:
            fields=sorted(set().union(*(r.keys() for r in rows))) if rows else ['step','track_token','distance_m','intersection_area_m2']
            w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    return findings
