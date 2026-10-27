"""Official outcome -> VLM image investigation -> VLM numeric verification -> summary."""
import argparse,csv,json,pickle,sys,time
import os
from collections import Counter
from pathlib import Path

PROJECT=Path(__file__).resolve().parent
sys.path.insert(0,str(PROJECT/'scripts'))
from evidence import Evidence,save
from official_results import load_official
from vlm_review import Client,run


def prepare_attempt(directory,token,resume,limit):
    link=directory/'scenes'/token
    if link.exists():
        if not resume:raise FileExistsError(link)
        status=link/'status.json'
        if status.is_file() and json.loads(status.read_text())['state']=='completed':return None
        if not link.is_symlink():raise ValueError('Resume requires attempt directories; keep legacy runs unchanged')
    attempts=directory/'attempts'/token
    attempts.mkdir(parents=True,exist_ok=True)
    number=len(list(attempts.glob('attempt_*')))+1
    if number>limit:return None
    out=attempts/f'attempt_{number:02d}';out.mkdir()
    link.parent.mkdir(parents=True,exist_ok=True)
    temporary=link.with_name(link.name+'.next')
    if temporary.is_symlink():temporary.unlink()
    temporary.symlink_to(out.resolve(),target_is_directory=True)
    temporary.replace(link)
    save(out/'status.json',{'state':'running','attempt':number,'pid':os.getpid(),'started_unix':time.time()})
    return out


def summarize(directory,rows):
    completed=[];failed=[]
    for row in rows:
        out=directory/'scenes'/row['token']
        if not (out/'status.json').is_file():continue
        status=json.loads((out/'status.json').read_text())
        if status['state']=='completed':completed.append(json.loads((out/'result.json').read_text()))
        else:failed.append(row['token'])
    failed_total=sum(float(r['no_at_fault_collisions'])<1 or float(r['drivable_area_compliance'])<1 for r in rows)
    summary={'official_scene_count':len(rows),'official_failure_count':failed_total,'vlm_cases_completed':len(completed),
             'supported_behavior_reasons':dict(Counter(r['reason_category'] for r in completed if r['status']=='supported')),
             'undetermined':sum(r['status']=='undetermined' for r in completed),'failed_or_incomplete':failed,
             'note':'Only completed VLM investigations count; historical numeric-only checks do not count as completed analysis.'}
    save(directory/'summary.json',summary)
    lines=['# 失败行为原因汇总','',f"VLM分析完成 {len(completed)} 例；官方失败共 {failed_total} 例。",'']
    for r in completed:
        lines.extend([f"- [{r['token']}](scenes/{r['token']}/report.md)：{r['failure_reason']}"])
    lines+=['',f"待确认：{summary['undetermined']}例；调用失败或分析未完成：{len(failed)}例。"]
    (directory/'report.md').write_text('\n'.join(lines)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['scene','batch','report']);p.add_argument('--config',type=Path,default=PROJECT/'configs/complete.json')
    p.add_argument('--token');p.add_argument('--tokens',nargs='+');p.add_argument('--run-name',default='vlm_behavior_v3')
    p.add_argument('--resume',action='store_true');p.add_argument('--attempts',type=int,default=1)
    args=p.parse_args();cfg=json.loads(args.config.read_text());root=Path(cfg['result_root'])
    if not 1<=args.attempts<=3:p.error('--attempts must be 1..3')
    if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_name):p.error('Invalid run-name')
    rows=list(csv.DictReader((root/'scene_index.csv').open()))
    with (root/'WE_output/openscene_format/all_scenes_pdm_averages_R.csv').open() as stream:
        official_rows={r['token']:r for r in csv.DictReader(stream) if r['token']!='overall_average'}
    if len({r['token'] for r in rows})!=len(rows) or set(official_rows)!={r['token'] for r in rows}:raise ValueError('Invalid official scene inventory')
    rows=[{**r,**official_rows[r['token']]} for r in rows]
    directory=PROJECT/'runs'/args.run_name;directory.mkdir(parents=True,exist_ok=True)
    if args.command=='report':print(json.dumps(summarize(directory,rows),ensure_ascii=False,indent=2));return
    requested=[args.token] if args.command=='scene' else args.tokens
    if args.command=='scene' and not args.token:p.error('--token required')
    selected=[]
    for token in requested or []:
        matches=[r for r in rows if r['token']==token or r['token'].endswith('-'+token)]
        if len(matches)!=1:p.error('Token must match exactly one scene')
        if matches[0] not in selected:selected.append(matches[0])
    if not requested:selected=[r for r in rows if float(r['no_at_fault_collisions'])<1 or float(r['drivable_area_compliance'])<1]
    if any(r['token'] not in cfg['vlm_allowed_tokens'] for r in selected):p.error('Selected token is outside configured external-image authorization')
    secret_file=PROJECT/'.secrets/vlm.json'
    if secret_file.stat().st_mode & 0o077:raise ValueError('Secret file must have mode 0600')
    secret=json.loads(secret_file.read_text())
    for row in selected:
        out=prepare_attempt(directory,row['token'],args.resume,args.attempts)
        if out is None:continue
        started=time.monotonic()
        status={'state':'failed'};prepared=None
        try:
            official=load_official(root,row);save(out/'official_results.json',official)
            metadata=(root/row['metadata']).resolve()
            if not metadata.is_relative_to(root.resolve()):raise ValueError('Metadata outside archive')
            frames=sorted(pickle.load(metadata.open('rb')),key=lambda f:f['frame_idx'])
            if len(frames)!=12:raise ValueError('Expected 12 archive frames')
            evidence=Evidence(PROJECT,cfg,row,frames,out);evidence.prepare_images();evidence.numbers()
            prepared=time.monotonic()
            print('PREPARED',row['token'],flush=True)
            result=run(Client(secret,out),evidence,official,out,cfg.get('max_turns',12),cfg.get('workers',2))
            result['official_metrics']=official['metrics'];save(out/'result.json',result)
            (out/'report.md').write_text('\n'.join(['# 失败行为原因','',row['token'],'',result['failure_reason'],'',
                '## 现象',json.dumps(result['observations'],ensure_ascii=False,indent=2),'','## 候选原因及核验',
                json.dumps(result['candidate_assessment'],ensure_ascii=False,indent=2),'','## 不确定之处',str(result.get('uncertainty',''))])+'\n')
            status={'state':'completed','model':secret['model'],'analysis_level':'driving_behavior'}
        except Exception as exc:
            status={'state':'failed','error_type':type(exc).__name__,'message':str(exc)}
        finally:
            status['total_seconds']=round(time.monotonic()-started,2)
            if prepared is not None:
                status['preprocess_seconds']=round(prepared-started,2)
                status['vlm_seconds']=round(time.monotonic()-prepared,2)
            save(out/'status.json',status)
            print(json.dumps(status,ensure_ascii=False),flush=True);summarize(directory,rows)
    if any(not (directory/'scenes'/r['token']/'status.json').is_file() or
           json.loads((directory/'scenes'/r['token']/'status.json').read_text())['state']!='completed' for r in selected):raise SystemExit(2)


if __name__=='__main__':main()
