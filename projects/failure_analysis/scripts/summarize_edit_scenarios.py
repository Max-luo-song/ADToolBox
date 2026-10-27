"""Summarize existing case results into scene families useful for scenario editing."""
import argparse
import hashlib
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evidence import save
from vlm_review import Client

PROJECT=Path(__file__).resolve().parents[1]
RULES='''你的任务是总结已有案例中可用于后续场景编辑的典型失败场景，不重新诊断模型内部原因。
只用提供的现象、原因和不确定性。原结果也可能过度推断，不能将其推测升级为事实。
每类按固定顺序写四句话：自车正在做什么；环境及其他参与者如何；自车的可观察错误行为；应有的驾驶响应。
场景名称只写“自车动作+关键环境”，不含未减速、间距不足、导致、越界、追尾等错误或结果。例如“直行接近同车道低速或静止前车场景”“路口转弯时与他车交互场景”。示例仅说明风格，不作为证据。
ego_action写自车动作；environment写道路、参与者及相对运动；wrong_behavior写记录支持的不当响应；expected_behavior写以“自车应”开头的测试目标。
场景归并依据自车动作与关键环境的共同组合，不只依据碰撞/越界。没有直行证据就写“前进接近”，没有他车干扰证据就不能写他车干扰。
“有他车干扰”要说明他车从哪里来、如何运动；若并非整个组都有该条件，不推广至整组。不能把未知对象、信号、墙距补成事实。
错误行为限未及时减速、持续接近、横向偏移或越出边界等可见事实，不猜未识别、感知失败、控制失效、避让动机。
没有因果证据时写“持续加速且车角越界”，不能写“加速导致越界”。不能把部分成员的动作或环境写成整组共有；可写“部分案例中”。
应有响应是后续仿真的测试目标，不是已经证明有效的修复方案。结合场景写提前减速、保持间距、让行或停车等，不编造通行权与安全阈值。
正文只输出上述四句话，不添加编辑参数、指标定义或模块分析。应有响应不能当作直接编辑自车轨迹的指令。
没有明确触发情境的案例进U0，不强行归因。每例只有一个主场景组；复合条件可在依据中保留。
U0明确写动作/环境/错误行为哪些未知，expected_behavior写“暂不确定，不直接生成编辑任务”，不能凑造标准响应。
每类四句话尽量简短，约120至180字，保留必要条件，不设硬字数上限。写共通情境，少用重复的“部分案例”。正文不用“导致”，以“并出现”描述共现结果。不要写模块猜测、指标科普或大段证据。'''


def validate_description(g):
    fields=('name','ego_action','environment','wrong_behavior','expected_behavior')
    if any(not isinstance(g.get(k),str) or not g[k].strip() for k in fields):raise ValueError('Provide five nonempty scene description fields')
    if any(word in g['name'] for word in ('越界','追尾','不足','未减速','未停车','导致','错误')):raise ValueError('Title must contain only ego action and environment, not error/outcome')
    paragraph=''.join(g[k] for k in fields[1:])
    if any(word in paragraph for word in ('导致','控制不足','控制失效','感知失败','未识别')):raise ValueError('Describe observed motion and co-occurring outcomes; omit unsupported module/cause claims')
    if not g['expected_behavior'].startswith('自车应'):raise ValueError('Expected response starts with 自车应')


def ask(client,messages,name,validate=None):
    for attempt in range(3):
        call=name if attempt==0 else f'{name}_correction_{attempt}'
        path=client.out/(call+'.json')
        answer=json.loads(path.read_text()) if path.exists() else client.ask(messages,call)
        try:
            if validate is not None:validate(answer)
            return answer
        except (ValueError,KeyError,TypeError) as exc:
            if attempt==2:raise
            messages=messages+[{'role':'assistant','content':json.dumps(answer,ensure_ascii=False)},
                {'role':'user','content':'修正本次总结的格式或措辞，不改变成员、不补造证据：'+str(exc)}]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name',required=True)
    parser.add_argument('--output-name',required=True)
    parser.add_argument('--groups-from',type=Path,help='Keep existing memberships and rewrite descriptions only')
    args=parser.parse_args()
    for name in (args.run_name,args.output_name):
        if not name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):parser.error('Invalid name')
    root=PROJECT/'runs'/args.run_name;out=root/args.output_name;out.mkdir(exist_ok=True)
    cases=[]
    for i,token in enumerate(json.loads((root/'selected_tokens.json').read_text()),1):
        p=root/'scenes'/token;r=json.loads((p/'result.json').read_text())
        if json.loads((p/'status.json').read_text())['state']!='completed':raise ValueError('Finish all cases first')
        visual=json.loads((p/'visual_hypotheses.json').read_text())
        cases.append({'id':f'C{i:03d}','token':token,'reason':r['failure_reason'],
            'visual_observations':visual['observations'],
            'observations':r['observations'],'uncertainty':r.get('uncertainty',''),'reason_status':r['status']})
    snapshot={'cases':cases,'rules':RULES,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    existing=None
    if args.groups_from:
        source=args.groups_from if args.groups_from.is_absolute() else PROJECT/args.groups_from
        existing=json.loads(source.read_text())
        if Counter(c['token'] for g in existing['groups'] for c in g['cases'])!=Counter(c['token'] for c in cases):
            raise ValueError('Existing memberships must cover the same cases exactly once')
        snapshot['group_source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    snapshot_path=out/'input.json'
    if snapshot_path.exists() and json.loads(snapshot_path.read_text())!=snapshot:raise ValueError('Inputs changed; use a new output-name')
    save(snapshot_path,snapshot)
    secret_path=PROJECT/'.secrets/vlm.json'
    if secret_path.stat().st_mode & 0o077:raise ValueError('Secret file must have mode 0600')
    client=Client(json.loads(secret_path.read_text()),out)
    if existing is not None:
        by_token={c['token']:c for c in cases}
        def rewrite(g):
            if g['id']=='U0':return {**g,'name':'关键场景信息不足','ego_action':'部分案例缺少明确的自车动作描述。','environment':'关键参与者或交互条件尚不明确。','wrong_behavior':'无法据此形成一致的错误行为描述。','expected_behavior':'暂不确定，不直接生成编辑任务。'}
            task={'task':'保持本组成员不变，依据它们的共同情境重新命名，并生成动作、环境、错误行为、应有响应四句话。组内条件不一致时适当概括或注明部分案例，不强加直行/转弯或他车干扰。',
                  'previous_name':g['name'],'cases':[by_token[c['token']] for c in g['cases']],
                  'output':{'name':'动作+环境+场景，不写错误或结果','ego_action':'自车正在做什么。','environment':'环境如何。','wrong_behavior':'自车错误行为。','expected_behavior':'自车应如何响应。'}}
            description=ask(client,[{'role':'system','content':RULES},{'role':'user','content':json.dumps(task,ensure_ascii=False)}],'describe_'+g['id'],validate_description)
            return {**g,**{k:description[k] for k in ('name','ego_action','environment','wrong_behavior','expected_behavior')}}
        with ThreadPoolExecutor(max_workers=2) as pool:groups=list(pool.map(rewrite,existing['groups']))
        assignments=[{'case':by_token[c['token']]['id'],'group':g['id'],'basis':c['basis']} for g in groups for c in g['cases']]
        write_summary(out,cases,groups,assignments)
        return
    brief=[{**c,'visual_observations':c['visual_observations'][:3],'observations':c['observations'][:2]} for c in cases]
    task={'task':'阅读全部案例，提炼5至7个有实际案例支持的主要场景，另设U0信息不足。每类用自然连贯的四句话，按自车动作、环境、错误行为、应有响应的顺序。不要输出每例分组。',
          'cases':brief,'output':{'groups':[{'id':'G1','name':'自车动作+关键环境+场景',
          'ego_action':'自车正在进行的动作。','environment':'道路及其他参与者的位置和相对运动。',
          'wrong_behavior':'自车有证据支持的错误行为。','expected_behavior':'自车应采取的适当响应，作为测试目标。'}],
          'note':'U0也包含在groups中，明确缺少什么信息，不编造动作或响应。'}}
    ontology=ask(client,[{'role':'system','content':RULES},{'role':'user','content':json.dumps(task,ensure_ascii=False)}],'taxonomy',
                 lambda value:[validate_description(g) for g in value['groups'] if g['id']!='U0'])
    groups=ontology['groups'];group_ids={g['id'] for g in groups}
    if len(groups)!=len(group_ids) or 'U0' not in group_ids or not 6<=len(groups)<=8:raise ValueError('Expected 5..7 scene families and U0')
    if any(not isinstance(g.get(k),str) or not g[k].strip() for g in groups for k in ('name','ego_action','environment','wrong_behavior','expected_behavior')):
        raise ValueError('Each scene needs action, environment, wrong behavior and expected response')
    def classify(job):
        number,batch=job
        prompt={'task':'按自车动作与关键环境组合，将每例归入一个主场景组。直行/转弯、切入/横穿不能混淆；没有他车干扰证据不能归入依赖该条件的组。缺少符合组定义的关键情境则U0。依据短于40字，不凭侧向间距标签猜窄路或停靠车辆。',
                'groups':groups,'cases':batch,'output':{'assignments':[{'case':'C001','group':'G1','basis':'归组所依据的明确场景条件，简短'}]}}
        answer=ask(client,[{'role':'system','content':RULES},{'role':'user','content':json.dumps(prompt,ensure_ascii=False)}],f'assign_{number:02d}')
        assignments=answer['assignments']
        if Counter(a['case'] for a in assignments)!=Counter(c['id'] for c in batch):raise ValueError('Each case must occur exactly once')
        if any(a['group'] not in group_ids or not a.get('basis') for a in assignments):raise ValueError('Invalid assignment')
        return assignments
    jobs=list(enumerate([cases[i:i+16] for i in range(0,len(cases),16)]))
    with ThreadPoolExecutor(max_workers=2) as pool:assignments=[a for batch in pool.map(classify,jobs) for a in batch]
    write_summary(out,cases,groups,assignments)


def write_summary(out,cases,groups,assignments):
    index={c['id']:c for c in cases};counts=Counter(a['group'] for a in assignments)
    result={'source':'external_kimi_summary_of_saved_results','scene_count':len(cases),'groups':[],
            'note':'场景情境归并，不是已验证的因果统计；不改写逐例原因。一个案例只计入一个主组。'}
    lines=['# 典型失败场景总结','',f'共{len(cases)}例，每例归入一个主场景组。以下由Kimi归并已有结果；应有响应是后续测试目标。','']
    for g in sorted(groups,key=lambda g:(g['id']=='U0',-counts[g['id']])):
        members=[{**a,'token':index[a['case']]['token'],'reason_status':index[a['case']]['reason_status']} for a in assignments if a['group']==g['id']]
        result['groups'].append({**g,'count':len(members),'cases':members})
        if not members:continue
        paragraph=''.join(g[k].strip().rstrip('。')+'。' for k in ('ego_action','environment','wrong_behavior','expected_behavior'))
        lines += [f"## {g['name']}（{len(members)}例）",'',paragraph,'']
    lines+=['逐例归组依据与案例编号另存groups.json；原始归因不变。应有响应不等于直接设定自车轨迹。']
    save(out/'groups.json',result);(out/'summary.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'cases':len(cases),'counts':dict(counts),'summary':str(out/'summary.md')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
