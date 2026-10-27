"""VLM-driven observation, candidate generation, and on-demand numeric verification."""
import hashlib
import json
import socket
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from evidence import POSITIONS, save

RULES='''官方指标是正确事实，不重新裁决NC/DAC。目标是驾驶行为原因（如未及时减速、侧向间距不足），不追究模型内部根因。
每步0.5秒，step3开始控制。先描述现象，提出若干合理候选；按需要继续看相机或BEV，然后主动查询数值比较候选。
刹车灯不能证明急刹，画面模糊不能证明物理冲击。BEV是256米固定世界视角，640像素为2.5px/m：白自车、橙周车、绿规划、黄历史、蓝路线。
几何接触对象只是候选，NC不提供对象ID；first_violation_step是NC=0或DAC=0的最早者，双失败不等于同时发生；EP=0不证明静止。
TTC等字段是评分项，不是原始秒数。DAC与boundary只能核对区域越界，不能单独排除侧车间距不足、墙体接触或侧向空间影响。
排除一个候选需要直接相关证据；没有支持不等于被否定，间接或不完整证据应标uncertain。几何重叠增大不等于冲击程度增大，不使用“无任何不确定性”的绝对结论。
候选不一定互斥，可以同时得到支持；区分主要行为原因与促成因素，不为了选一个原因而强行否定其余候选。
核验必须回答每个候选原先needs_evidence提出的问题。逐项写出判定需要的观测、实际返回的观测、仍缺少的证据。
rejected必须有直接矛盾观测；没有量到墙体/侧车间距时，DAC=1只能排除越界，侧向空间候选仍是uncertain。EP=0不能解释停止原因。
运动记录只能说明速度变化，不能写未采取制动指令或未识别目标。几何重叠对象只能称接触候选，不能写官方确认与该对象碰撞。
前车减速必须查同一track_token的世界速度历史，不能把相对间距变化当前车速度。只引用工具实际返回值，缺失不当成0。
不能从图片直接猜track_token，先查询actors列表再按位置匹配；匹配不唯一就保留不确定。
有支持、被否定、证据不足分开。选最有证据的行为原因；无法区分就待确认，不强行挑一个。不要推断未提供的感知错误或执行故障。'''

SYSTEM='''你分析可观察的驾驶行为。所有输出只描述图像、记录与官方指标能证明的事实。
没有感知输出或控制指令：不能写未识别、感知失败、未采取制动等；写速度是否下降、间距如何变化。
EP=0不能说明停止或停止原因。NC只能确认官方碰撞评分，几何对象只能称接触候选。
轴投影重叠不等于间距不足。DAC与道路边界不能否定墙体/侧车空间影响。
最终原因只写驾驶行为，例如“跟车时未及时减速，持续加速接近低速前车”；不要添加内部原因。'''


class Client:
    def __init__(self,secret,out):
        self.secret,self.out=secret,out

    def ask(self,messages,name):
        payload={'model':self.secret['model'],'thinking':{'type':'disabled'},'max_tokens':1800,'messages':messages}
        encoded=json.dumps(payload,ensure_ascii=False).encode()
        save(self.out/(name+'_request.json'),{'sha256':hashlib.sha256(encoded).hexdigest(),'model':self.secret['model']})
        attempts=[]
        for attempt in range(2):
            started=time.monotonic()
            try:
                request=urllib.request.Request(self.secret['endpoint'],data=encoded,
                    headers={'Authorization':'Bearer '+self.secret['api_key'],'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=30) as response:raw=json.load(response)
                save(self.out/(name+'_raw.json'),raw)
                choice=raw['choices'][0]
                if choice.get('finish_reason')=='length':raise ValueError('Truncated model response')
                text=choice['message']['content'].strip()
                if text.startswith('```'):text='\n'.join(text.splitlines()[1:-1])
                answer=json.loads(text)
                if not isinstance(answer,dict):raise ValueError('Expected JSON object')
                attempts.append({'state':'completed','seconds':round(time.monotonic()-started,2),'usage':raw.get('usage')})
                save(self.out/(name+'.json'),answer)
                print('VLM',name,attempts[-1]['seconds'],'seconds',flush=True)
                return answer
            except (TimeoutError,socket.timeout,urllib.error.URLError) as exc:
                attempts.append({'state':'failed','seconds':round(time.monotonic()-started,2),'error_type':type(exc).__name__})
                retryable=not isinstance(exc,urllib.error.HTTPError) or exc.code in (429,500,502,503,504)
                if attempt or not retryable:raise
            except Exception as exc:
                attempts.append({'state':'failed','seconds':round(time.monotonic()-started,2),'error_type':type(exc).__name__})
                raise
            finally:
                save(self.out/(name+'_status.json'),attempts)


def inspect_images(client,evidence,images,purpose,name,workers):
    def group(job):
        i,selected=job
        prompt={'task':'只观察本组图片，简洁中文JSON：observations、uncertainties。注明相机、step和图片id；最多3条，不给最终原因。',
                'purpose':purpose,'rules':RULES}
        content=[{'type':'text','text':json.dumps(prompt,ensure_ascii=False)}]+evidence.content(selected)
        review=client.ask([{'role':'system','content':SYSTEM},{'role':'user','content':content}],f'{name}_{i:02d}')
        return {'image_ids':[im['id'] for im in selected],'review':review}
    jobs=list(enumerate([images[i:i+3] for i in range(0,len(images),3)]))
    with ThreadPoolExecutor(max_workers=workers) as pool:return list(pool.map(group,jobs))


def validate_final(action,candidates,numeric_ids,image_ids=frozenset()):
    assessed=action.get('candidate_assessment')
    if not isinstance(assessed,list) or any(not isinstance(a,dict) for a in assessed) or {a.get('id') for a in assessed}!=set(candidates) or len(assessed)!=len(candidates):
        raise ValueError('Assess every candidate id exactly once')
    if not numeric_ids:raise ValueError('Query numeric evidence before finishing')
    for a in assessed:
        if a.get('verdict') not in ('supported','rejected','uncertain'):raise ValueError('Invalid verdict')
        if not isinstance(a.get('evidence_ids'),list) or any(i not in numeric_ids|image_ids|{'E0'} for i in a['evidence_ids']):
            raise ValueError('Cite only returned numeric ids or actually viewed image ids')
        if a['verdict'] in ('supported','rejected') and not set(a['evidence_ids'])&numeric_ids:
            raise ValueError('A supported/rejected candidate needs requested numeric evidence')
        explanation=a.get('explanation')
        if not isinstance(explanation,dict) or any(not explanation.get(k) for k in ('required_observation','actual_observation','missing_evidence')):
            raise ValueError('For each candidate explain required_observation, actual_observation and missing_evidence; use 无 only when justified')
    chosen=action.get('selected_candidate_id')
    if chosen is not None and not any(a['id']==chosen and a['verdict']=='supported' for a in assessed):
        raise ValueError('Choose a supported candidate, or null when undetermined')
    if not action.get('observations') or not action.get('failure_reason') or not action.get('reason_category'):
        raise ValueError('Final needs observations, failure_reason and reason_category')
    if chosen is None and action['reason_category']!='待确认':raise ValueError('No supported selection must be 待确认')


def run(client,evidence,official,out,max_turns=12,workers=2):
    initial=[evidence.images[f'I_overview_{i:02d}'] for i in range(12)]
    overview=inspect_images(client,evidence,initial,'整体12个时刻，逐组观察', 'overview',workers)
    contract={'rules':RULES,'official_evidence':{'id':'E0','metrics':official['metrics'],'source':official['source'],'sha256':official['source_sha256']},
       'layout':POSITIONS,'layout_note':'3x3布局，中央空白；八相机无旧BEV','overview_observations':overview,
       'numeric_tools':{'ego':'自车速度、纵向加速度、位置、朝向','actors':'对象ID、相对位置、几何间距、世界速度及速度变化；可选track_token、nearest',
                        'boundary':'车角边界余量、区域外车角','plans':'所选轨迹；steps表示输入帧，含executed字段'},
       'actions':{
          'images':{'action':'images','purpose':'想核实什么','requests':[{'camera':'CAM_F0 或其他实际相机名或 bev','steps':[6,7,8]}]},
          'visual_complete':{'action':'visual_complete','observations':['注明可见现象和step'],'candidates':[{'id':'A','reason':'一个候选行为原因','needs_evidence':'尚需核实什么'}]},
          'numbers':{'action':'numbers','purpose':'比较哪个候选','queries':[{'kind':'ego','steps':[3,4,5]},{'kind':'actors','steps':[3,4,5],'nearest':3}]},
          'final':{'action':'final','observations':['核验后现象'],'candidate_assessment':[{'id':'A','verdict':'supported/rejected/uncertain','evidence_ids':['E1'],'explanation':{'required_observation':'回答该候选needs_evidence，需要何种观测才能支持或直接否定','actual_observation':'实际证据中的相关观测；无直接反证不能rejected','missing_evidence':'尚缺什么，缺少关键证据则uncertain；确无缺失才写无'}}],
                   'selected_candidate_id':'支持最强的候选id；不能确定则null','failure_reason':'一句驾驶行为原因','reason_category':'简短可汇总的行为类别；不能确定则待确认','uncertainty':'仍不确定之处'}},
       'instruction':'你主导取证。先按需补看图片（不限制总张数），形成现象和若干候选后发visual_complete，再请求数值逐项比较。数值阶段也可继续补图。最终逐项检查引用的证据是否真正支持或排除该候选；不够则uncertain。一次只输出一个action的简洁中文JSON，不输出Markdown。不要用程序外知识补造证据。'}
    save(out/'context.json',contract)
    history=[{'role':'system','content':SYSTEM},{'role':'user','content':json.dumps(contract,ensure_ascii=False)}]
    candidates={};numeric_ids=set();image_ids={im['id'] for im in initial};trace=[]
    for turn in range(max_turns):
        action=client.ask(history,f'turn_{turn:02d}')
        print('ACTION',turn,action.get('action'),flush=True)
        history.append({'role':'assistant','content':json.dumps(action,ensure_ascii=False)})
        try:
            kind=action.get('action')
            if kind=='images':
                images=evidence.select_images(action.get('requests'))
                response={'image_reviews':inspect_images(client,evidence,images,action.get('purpose'),f'detail_{turn:02d}',workers)}
                image_ids.update(im['id'] for im in images)
            elif kind=='visual_complete':
                items=action.get('candidates')
                if not action.get('observations') or not isinstance(items,list) or not items or any(not c.get('id') or not c.get('reason') for c in items):
                    raise ValueError('Provide observations and candidates with unique ids and reasons')
                if len({c['id'] for c in items})!=len(items):raise ValueError('Duplicate candidate id')
                candidates={c['id']:c for c in items};save(out/'visual_hypotheses.json',action)
                response={'state':'numeric_verification','instruction':'现在主动请求数值，逐项核验候选。必要时仍可补图。'}
            elif kind=='numbers':
                if not candidates:raise ValueError('Finish visual observations and candidates first')
                queries=action.get('queries')
                if not isinstance(queries,list) or not queries:raise ValueError('Provide nonempty queries')
                returned=[]
                for query in queries:
                    try:
                        value=evidence.query(query);identifier=f'E{len(numeric_ids)+1}'
                        value={'id':identifier,**value};numeric_ids.add(identifier)
                        save(out/(identifier+'.json'),value);returned.append(value)
                    except (ValueError,KeyError,TypeError) as exc:returned.append({'request':query,'tool_error':str(exc)})
                response={'evidence':returned}
            elif kind=='final':
                validate_final(action,candidates,numeric_ids,image_ids)
                result={**action,'token':evidence.row['token'],'status':'supported' if action['selected_candidate_id'] is not None else 'undetermined',
                        'analysis_level':'driving_behavior','source':'external_vlm','visual_review_reused':False}
                save(out/'result.json',result)
                trace.append({'action':action,'response':{'state':'completed'}});save(out/'trace.json',trace)
                return result
            else:raise ValueError('Unknown action')
        except (ValueError,KeyError,TypeError,AttributeError,IndexError) as exc:
            response={'tool_error':str(exc),'instruction':'修正请求或补充证据，不要跳过核验。'}
        trace.append({'action':action,'response':response});save(out/'trace.json',trace)
        history.append({'role':'user','content':json.dumps(response,ensure_ascii=False)})
    raise RuntimeError('Turn budget exhausted before an evidence-backed final answer; analysis incomplete')
