"""Reuse official per-scene outputs; supplemental geometry never overwrites them."""
import csv,hashlib,math

FIELDS=('step','no_at_fault_collisions','drivable_area_compliance','ego_progress',
        'time_to_collision_within_bound','comfort','driving_direction_compliance','score','first_violation_step')

def load_official(root,index_row):
    path=root/'WE_output/openscene_format/all_scenes_pdm_averages_R.csv'
    with path.open() as f:matches=[r for r in csv.DictReader(f) if r['token']==index_row['token']]
    if len(matches)!=1:raise ValueError('Official metric row must exist exactly once')
    raw=matches[0];values={k:float(raw[k]) if raw[k] else None for k in FIELDS}
    if any(v is not None and not math.isfinite(v) for v in values.values()):raise ValueError('Non-finite official metric')
    mismatches=[k for k in FIELDS if k in index_row and raw[k]!=index_row[k]]
    nc,dac=values['no_at_fault_collisions'],values['drivable_area_compliance']
    rule='collision_and_DAC' if nc<1 and dac<1 else 'collision_only' if nc<1 else 'DAC_only' if dac<1 else 'success'
    timing=values['first_violation_step']
    conclusions={'at_fault_collision_recorded':nc<1,'drivable_area_violation_recorded':dac<1,
       'ttc_metric_failed':values['time_to_collision_within_bound']<1,
       'direction_metric_penalized':values['driving_direction_compliance']<1,
       'comfort_metric_failed':values['comfort']<1,
       'joint_first_violation_step':timing,
       'joint_first_violation_control_time_s':(timing-3)*.5 if timing is not None else None,
       'collision_actor_identity':'not present in official CSV',
       'timing_semantics':'Earliest NC=0 or DAC=0; not separate event times and not NC=0.5 timing.',
       'EP_semantics':'Official progress after multiplicative gating; zero does not prove stationary ego.'}
    artifacts={}
    for relative in ('consolidation_manifest.json','provenance/execution_check.json','provenance/audit.json',
                     'provenance/vocab_check.json','WE_output/simulation_completed.flag'):
        p=root/relative
        if p.is_file():
            entry={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
            if p.suffix=='.json':
                import json
                entry['saved_result']=json.loads(p.read_text())
            artifacts[relative]=entry
    return {'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'saved_run_artifacts':artifacts,
            'raw_row':raw,'metrics':values,'rule_class':rule,'conclusions':conclusions,
            'index_metric_mismatches':mismatches,'authority':'official saved evaluation; no rescoring'}

def compare_evidence(official,findings):
    c=official['conclusions'];checks=[]
    outside=findings.get('first_sampled_corner_outside_step')
    overlap=findings.get('first_sampled_overlap_candidate_step')
    checks.append({'name':'DAC_vs_sampled_corners','status':'consistent' if (outside is not None)==c['drivable_area_violation_recorded'] else 'discrepancy',
                   'official_violation':c['drivable_area_violation_recorded'],'supplemental_first_step':outside})
    checks.append({'name':'NC_vs_sampled_overlap','status':'supported' if c['at_fault_collision_recorded'] and overlap is not None else 'not_attribution_comparable',
                   'official_at_fault_collision':c['at_fault_collision_recorded'],'supplemental_first_step':overlap,
                   'note':'Overlap cannot assign official responsibility or identify the official collision actor.'})
    if official['rule_class']=='collision_only' and official['metrics']['no_at_fault_collisions']==0 and overlap is not None:
        checks.append({'name':'NC_event_timing','status':'consistent' if overlap==c['joint_first_violation_step'] else 'discrepancy'})
    return {'checks':checks,'discrepancies':[x for x in checks if x['status']=='discrepancy'],
            'policy':'Keep official metrics authoritative; flag differences for investigation without replacing official conclusions.'}
