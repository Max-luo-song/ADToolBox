"""Run an authorized failure batch with resume, bounded retries and a progress watchdog."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import PROJECT
from evidence import save


def counts(directory,tokens):
    result={'total':len(tokens),'completed':0,'failed':0,'running':0,'pending':0}
    for token in tokens:
        path=directory/'scenes'/token/'status.json'
        state=json.loads(path.read_text())['state'] if path.is_file() else 'pending'
        result[state if state in result else 'failed']+=1
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-name',required=True)
    args=parser.parse_args()
    if not args.run_name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.run_name):parser.error('Invalid run-name')
    directory=PROJECT/'runs'/args.run_name;directory.mkdir(parents=True,exist_ok=True)
    lock=(directory/'supervisor.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tokens=json.loads((directory/'selected_tokens.json').read_text())
    started=time.time();last_exit=None
    for round_number in range(1,4):
        if counts(directory,tokens)['completed']==len(tokens):break
        with (directory/'batch.log').open('a') as log:
            process=subprocess.Popen([sys.executable,'-u',str(PROJECT/'run.py'),'batch','--tokens',*tokens,
                '--run-name',args.run_name,'--resume','--attempts','3'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            launched=time.time()
            while process.poll() is None:
                save(directory/'watchdog.json',{'state':'running','pid':os.getpid(),'runner_pid':process.pid,
                    'round':round_number,'started_unix':started,'updated_unix':time.time(),**counts(directory,tokens)})
                if time.time()-max(launched,(directory/'batch.log').stat().st_mtime)>600:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL)
                    log.write('\nWATCHDOG: restarted after 600 seconds without log progress\n');log.flush()
                    break
                time.sleep(10)
            last_exit=process.wait()
        if counts(directory,tokens)['completed']<len(tokens) and round_number<3:time.sleep(30)
    result=counts(directory,tokens)
    save(directory/'watchdog.json',{'state':'completed' if result['completed']==len(tokens) else 'needs_attention',
        'pid':os.getpid(),'last_runner_exit':last_exit,'started_unix':started,'updated_unix':time.time(),**result})


if __name__=='__main__':main()
