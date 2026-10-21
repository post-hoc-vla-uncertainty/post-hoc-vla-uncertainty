#!/usr/bin/env python3
"""Read-only progress attachment for an already-running reproduction or extension."""
import argparse
import json
import os
from pathlib import Path
import time
from run_progress import LogFollower,emit


def summaries(roots,seen):
    for root in roots:
        for path in sorted(root.rglob('run_state.json')):
            try:
                data=json.loads(path.read_text());rows=data.get('rounds',[])
                if not rows:continue
                row=max(rows,key=lambda r:r['round_index'])
                latest=dict(round=row['round_index'],stages=row.get('completed_stages',[]))
                evaluations=sorted(path.parent.glob('round_*/training/member_00/member_evaluation.json'))
                evaluation=json.loads(evaluations[-1].read_text()) if evaluations else None
                if evaluation:latest['evaluation']={k:evaluation.get(k) for k in ('round_index','total_successes','total_rollouts','macro_avg_success_rate')}
                if seen.get(path)!=latest:
                    stage=', '.join(latest['stages']) or 'none yet'
                    emit(str(path.parent),f'ARTIFACT STATUS: latest round {latest["round"]+1}; completed stages: {stage}')
                    if evaluation:emit(str(path.parent),f'Last completed evaluation: round {evaluation["round_index"]+1}; success={evaluation.get("total_successes")}/{evaluation.get("total_rollouts")}; task-macro={evaluation.get("macro_avg_success_rate"):.1%}')
                    seen[path]=latest
            except (FileNotFoundError,json.JSONDecodeError,KeyError,TypeError):continue


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,action='append',help='Directory to watch recursively; repeat for multiple runs')
    parser.add_argument('--seconds',type=float,default=60,help='Heartbeat interval')
    parser.add_argument('--tail',type=int,default=8,help='Initial lines from the six most recently written logs')
    parser.add_argument('--once',action='store_true',help='Print a snapshot and exit')
    args=parser.parse_args()
    if not 0<args.seconds<float('inf') or args.tail<0:parser.error('Invalid interval or tail size')
    storage=Path(os.environ.get('STORAGE_ROOT',Path(__file__).resolve().parents[1]/'storage'))
    roots=args.run_dir or [storage/'logs/reproduction',storage/'outputs/reproduction']
    emit('watch','Read-only attachment; it does not start, stop or resume any job. Ctrl-C only stops this viewer.')
    emit('watch','Existing processes can only report metrics already written to their logs; artifact status does not prove a worker is alive.')
    follower=LogFollower(roots=roots,label='watch',replay=args.tail,seconds=args.seconds);seen={}
    summaries(roots,seen)
    if args.once:return
    try:
        while True:
            follower.poll();summaries(roots,seen);time.sleep(min(2,args.seconds))
    except KeyboardInterrupt:emit('watch','Viewer stopped; jobs were left running.')


if __name__=='__main__':main()
