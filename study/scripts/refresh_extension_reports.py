"""Refresh the current preview reports from saved artifacts, without GPU work."""
import argparse
from pathlib import Path

from uq_extensions.common import read_json
from uq_extensions.reporting import experiment_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='storage/extensions/experiments/v1')
    args=parser.parse_args()
    result=read_json(Path(args.root)/'01_preview/result.json')
    output=Path(result['report']).parent
    matrix=read_json(output/'scoring/matrix.json')
    run=Path(result['run'])
    summary_path=output/'active_learning/summary.json'
    summaries={str(run):read_json(summary_path)} if summary_path.exists() else None
    print('Refreshing HTML, PNG/PDF plots and derived CSV/JSON; reusing saved scores and outcomes.',flush=True)
    for target,runs in ((output/'scoring',()),(output,(run,))):
        report=experiment_report(target,matrix=matrix,runs=runs,run_summaries=summaries)
        print(report,flush=True)


if __name__=='__main__':main()
