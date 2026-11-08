"""Opt-in fast backend for the original paired pilot, preserving its protocol.

python -m research.train_parallel --output <new-folder> --seconds 600 --envs 1024
Use --smoke for disposable timing/qualification only; no learned checkpoint.
"""
import argparse
from dataclasses import asdict
import sys
from research import pilot_10m as p
from research.parallel_ppo import ParallelArm, Options


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--no-rollout-graph',action='store_true')
    parser.add_argument('--no-network-graph',action='store_true')
    parser.add_argument('--no-fused-adam',action='store_true')
    parser.add_argument('--deferred-metrics',action='store_true')
    selected,remaining=parser.parse_known_args()
    options=Options(not selected.no_rollout_graph,not selected.no_network_graph,
                    not selected.no_fused_adam,selected.deferred_metrics)
    original=p.Arm
    def arm_factory(initial,scenes,seed,native,device,cache,dense,reference=False,rollout=32):
        args=(initial,scenes,seed,native,device,cache,dense,reference,rollout)
        if reference or device!='cuda':return original(*args)
        return ParallelArm(*args,options=options)
    sys.argv=[sys.argv[0],*remaining]
    p.main(arm_factory=arm_factory,backend_config=dict(name='parallel-ppo-v1',**asdict(options)),
           extra_sources=('research/parallel_ppo.py','research/train_parallel.py'))


if __name__=='__main__':main()
