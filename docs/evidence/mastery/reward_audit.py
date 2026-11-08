"""Exact trajectory counterexample: discounted survival is not clear probability."""
import json
from pathlib import Path


def legacy(outcome,length):
    gamma=.995
    terminal={'win':3.,'death':-3.1,'timeout':-2.}[outcome]
    return -.002*(1-gamma**length)/(1-gamma)+gamma**(length-1)*terminal


def main():
    waiting=legacy('timeout',400);win=legacy('win',200);death=legacy('death',100)
    probability=.4
    fight=probability*win+(1-probability)*death
    threshold=(waiting-death)/(win-death)
    result=dict(description='Constructed feasible return comparison, not a measured agent trajectory',
                legacy=dict(wait_timeout400=waiting,win200=win,death100=death,
                            fight_success_probability=probability,fight_expected_return=fight,
                            probability_threshold_to_prefer_fighting=threshold),
                win_objective=dict(gamma=1.,win=3.,all_failure=-3.,time_cost=0.,
                                   waiting_return=-3.,fight_expected_return=6*probability-3),
                finding='Legacy prefers zero-win waiting over a 40%-win attempt; equal undiscounted failure terminals rank by win probability')
    assert fight<waiting
    assert result['win_objective']['fight_expected_return']>-3
    (Path(__file__).parent/'reward-audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
