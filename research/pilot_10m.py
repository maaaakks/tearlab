"""Ten-minute paired PPO pilot on the native simulator; no live Isaac calls."""
from __future__ import annotations
import argparse
import copy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
from research.gpu_simulator import CudaGraphSimulator
from research.staged_simulator import StagedCombatSuiteEnv, ALL
from research.rapid_physics import StrictCombatSuiteEnv
from research.native_runtime import source_digest
from physics_v2 import NativePhysics
from shooter_lab import ShooterPolicy
from combat_ppo import clipped_actor_loss

ROOT = Path(__file__).resolve().parents[1]
GAMMA, LAMBDA = .995, .95
HOLDOUT = {(3,1), (4,2), (4,3)}  # (total enemies, shooters), not seen in training


@dataclass(frozen=True)
class Scene:
    enemies: int
    shooters: int
    chaser_hp: int

    @property
    def family(self):
        return f'{self.enemies-self.shooters}P_{self.shooters}T'

    @property
    def name(self):
        return f'{self.family}_hp{self.chaser_hp}'


def cells(heldout=False, hard=False):
    result=[]
    for n in range(1,5):
        for s in range(n+1):
            if ((n,s) in HOLDOUT) != heldout: continue
            if hard and n==s: continue
            for hp in ([12] if n==s or hard else [2,4,8]):
                result.append(Scene(n,s,hp))
    return result


def balanced_training(n):
    groups={}
    for scene in cells():groups.setdefault(scene.family,[]).append(scene)
    families=list(groups.values())
    return [families[i%len(families)][(i//len(families))%len(families[i%len(families)])] for i in range(n)]


class AutoResetMixin:
    @torch.no_grad()
    def step(self,actions):
        # The captured path returns the genuine final observation before reset,
        # and the reset observation separately. No reset-state timeout bootstrap.
        final,reward,done,info=super().step(actions)
        self.keep_terminal=False
        try:self.reset(done)
        finally:self.keep_terminal=True
        info['terminal_observation']=final
        return self.obs(),reward,done,info


class NativeTrainingEnv(AutoResetMixin,StagedCombatSuiteEnv):pass
class ReferenceTrainingEnv(AutoResetMixin,StrictCombatSuiteEnv):pass


def make_env(scenes,seed,native,device='cuda',cache=None,reference=False,width=520,height=280):
    cls=ReferenceTrainingEnv if reference else NativeTrainingEnv
    kwargs={} if reference else dict(optimizations=ALL,cache=cache)
    env=cls([12]*len(scenes),device,seed,native,**kwargs)
    env.counts.copy_(torch.tensor([s.enemies for s in scenes],device=device))
    env.shooters.copy_(torch.tensor([s.shooters for s in scenes],device=device))
    env.chaser_hits.copy_(torch.tensor([s.chaser_hp for s in scenes],device=device))
    slots=torch.arange(4,device=device)[None]
    env.is_chaser.copy_((slots>=env.shooters[:,None])&(slots<env.counts[:,None]))
    env.size.copy_(torch.tensor([width,height],device=device))
    env.p.room_width=width;env.p.room_height=height;env.p.max_steps=400
    env.reset(torch.ones(env.n,device=device,dtype=torch.bool))
    env.keep_terminal=True
    if not reference:env.prepare()
    sim=env if reference or device=='cpu' else CudaGraphSimulator(env)
    initial_hp=torch.tensor([(s.enemies-s.shooters)*s.chaser_hp+s.shooters*12 for s in scenes],device=device,dtype=torch.float32)
    return sim,env,initial_hp


class PilotPolicy(ShooterPolicy):
    """Existing entity encoders/trunk, extended only by last request/BOS input."""
    def __init__(self):
        super().__init__()
        self.body[0]=nn.Linear(176+46,128)
        nn.init.orthogonal_(self.actor.weight,gain=.01)
        nn.init.zeros_(self.actor.bias)

    def forward(self,obs,previous):
        enemies=obs[:,48:76].reshape(-1,4,7)
        projectiles=obs[:,76:].reshape(-1,32,5)
        features=torch.cat((obs[:,:48],self.pool(self.enemy,enemies,enemies[:,:,5]>0),
                            self.pool(self.projectile,projectiles,projectiles[:,:,4]>0),
                            F.one_hot(previous,46).to(obs.dtype)),1)
        state=self.body(features)
        return self.actor(state),self.value(state).squeeze(-1)


def task_reward(info,initial_hp,dense):
    task=info['success'].float()-info['contact'].float()
    time_cost=torch.full_like(task,-.001)
    progress=.1*info['damage']/initial_hp if dense else torch.zeros_like(task)
    return task+time_cost+progress,dict(task=task,time=time_cost,progress=progress)


def gae(rewards,dones,timeouts,values,bootstrap,timeout_values,gamma=GAMMA,lam=LAMBDA):
    adv=torch.zeros_like(values);carry=torch.zeros_like(bootstrap)
    for t in reversed(range(len(rewards))):
        following=bootstrap if t==len(rewards)-1 else values[t+1]
        following=torch.where(dones[t],torch.where(timeouts[t],timeout_values[t],0),following)
        delta=rewards[t]+gamma*following-values[t]
        carry=delta+gamma*lam*(~dones[t])*carry
        adv[t]=carry
    return adv,adv+values


class Arm:
    def __init__(self,initial,scenes,seed,native,device,cache,dense,reference=False,rollout=32):
        self.name='bounded_damage' if dense else 'task_only';self.dense=dense
        self.policy=copy.deepcopy(initial).to(device)
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=3e-4,eps=1e-5)
        self.sim,self.env,self.initial_hp=make_env(scenes,seed,native,device,cache,reference)
        self.obs=self.env.obs();self.previous=torch.full((len(scenes),),45,dtype=torch.long,device=device)
        self.action_rng=torch.Generator(device=device).manual_seed(seed+91)
        self.shuffle_rng=torch.Generator(device=device).manual_seed(seed+193)
        self.t=rollout;self.n=len(scenes);self.device=device;self.updates=0
        self.batch={
            'obs':torch.empty(rollout,self.n,236,device=device),
            'terminal_obs':torch.empty(rollout,self.n,236,device=device),
            **{k:torch.empty(rollout,self.n,device=device) for k in ['reward','value','logp']},
            **{k:torch.empty(rollout,self.n,device=device,dtype=torch.long) for k in ['action','previous']},
            **{k:torch.empty(rollout,self.n,device=device,dtype=torch.bool) for k in ['done','timeout']}}
        self.totals=torch.zeros(8,device=device,dtype=torch.float64)
        self.action_counts=torch.zeros(45,device=device,dtype=torch.long)

    def update(self,epochs=2,minibatch=4096):
        torch.cuda.synchronize() if self.device=='cuda' else None
        start=time.perf_counter();b=self.batch
        with torch.no_grad():
            for t in range(self.t):
                b['obs'][t].copy_(self.obs);b['previous'][t].copy_(self.previous)
                logits,value=self.policy(self.obs,self.previous)
                logp=logits.log_softmax(-1)
                action=torch.multinomial(logp.exp(),1,generator=self.action_rng).squeeze(-1)
                self.action_counts.add_(torch.bincount(action,minlength=45))
                obs,_,done,info=self.sim.step(action)
                reward,parts=task_reward(info,self.initial_hp,self.dense)
                for key,val in [('value',value),('action',action),('logp',logp.gather(1,action[:,None]).squeeze(1)),('reward',reward),('done',done),('timeout',info['timeout']),('terminal_obs',info['terminal_observation'])]:
                    b[key][t].copy_(val)
                self.totals.add_(torch.stack((done.sum(),info['success'].sum(),info['contact'].sum(),info['damage'].sum(),parts['task'].sum(),parts['progress'].sum(),parts['time'].sum(),info['fired'].sum())))
                self.obs=obs;self.previous=torch.where(done,45,action)
            bootstrap=self.policy(self.obs,self.previous)[1]
            timeout_values=torch.zeros_like(b['value'])
            index=b['timeout'].flatten().nonzero().squeeze(-1)
            if len(index):
                value=self.policy(b['terminal_obs'].flatten(0,1)[index],b['action'].flatten()[index])[1]
                timeout_values.view(-1)[index]=value
            advantage,target=gae(b['reward'],b['done'],b['timeout'],b['value'],bootstrap,timeout_values)
            advantage=(advantage-advantage.mean())/advantage.std().clamp_min(1e-8)
        torch.cuda.synchronize() if self.device=='cuda' else None
        collected=time.perf_counter();steps=0;klmax=0.;clip_mean=0.;entropy_mean=0.
        flat_obs=b['obs'].flatten(0,1);flat_prev=b['previous'].flatten();old=b['logp'].flatten()
        for _ in range(epochs):
            permutation=torch.randperm(self.t*self.n,device=self.device,generator=self.shuffle_rng)
            for ids in permutation.split(minibatch):
                logits,value=self.policy(flat_obs[ids],flat_prev[ids])
                lp=logits.log_softmax(-1);probs=lp.exp();entropy=-(probs*lp).sum(-1).mean()
                selected=lp.gather(1,b['action'].flatten()[ids,None]).squeeze(1)
                delta=selected-old[ids];kl=(delta.exp()-1-delta).mean()
                kval=float(kl.detach());klmax=max(klmax,kval)
                if kval>.02:break
                loss=clipped_actor_loss(selected,old[ids],advantage.flatten()[ids])+.5*(value-target.flatten()[ids]).square().mean()-.01*entropy
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite PPO loss')
                self.optimizer.zero_grad(set_to_none=True);loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),.5)
                self.optimizer.step();steps+=1
                clip_mean+=float(((delta.detach().exp()-1).abs()>.2).float().mean());entropy_mean+=float(entropy.detach())
            if kval>.02:break
        torch.cuda.synchronize() if self.device=='cuda' else None
        self.updates+=1
        return dict(collection_seconds=collected-start,optimization_seconds=time.perf_counter()-collected,
                    optimizer_steps=steps,max_kl=klmax,entropy=entropy_mean/max(steps,1),clip_fraction=clip_mean/max(steps,1))


@torch.no_grad()
def evaluate(policy,scenes,per_cell,seed,native,cache,width=520,height=280):
    expanded=[s for s in scenes for _ in range(per_cell)]
    sim,env,hp=make_env(expanded,seed,native,'cuda',cache,width=width,height=height)
    obs=env.obs();previous=torch.full((env.n,),45,device='cuda',dtype=torch.long)
    active=torch.ones(env.n,device='cuda',dtype=torch.bool)
    hits=torch.zeros(env.n,device='cuda');length=torch.zeros_like(hits)
    success=torch.zeros_like(active);contacts=torch.zeros_like(active);timeout=torch.zeros_like(active)
    action_rng=torch.Generator(device='cuda').manual_seed(seed+900)
    for _ in range(env.p.max_steps):
        action=(policy(obs,previous)[0].argmax(-1) if policy is not None else
                torch.randint(45,(env.n,),device='cuda',generator=action_rng))
        obs,_,done,info=sim.step(action)
        hits+=active*info['damage'];length+=active
        success|=active&info['success'];contacts|=active&info['contact'];timeout|=active&info['timeout']
        active&=~done;previous=torch.where(done,45,action)
        if not active.any():break
    rows=[]
    for i,scene in enumerate(scenes):
        sl=slice(i*per_cell,(i+1)*per_cell);wins=success[sl]
        rows.append(dict(name=scene.name,family=scene.family,episodes=per_cell,
                         clear_rate=float(wins.float().mean()),contact_rate=float(contacts[sl].float().mean()),
                         timeout_rate=float(timeout[sl].float().mean()),hp_fraction=float((hits[sl]/hp[sl]).mean()),
                         clear_seconds=float((length[sl][wins]*.1).mean()) if wins.any() else None))
    families={}
    for row in rows:families.setdefault(row['family'],[]).append(row)
    metrics={f:{k:statistics.mean(r[k] for r in values) for k in ['clear_rate','contact_rate','timeout_rate','hp_fraction']} for f,values in families.items()}
    return dict(cells=rows,families=metrics,macro={k:statistics.mean(v[k] for v in metrics.values()) for k in ['clear_rate','contact_rate','timeout_rate','hp_fraction']},
                incomplete=int(active.sum()),width=width,height=height,seed=seed,episodes=len(expanded))


def sources(folder):
    files=['research/pilot_10m.py','research/staged_simulator.py','research/native_runtime.py','research/gpu_simulator.py',
           'research/rapid_physics.py','research/native/isaac_kernels.h','research/native/isaac_kernels.cu',
           'python/physics_v2.py','python/shooter_lab.py','python/combat_suite.py','python/multi_combat_lab.py',
           'python/alignment_lab.py','python/combat_ppo.py']
    hashes={}
    for name in files:
        src=ROOT/name;dst=folder/'sources'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(src,dst);hashes[name]=hashlib.sha256(src.read_bytes()).hexdigest()
    return hashes


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cache',type=Path)
    ap.add_argument('--seconds',type=float,default=600)
    ap.add_argument('--envs',type=int,default=1024)
    ap.add_argument('--seed',type=int,default=104729)
    ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    torch.set_num_threads(1);torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    native=NativePhysics.load(ROOT/'data/physics_v2/calibration_r5.json')
    folder=args.output;folder.mkdir(parents=True,exist_ok=False)
    report=dict(config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                protocol='PPO paired rewards, one fresh training seed, no live game',
                from_scratch=True,algorithm='PPO',native=asdict(native),source_sha256=sources(folder),native_source_sha256=source_digest(),
                train_cells=[asdict(s) for s in cells()],holdout_cells=[asdict(s) for s in cells(True)],
                reward={'task_only':'success - contact - .001','bounded_damage':'task_only + .1 * actual HP removed / initial total HP'},
                reward_audit='No wall, aiming, distance or kill bonus. Progress bonus at most .1 per episode; all failures remain negative at the full timeout.',
                observations='236 instrumented features + explicit previous action/BOS; velocities, HP and attack warning are privileged, not RGB',
                selection='ID family-macro clear rate, HP fraction only breaks ties. OOD never selects checkpoints.',
                gamma=GAMMA,gae_lambda=LAMBDA,epochs=2,minibatch=4096,lr=.0003,entropy=.01,
                live_validated=False,initial={},evaluations=[],complete=False)
    def write():(folder/'report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    initial=PilotPolicy().cuda()
    # Preflight on disposable fresh copies, excludes its interactions from learning.
    from research.gpu_ablation import assert_pair
    verify_scenes=balanced_training(64)
    fast,fe,_=make_env(verify_scenes,args.seed,native,'cuda',args.cache)
    reference,re,_=make_env(verify_scenes,args.seed,native,'cuda',args.cache,True)
    errors={}
    for t in range(24):
        action=(torch.arange(64,device='cuda')+t*17)%45
        x=reference.step(action);y=fast.step(action)
        # Extra terminal observation is a normal float tensor in info.
        assert_pair(re,fe,x,y,3e-4,1e-5,errors)
    report['backend_qualification']=dict(passed=True,max_abs=max(errors.values()),rng_exact=torch.equal(re.rng.get_state(),fe.rng.get_state()))
    if not report['backend_qualification']['rng_exact']:raise RuntimeError('RNG mismatch in training wrapper')
    del fast,fe,reference,re
    performance={}
    for use_reference in (True,False):
        arm=Arm(initial,balanced_training(args.envs),args.seed,native,'cuda',args.cache,True,use_reference)
        arm.update()
        samples=[arm.update() for _ in range(3)]
        performance['reference_eager' if use_reference else 'native_graph']=dict(samples=samples,
                 median_update_seconds=statistics.median(s['collection_seconds']+s['optimization_seconds'] for s in samples),
                 decisions_per_update=args.envs*32)
        del arm
    report['end_to_end_preflight']=performance;write()
    if args.smoke:
        report['complete']=True;report['smoke_only']=True;write();print(json.dumps(report['end_to_end_preflight']),flush=True);return
    arms=[Arm(initial,balanced_training(args.envs),args.seed,native,'cuda',args.cache,dense) for dense in (False,True)]
    # Entire pilot clock includes baseline evaluation, periodic checks and final audits.
    started=time.perf_counter();train_end=started+max(1,args.seconds-45)
    eval_cells=cells()
    report['initial']['id']=evaluate(initial,eval_cells,12,700001,native,args.cache)
    report['initial']['id_audit']=evaluate(initial,eval_cells,24,700002,native,args.cache)
    report['initial']['composition']=evaluate(initial,cells(True),24,800001,native,args.cache)
    report['initial']['hp12']=evaluate(initial,cells(hard=True),24,800002,native,args.cache)
    report['initial']['geometry']=evaluate(initial,eval_cells,12,800003,native,args.cache,width=640,height=360)
    report['random_controls']=dict(id_audit=evaluate(None,eval_cells,24,700002,native,args.cache),
                                   composition=evaluate(None,cells(True),24,800001,native,args.cache))
    report['initialization_and_preflight_excluded_from_pilot_clock']=True
    timing={a.name:[] for a in arms};best={a.name:(-1.,-1.) for a in arms}
    def save_checkpoint(arm,label,metric):
        path=folder/f'{arm.name}_{label}.pt'
        torch.save(dict(schema='native-ppo-pilot-v1',state_dict=arm.policy.state_dict(),optimizer=arm.optimizer.state_dict(),
                        previous_action_input=True,native_physics=asdict(native),train_cells=report['train_cells'],
                        updates=arm.updates,metric=metric,source_sha256=report['source_sha256']),path)
        path.with_suffix('.meta.json').write_text(json.dumps(dict(arm=arm.name,label=label,updates=arm.updates,metric=metric,
                        checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),live_validated=False),indent=2),encoding='utf8')
    for arm in arms:
        metric=(report['initial']['id']['macro']['clear_rate'],report['initial']['id']['macro']['hp_fraction'])
        best[arm.name]=metric;save_checkpoint(arm,'best',metric)
    # Offline artifacts satisfy provenance without uploading or using home dirs.
    for key,name in [('WANDB_CONFIG_DIR','wandb_config'),('WANDB_CACHE_DIR','wandb_cache'),('WANDB_DATA_DIR','wandb_data')]:
        os.environ[key]=str(folder/name)
    import wandb
    logger=wandb.init(project='IsaacRL',name='gpu10m-paired-'+folder.name,mode='offline',dir=str(folder),
                      tags=['combat','ppo','native-cuda','reward-ablation','synthetic','fresh-seed'],
                      config=dict(seconds=args.seconds,envs=args.envs,seed=args.seed,gamma=GAMMA,gae_lambda=LAMBDA))
    report['wandb']=dict(mode='offline',run_id=logger.id,run_dir=logger.dir)
    next_check=time.perf_counter()+60;write();print(json.dumps(dict(started=str(folder),baseline=report['initial']['id']['macro'])),flush=True)
    pairs=0
    while time.perf_counter()<train_end:
        for arm in arms:timing[arm.name].append(arm.update())
        pairs+=1
        if time.perf_counter()>=next_check:
            row=dict(seconds=time.perf_counter()-started,updates=pairs,train_decisions_per_arm=pairs*args.envs*32,arms={})
            for arm in arms:
                score=evaluate(arm.policy,eval_cells,12,700001,native,args.cache)
                metric=(score['macro']['clear_rate'],score['macro']['hp_fraction'])
                row['arms'][arm.name]=score
                if metric>best[arm.name]:best[arm.name]=metric;save_checkpoint(arm,'best',metric)
                save_checkpoint(arm,'latest',metric)
                logger.log({f'{arm.name}/{k}':v for k,v in score['macro'].items()},step=pairs)
            row['diagnostics']={arm.name:dict(totals=arm.totals.tolist(),last_update=timing[arm.name][-1]) for arm in arms}
            report['evaluations'].append(row);write()
            print(json.dumps(dict(seconds=round(row['seconds'],1),updates=pairs,decisions_per_arm=row['train_decisions_per_arm'],
                                  arms={k:v['macro'] for k,v in row['arms'].items()},diagnostics=row['diagnostics'])),flush=True)
            next_check=time.perf_counter()+60
    report['final']={}
    for arm in arms:
        # Keep latest-vs-best explicit; select solely with the ID validation set.
        latest=evaluate(arm.policy,eval_cells,12,700001,native,args.cache)
        metric=(latest['macro']['clear_rate'],latest['macro']['hp_fraction'])
        save_checkpoint(arm,'latest',metric)
        if metric>best[arm.name]:best[arm.name]=metric;save_checkpoint(arm,'best',metric)
        checkpoint=torch.load(folder/f'{arm.name}_best.pt',map_location='cuda',weights_only=True)
        arm.policy.load_state_dict(checkpoint['state_dict'])
        report['final'][arm.name]=dict(best_updates=checkpoint['updates'],latest_id=latest,
            id_audit=evaluate(arm.policy,eval_cells,24,700002,native,args.cache),
            composition=evaluate(arm.policy,cells(True),24,800001,native,args.cache),
            hp12=evaluate(arm.policy,cells(hard=True),24,800002,native,args.cache),
            geometry=evaluate(arm.policy,eval_cells,12,800003,native,args.cache,width=640,height=360),
            totals=arm.totals.tolist(),action_counts=arm.action_counts.tolist(),timing=timing[arm.name],updates=arm.updates,train_decisions=arm.updates*args.envs*32)
    report.update(complete=True,elapsed_seconds=time.perf_counter()-started,stop_reason='ten_minute_shared_wall_budget',
                  one_training_seed=True,equal_environment_interactions=True,wall_budget_shared_between_arms=True)
    for arm in arms:
        for label in ['best','latest']:
            path=folder/f'{arm.name}_{label}.pt'
            artifact=wandb.Artifact(f'{folder.name}-{arm.name}-{label}',type='model',metadata=dict(live_validated=False))
            artifact.add_file(str(path));artifact.add_file(str(path.with_suffix('.meta.json')))
            logger.log_artifact(artifact)
            metadata=json.loads(path.with_suffix('.meta.json').read_text())
            metadata['wandb_artifact_reference']=f'offline:{logger.id}/{artifact.name}'
            path.with_suffix('.meta.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    logger.finish()
    write();print(json.dumps(dict(complete=True,seconds=report['elapsed_seconds'],updates=pairs,
        final={k:{s:v[s]['macro'] for s in ('id_audit','composition','hp12','geometry')} for k,v in report['final'].items()})),flush=True)


if __name__=='__main__':main()
