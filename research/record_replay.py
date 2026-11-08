"""CPU-only synthetic replay of four preselected unseen-composition episodes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
ROOT=Path(r'C:\Users\maxbr\Desktop\game\IsaacRL\binding-of-ai')
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
try:
    from research.campaign_core import *
except ModuleNotFoundError:
    from campaign_core import *
from physics_v2 import NativePhysics

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--cache',type=Path,required=True)
    args=ap.parse_args();args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    names=['pos','targets','hp','health','hostile_pos','hostile_life','bpos','life']
    history={k:[] for k in names};active=np.ones(4,dtype=bool);won=np.zeros(4,dtype=bool);length=np.zeros(4,dtype=int)
    with patch('torch.cuda._lazy_init',side_effect=AssertionError('CPU replay initialized CUDA')):
        checkpoint=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
        policy=Policy();policy.load_state_dict(checkpoint['state_dict']);policy.eval()
        native=NativePhysics.load(ROOT/'data/physics_v2/calibration_r5.json')
        env=TaskEnv([Scene(2,2,4)]*4,949001,native,'cpu',args.cache,budget=1,autoreset=False)
        previous=torch.full((4,),45,dtype=torch.long)
        frozen={name:getattr(env,name).cpu().numpy().copy() for name in names}
        with torch.no_grad():
            for step in range(400):
                for name in names:
                    value=getattr(env,name).cpu().numpy().copy()
                    value[~active]=frozen[name][~active]
                    history[name].append(value)
                if not active.any():break
                action=policy(env.obs(),previous)[0].argmax(-1)
                obs,_,done,info=env.step(action)
                length+=active
                newly=active&done.numpy()
                for name in names:frozen[name][newly]=getattr(env,name).cpu().numpy()[newly]
                won|=newly&info['success'].numpy()
                active&=~done.numpy();previous=action
    arrays={k:np.stack(v) for k,v in history.items()}
    np.savez_compressed(args.output/'trajectory.npz',**arrays,length=length,won=won)
    metadata=dict(cpu_only=True,synthetic=True,seed=949001,budget=1,composition='2P_2T',
                  checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  episodes=[dict(index=i,clear=bool(won[i]),decisions=int(length[i])) for i in range(4)])
    (args.output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(10,6.1));axes=axes.ravel()
    artists=[]
    for i,ax in enumerate(axes):
        ax.set(xlim=(0,520),ylim=(280,0),aspect='equal');ax.set_facecolor('#faf9f5')
        ax.tick_params(labelsize=7)
        trail,=ax.plot([],[],color='#4a85bc',alpha=.45,lw=1)
        player=ax.scatter([],[],s=70,c='#2674b8',edgecolors='white',zorder=5)
        shooters=ax.scatter([],[],s=90,c='#b94742',marker='s',zorder=4)
        chasers=ax.scatter([],[],s=65,c='#d88e2a',marker='D',zorder=4)
        hostile=ax.scatter([],[],s=13,c='#d45a55',zorder=3)
        tears=ax.scatter([],[],s=10,c='#64b4dd',zorder=3)
        title=ax.set_title('Episode '+str(i+1),fontsize=9)
        artists.append((trail,player,shooters,chasers,hostile,tears,title))
    fig.suptitle('Simulation synthétique — 2 poursuivants + 2 tireurs jamais entraînés\n4 premiers épisodes CPU, graine fixée ; premier dégât = fin',fontsize=10)
    first=artists[0]
    fig.legend([first[1],first[2],first[3],first[4],first[5]],
               ['Joueur','Tireur','Poursuivant','Projectile ennemi','Larme'],
               loc='lower center',ncol=5,fontsize=7)
    fig.tight_layout(rect=(0,.04,1,.92),h_pad=3.0)
    def draw(frame):
        for i,(trail,player,shooters,chasers,hostile,tears,title) in enumerate(artists):
            t=min(frame,int(length[i]));pos=arrays['pos'][t,i]
            path=arrays['pos'][:t+1,i];trail.set_data(path[:,0],path[:,1]);player.set_offsets(pos[None])
            hp=arrays['hp'][t,i];targets=arrays['targets'][t,i]
            shooters.set_offsets(targets[:2][hp[:2]>0]);chasers.set_offsets(targets[2:][hp[2:]>0])
            hostile.set_offsets(arrays['hostile_pos'][t,i][arrays['hostile_life'][t,i]>0])
            tears.set_offsets(arrays['bpos'][t,i][arrays['life'][t,i]>0])
            status=('victoire' if won[i] else 'échec') if frame>=length[i] else 'en cours'
            title.set_text(f'Episode {i+1} — {t/10:.1f} s — {status}')
    frames=list(range(0,int(length.max())+1,2))
    if frames[-1]!=int(length.max()):frames.append(int(length.max()))
    animation=FuncAnimation(fig,draw,frames=frames,interval=200,blit=False)
    animation.save(args.output/'replay.gif',writer=PillowWriter(fps=5),dpi=85)
    draw(frames[-1]);fig.savefig(args.output/'final.png',dpi=150);plt.close(fig)
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
