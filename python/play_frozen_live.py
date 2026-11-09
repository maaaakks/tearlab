"""Frozen 238-feature policy at normal game speed; user arms control with C."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import signal
import socket
import sys
import time
ROOT=Path(r'C:\Users\maxbr\Desktop\game\IsaacRL\binding-of-ai')
sys.path[:0]=[str(ROOT),str(ROOT/'python')]
import torch
from research.campaign_core import Policy
from shooter_lab import physics


class LiveEncoder:
    def __init__(self):self.p=physics();self.slots=[None]*8

    def reset(self):self.slots=[None]*8

    def encode(self,state):
        p=state['player'];room=state['room'];enemies=state.get('enemies',[])
        if not 1<=len(enemies)<=4:raise ValueError('Ce modele accepte 1 a 4 ennemis')
        if not 0<room['width']<=2000 or not 0<room['height']<=2000:raise ValueError('Geometrie invalide')
        px,py=p['position'];left,top=room['top_left'];vx,vy=p['velocity']
        obs=[vx*self.p.player_velocity_scale/100,vy*self.p.player_velocity_scale/100,
             (px-left)/640,(py-top)/640,(room['width']-(px-left))/640,(room['height']-(py-top))/640,
             max(0,p['fire_delay'])/30/self.p.fire_interval,min(1.,state['control_tick']/3/400)]
        tears={b['init_seed']:b for b in state.get('projectiles',[]) if b.get('friendly') and b['type']==2}
        if len(tears)>8:raise ValueError('Plus de 8 larmes: controle rendu')
        self.slots=[seed if seed in tears else None for seed in self.slots]
        for seed in sorted(tears):
            if seed not in self.slots:self.slots[self.slots.index(None)]=seed
        for seed in self.slots:
            if seed is None:obs.extend([0.]*5)
            else:
                b=tears[seed];x,y=b['position'];bx,by=b['velocity']
                life=max(0,self.p.tear_lifetime-b['age_ticks']/30)
                obs.extend([(x-px)/320,(y-py)/320,bx*30/400,by*30/400,life/self.p.tear_lifetime])
        for enemy in enemies:
            ex,ey=enemy['position'];evx,evy=enemy['velocity']
            obs.extend([(ex-px)/320,(ey-py)/320,evx*30/100,evy*30/100,
                        max(0,enemy['hp'])/(max(.01,p['damage'])*12),float(enemy['hp']>0),float(enemy.get('attack_warning',0))])
        obs.extend([0.]*(7*(4-len(enemies))))
        hostile=[b for b in state.get('projectiles',[]) if b['type']==9 and not b.get('friendly')]
        if len(hostile)>32:raise ValueError('Plus de 32 projectiles: controle rendu')
        for b in hostile:
            x,y=b['position'];bx,by=b['velocity'];obs.extend([(x-px)/320,(y-py)/320,bx*30/400,by*30/400,1.])
        obs.extend([0.]*(5*(32-len(hostile))))
        # Training ended at first contact: retain the trained abstract 1-HP context.
        # Actual hearts are logged, never presented as calibrated simulator lives.
        obs.extend([1/3,0.])
        if len(obs)!=238 or not all(math.isfinite(v) for v in obs):raise ValueError('Observation hors contrat')
        return torch.tensor([obs],dtype=torch.float32)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--port',type=int,default=10009);args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1);cp=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    digest=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    sidecar=json.loads(args.checkpoint.with_suffix('.meta.json').read_text())
    if digest!=sidecar['sha256']:raise ValueError('SHA du checkpoint incorrect')
    if cp.get('schema')!='crossfire-v1' or cp.get('variant')!='pool':raise ValueError('Checkpoint de preview incompatible')
    model=Policy().eval();model.load_state_dict(cp['state_dict'])
    for parameter in model.parameters():parameter.requires_grad_(False)
    meta=dict(checkpoint=str(args.checkpoint.resolve()),sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              inference='CPU greedy 10 Hz, no optimizer',control='manual until C in game',unknown_attack_warning=True,
              live_validated=False,started=time.time())
    (args.output/'session.json').write_text(json.dumps(meta,indent=2));status_path=args.output/'status.json'
    encoder=LiveEncoder();sock=None;buffer=bytearray();previous=45;last_room=None;last_control=None;was_enabled=False;action_id=0;stop=False
    def stopping(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGINT,stopping);signal.signal(signal.SIGTERM,stopping)
    def status(**fields):status_path.write_text(json.dumps(dict(time=time.time(),**fields),indent=2))
    def send(message):sock.sendall((json.dumps(message,separators=(',',':'))+'\n').encode())
    status(state='waiting_game',ready=True)
    print('Agent charge. Demarre Isaac. C active/coupe le controle; aucun reset automatique.',flush=True)
    try:
        with (args.output/'trace.jsonl').open('a',encoding='utf8',buffering=1) as trace:
            while not stop:
                if (args.output/'STOP').exists():break
                if sock is None:
                    try:
                        sock=socket.create_connection(('127.0.0.1',args.port),timeout=.5);sock.settimeout(.5)
                        buffer.clear();previous=45;was_enabled=False;encoder.reset();status(state='connected_manual',ready=True)
                    except OSError:
                        if sock:sock.close()
                        sock=None;time.sleep(.5);continue
                try:
                    data=sock.recv(65536)
                    if not data:raise ConnectionError('Mod disconnected')
                    buffer.extend(data);messages=[]
                    while b'\n' in buffer:
                        line,_,rest=buffer.partition(b'\n');buffer=bytearray(rest)
                        if line:messages.append(json.loads(line))
                    states=[m for m in messages if m.get('kind')=='state']
                    for message in messages:
                        if message.get('kind')=='hello':
                            if message.get('protocol')!='isaac-live-238-v1':raise ValueError('Protocole incompatible')
                            status(state='mod_ready_manual',ready=True,hello=message)
                    if not states:continue
                    state=states[-1];enabled=bool(state.get('assistant_enabled'))
                    if state['room_id']!=last_room or state['control_id']!=last_control or enabled!=was_enabled:
                        encoder.reset();previous=45;last_room=state['room_id'];last_control=state['control_id']
                    was_enabled=enabled;entry=dict(received=time.time(),state=state)
                    if enabled:
                        try:
                            start=time.perf_counter();obs=encoder.encode(state)
                            with torch.inference_mode():logits,_=model(obs,torch.tensor([previous]));action=int(logits.argmax(-1))
                            action_id+=1;message=dict(action=dict(action_id=action_id,room_id=last_room,control_id=state['control_id'],move=action//5,shoot=action%5))
                            send(message);previous=action;entry.update(requested_action=message['action'],inference_ms=(time.perf_counter()-start)*1000)
                            status(state='agent_active',ready=True,room=last_room,enemies=len(state['enemies']),action=action)
                        except (ValueError,KeyError) as exc:
                            send(dict(command='release',reason=str(exc)));previous=45;entry['projection_error']=str(exc)
                            status(state='manual_projection_guard',ready=True,error=str(exc))
                    else:status(state='manual',ready=True,room=last_room,enemies=len(state['enemies']))
                    trace.write(json.dumps(entry,separators=(',',':'))+'\n')
                except socket.timeout:continue
                except (OSError,ConnectionError,json.JSONDecodeError) as exc:
                    status(state='waiting_game',ready=True,error=str(exc));sock.close();sock=None;was_enabled=False
    finally:
        if sock:
            try:send(dict(command='release',reason='Agent ferme - MANUEL'))
            except OSError:pass
            sock.close()
        status(state='stopped',ready=False)


if __name__=='__main__':main()
