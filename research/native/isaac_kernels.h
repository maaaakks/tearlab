// Portable scalar primitives shared by CPU verification and CUDA kernels.
// Pointer/config ABI version 1. No fast math or fused multiply-add required.
#pragma once
#include <stdint.h>
#include <math.h>
#ifdef __CUDACC__
#define FN __device__ __forceinline__
#else
#define FN inline
#endif
enum P {POS,VEL,TARGET,TVEL,SPEED,BPOS,BVEL,LIFE,COOLDOWN,HPOS,HVEL,HLIFE,
 ASPEED,NVEL,EYE,HP,STEPS,TIMER,IMMUNITY,LAST,CLOCK,ATTACKS,MOVE,CHASER,COUNTS,
 SHOOTERS,CHITS,OLDPOS,OLDTARGET,ALIVE,ACTIVE,TOTALS,FLAGS,MID,RANDOM,OBS,
 MASK,RBASE,RTIMER,RSPEED,RPOS,RANGLE,RRADIUS,RCHSPEED,NPTR};
enum C {WIDTH,HEIGHT,IDLE,PARALLEL,REVERSE,TRANSVERSE,ACCEL,MARGIN,REBOUND,BIAS,
 CORRECT,INHERIT,EYEOFF,FIRETICKS,WALLORDER,VSCALE,TLIFE,HITR,CONTACTR,HOSTR,
 HOSTLIFE,IMMUTICKS,FIREINTERVAL,MAXSTEPS,TARGETHITS,SMIN,SMAX,MIXED,ENEMIES,
 ASMIN,ASMAX,WARNING,DELAY,NCFG};
FN float lo(float a,float b){return a<b?a:b;}
FN float hi(float a,float b){return a>b?a:b;}
FN float clip(float x,float l,float h){return lo(hi(x,l),h);}
FN float norm2(float x,float y){return sqrtf(x*x+y*y);}
struct S {
 void** p; const float* c; int n; int random_offset=0;
 FN float* f(int k) const{return (float*)p[k];}
 FN int64_t* q(int k) const{return (int64_t*)p[k];}
 FN bool* b(int k) const{return (bool*)p[k];}
};
FN bool outside(float x,float y,const float* c){return x<0||y<0||x>c[WIDTH]||y>c[HEIGHT];}
FN bool swept(float sx,float sy,float ex,float ey,float r){
 float dx=ex-sx,dy=ey-sy;
 float t=-(sx*dx+sy*dy)/hi(dx*dx+dy*dy,1e-8f);
 t=clip(t,0,1);float x=sx+t*dx,y=sy+t*dy;
 return x*x+y*y<=r*r;
}
FN float first(float sx,float sy,float ex,float ey,float r){
 float dx=ex-sx,dy=ey-sy,a=dx*dx+dy*dy,b=sx*dx+sy*dy,c=sx*sx+sy*sy-r*r;
 float d=b*b-a*c,t=(-b-sqrtf(hi(d,0)))/hi(a,1e-12f);
 return c<=0?0:((d>=0&&a>1e-12f&&t>=0&&t<=1)?t:INFINITY);
}
FN void motion(S s,int i,int64_t command,bool masked){
 if(masked&&!s.b(ACTIVE)[i])return;
 const float* c=s.c;int m=(int)(command/5);
 float ux=s.f(MOVE)[m*2],uy=s.f(MOVE)[m*2+1];
 float x=s.f(POS)[i*2],y=s.f(POS)[i*2+1];
 float vx=s.f(NVEL)[i*2],vy=s.f(NVEL)[i*2+1];
 for(int half=0;half<2;half++){
  float px=x+vx,py=y+vy;
  x=clip(px,c[MARGIN],c[WIDTH]-c[MARGIN]);y=clip(py,c[MARGIN],c[HEIGHT]-c[MARGIN]);
  bool cx=x!=px,cy=y!=py;int order=(int)c[WALLORDER];
  if(order<3&&(half==1||order==1)){if(cx)vx=0;if(cy)vy=0;}
  float dot=vx*ux+vy*uy,ax=dot*ux,ay=dot*uy;
  float drag=dot>=0?c[PARALLEL]:c[REVERSE];
  float wx=ax*drag+(vx-ax)*c[TRANSVERSE],wy=ay*drag+(vy-ay)*c[TRANSVERSE];
  if(ux*ux+uy*uy==0){wx=vx*c[IDLE];wy=vy*c[IDLE];}
  vx=wx+c[ACCEL]*ux;vy=wy+c[ACCEL]*uy;
  if(order>=3){
   float ix=px<c[MARGIN]?1.f:-1.f,iy=py<c[MARGIN]?1.f:-1.f;
   if(order==4){x=px+clip(x-px,-c[CORRECT],c[CORRECT]);y=py+clip(y-py,-c[CORRECT],c[CORRECT]);cx=cx&&vx*ix<0;cy=cy&&vy*iy<0;}
   if(cx)vx=c[REBOUND]*vx+ix*c[BIAS];if(cy)vy=c[REBOUND]*vy+iy*c[BIAS];
  }
  if(half==0){s.f(MID)[i*2]=vx;s.f(MID)[i*2+1]=vy;}
  if(half==1&&order==2&&cx)vx=0;
 }
 s.f(POS)[i*2]=x;s.f(POS)[i*2+1]=y;
 s.f(NVEL)[i*2]=vx;s.f(NVEL)[i*2+1]=vy;
 s.f(VEL)[i*2]=vx*c[VSCALE];s.f(VEL)[i*2+1]=vy*c[VSCALE];
}
FN void target_velocity(S s,int i){
 for(int e=0;e<4;e++){int j=i*4+e,k=j*2;
  float dx=s.f(POS)[i*2]-s.f(TARGET)[k],dy=s.f(POS)[i*2+1]-s.f(TARGET)[k+1];
  float length=hi(norm2(dx,dy),1e-6f);float speed=s.f(SPEED)[j];
  float alive=s.q(HP)[j]>0?1.f:0.f;
  s.f(TVEL)[k]=(dx/length*speed)*alive;s.f(TVEL)[k+1]=(dy/length*speed)*alive;
 }
}
FN void attacks(S s,int i,bool masked){
 if(masked&&!s.b(ACTIVE)[i])return;
 const float* c=s.c;int count=0;
 int64_t imm=s.q(IMMUNITY)[i]-1;s.q(IMMUNITY)[i]=imm>0?imm:0;
 for(int e=0;e<4;e++){
  int j=i*4+e;
  if(c[MIXED]&&s.b(CHASER)[j])s.q(TIMER)[j]=1000000;
  s.q(TIMER)[j]--;bool fire=s.q(HP)[j]>0&&s.q(TIMER)[j]<=0;
  count+=fire?1:0;
  int slot=0;float life=s.f(HLIFE)[i*32];
  for(int b=1;b<32;b++)if(s.f(HLIFE)[i*32+b]<life){life=s.f(HLIFE)[i*32+b];slot=b;}
  if(fire&&life<=0){
   int k=j*2,d=(i*32+slot)*2;
   float dx=s.f(POS)[i*2]-s.f(TARGET)[k],dy=s.f(POS)[i*2+1]-s.f(TARGET)[k+1];
   float length=hi(norm2(dx,dy),1e-6f);
   s.f(HPOS)[d]=s.f(TARGET)[k];s.f(HPOS)[d+1]=s.f(TARGET)[k+1];
   s.f(HVEL)[d]=dx/length*s.f(ASPEED)[j];s.f(HVEL)[d+1]=dy/length*s.f(ASPEED)[j];
   s.f(HLIFE)[i*32+slot]=c[HOSTLIFE];
  }
  if(fire)s.q(TIMER)[j]=s.q(RANDOM)[j+s.random_offset];
 }
 s.q(ATTACKS)[i]=count;
 if(s.b(ACTIVE)[i])s.f(TOTALS)[i*4+2]+=(float)count;
 if(c[MIXED]){
  target_velocity(s,i);
  for(int e=0;e<4;e++)for(int a=0;a<2;a++){
   int k=(i*4+e)*2+a;float limit=c[a]-20;
   float v=s.f(TVEL)[k],p=s.f(TARGET)[k]+v*(1.f/30.f);
   bool low=p<20,high=p>limit;
   s.f(TARGET)[k]=low?40-p:(high?2*limit-p:p);if(low||high)s.f(TVEL)[k]=-v;
  }
 }
}
FN bool emit(S s,int i,int64_t command){
 int shoot=(int)(command%5);s.q(CLOCK)[i]--;
 int slot=0;float life=s.f(LIFE)[i*8];
 for(int b=1;b<8;b++)if(s.f(LIFE)[i*8+b]<life){life=s.f(LIFE)[i*8+b];slot=b;}
 bool fire=s.b(ACTIVE)[i]&&shoot!=0&&s.q(CLOCK)[i]<0&&life<=0;
 if(fire){
  const float* c=s.c;float ux=s.f(MOVE)[shoot*2],uy=s.f(MOVE)[shoot*2+1];
  float vx=s.f(MID)[i*2],vy=s.f(MID)[i*2+1],dot=vx*ux+vy*uy;
  float px=dot*ux,py=dot*uy,eye=s.f(EYE)[i];int k=(i*8+slot)*2;
  s.f(BPOS)[k]=s.f(POS)[i*2]+(-uy*eye)*c[EYEOFF];
  s.f(BPOS)[k+1]=s.f(POS)[i*2+1]+(ux*eye)*c[EYEOFF];
  s.f(BVEL)[k]=(10*ux+c[INHERIT]*((vx-px)+hi(dot,0)*ux))*30;
  s.f(BVEL)[k+1]=(10*uy+c[INHERIT]*((vy-py)+hi(dot,0)*uy))*30;
  s.f(LIFE)[i*8+slot]=c[TLIFE];s.q(CLOCK)[i]=(int64_t)c[FIRETICKS]-1;s.f(EYE)[i]=-eye;
 }
 s.f(COOLDOWN)[i]=(float)(s.q(CLOCK)[i]>0?s.q(CLOCK)[i]:0)*(1.f/30.f);
 return fire;
}
FN int friendly(S s,int i,int b){
 int j=i*8+b,k=j*2;float life=s.f(LIFE)[j],dt=clip(life,0,1.f/30.f);
 float sx=s.f(BPOS)[k],sy=s.f(BPOS)[k+1];
 float ex=sx+s.f(BVEL)[k]*dt,ey=sy+s.f(BVEL)[k+1]*dt;
 float best=INFINITY;int who=-1;
 if(life>0)for(int e=0;e<4;e++)if(s.b(ALIVE)[i*4+e]){
  int t=(i*4+e)*2;float tx=s.f(OLDTARGET)[t],ty=s.f(OLDTARGET)[t+1];
  float fx=tx+(s.f(TARGET)[t]-tx)*(dt*30),fy=ty+(s.f(TARGET)[t+1]-ty)*(dt*30);
  float v=first(sx-tx,sy-ty,ex-fx,ey-fy,s.c[HITR]);if(v<best){best=v;who=e;}
 }
 s.f(BPOS)[k]=ex;s.f(BPOS)[k+1]=ey;
 s.f(LIFE)[j]=who>=0||outside(ex,ey,s.c)?0:hi(life-1.f/30.f,0);
 return who;
}
FN bool hostile(S s,int i,int b){
 int j=i*32+b,k=j*2;float life=s.f(HLIFE)[j],dt=clip(life,0,1.f/30.f);
 float sx=s.f(HPOS)[k],sy=s.f(HPOS)[k+1];
 float ex=sx+s.f(HVEL)[k]*dt,ey=sy+s.f(HVEL)[k+1]*dt;
 bool hit=life>0&&swept(sx-s.f(OLDPOS)[i*2],sy-s.f(OLDPOS)[i*2+1],ex-s.f(POS)[i*2],ey-s.f(POS)[i*2+1],s.c[HOSTR]);
 s.f(HPOS)[k]=ex;s.f(HPOS)[k+1]=ey;
 s.f(HLIFE)[j]=hit||outside(ex,ey,s.c)?0:hi(life-1.f/30.f,0);
 return hit;
}
FN void finish_contacts(S s,int i,const int* hits,bool phit,bool fire,bool damage_ends){
 int damage=0,kills=0;bool cleared=true,touch=false;
 for(int e=0;e<4;e++){
  int j=i*4+e,k=j*2;int64_t hp=s.q(HP)[j];int d=hits[e]<(int)hp?hits[e]:(int)hp;
  s.q(HP)[j]-=d;damage+=d;kills+=s.b(ALIVE)[j]&&s.q(HP)[j]<=0;cleared=cleared&&s.q(HP)[j]<=0;
  touch=touch||(s.b(ALIVE)[j]&&swept(s.f(OLDPOS)[i*2]-s.f(OLDTARGET)[k],s.f(OLDPOS)[i*2+1]-s.f(OLDTARGET)[k+1],s.f(POS)[i*2]-s.f(TARGET)[k],s.f(POS)[i*2+1]-s.f(TARGET)[k+1],s.c[CONTACTR]));
 }
 bool projectile=phit&&s.q(IMMUNITY)[i]<=0;touch=touch&&s.q(IMMUNITY)[i]<=0;
 bool contact=projectile||touch;if(contact)s.q(IMMUNITY)[i]=(int64_t)s.c[IMMUTICKS];
 if(s.b(ACTIVE)[i]){
  s.f(TOTALS)[i*4]+=damage;s.f(TOTALS)[i*4+1]+=kills;s.f(TOTALS)[i*4+3]+=fire?1.f:0.f;
  s.b(FLAGS)[i*3]=s.b(FLAGS)[i*3]||contact;
  s.b(FLAGS)[i*3+1]=s.b(FLAGS)[i*3+1]||projectile;
  s.b(FLAGS)[i*3+2]=s.b(FLAGS)[i*3+2]||touch;
 }
 s.b(ACTIVE)[i]=s.b(ACTIVE)[i]&&!(cleared||(contact&&damage_ends));
}
FN void projectiles(S s,int i,int64_t command,bool masked,bool damage_ends){
 if(masked&&!s.b(ACTIVE)[i])return;
 bool fire=emit(s,i,command),phit=false;int hits[4]={0,0,0,0};
 for(int b=0;b<8;b++){int e=friendly(s,i,b);if(e>=0)hits[e]++;}
 for(int b=0;b<32;b++)phit=hostile(s,i,b)||phit;
 finish_contacts(s,i,hits,phit,fire,damage_ends);
}
FN float observation_value(S s,int i,int f){
 const float* c=s.c;
 if(f<2)return s.f(VEL)[i*2+f]*.01f;
 if(f<4)return s.f(POS)[i*2+f-2]*(1.f/640.f);
 if(f<6)return (c[f-4]-s.f(POS)[i*2+f-4])*(1.f/640.f);
 if(f==6)return s.f(COOLDOWN)[i]/c[FIREINTERVAL];
 if(f==7)return (float)s.q(STEPS)[i]/c[MAXSTEPS];
 if(f<48){
  int b=(f-8)/5,a=(f-8)%5,j=i*8+b,k=j*2;if(s.f(LIFE)[j]<=0)return 0;
  if(a<2)return (s.f(BPOS)[k+a]-s.f(POS)[i*2+a])*(1.f/320.f);
  if(a<4)return s.f(BVEL)[k+a-2]*(1.f/400.f);return s.f(LIFE)[j]/c[TLIFE];
 }
 if(f<76){
  int e=(f-48)/7,a=(f-48)%7,j=i*4+e,k=j*2;if(s.q(HP)[j]<=0)return 0;
  if(a<2)return (s.f(TARGET)[k+a]-s.f(POS)[i*2+a])*(1.f/320.f);
  if(a<4)return s.f(TVEL)[k+a-2]*.01f;
  if(a==4)return (float)s.q(HP)[j]/c[TARGETHITS];if(a==5)return 1;
  return s.q(TIMER)[j]<=c[WARNING]?1.f:0;
 }
 int b=(f-76)/5,a=(f-76)%5,j=i*32+b,k=j*2;if(s.f(HLIFE)[j]<=0)return 0;
 if(a<2)return (s.f(HPOS)[k+a]-s.f(POS)[i*2+a])*(1.f/320.f);
 if(a<4)return s.f(HVEL)[k+a-2]*(1.f/400.f);return 1;
}
FN void observation(S s,int i){
 for(int f=0;f<236;f++)s.f(OBS)[i*236+f]=observation_value(s,i,f);
}
FN void reset_room(S s,int i){
 const float* c=s.c;bool reset=s.b(MASK)[i];
 if(reset){
  float* r=s.f(RBASE)+i*15;
  float px=c[WIDTH]*.5f+(r[0]-.5f)*40,py=c[HEIGHT]*.5f+(r[1]-.5f)*20;
  s.f(POS)[i*2]=px;s.f(POS)[i*2+1]=py;
  for(int e=0;e<4;e++){
   int j=i*4+e,k=j*2;
   float angle=(r[2]*6.283185307f+((float)e*6.283185307f)/c[ENEMIES])+(r[3+e]-.5f)*.6f;
   float radius=90+20*r[7+e];s.f(TARGET)[k]=px+cosf(angle)*radius;s.f(TARGET)[k+1]=py+sinf(angle)*radius;
   s.f(SPEED)[j]=c[SMIN]+r[11+e]*(c[SMAX]-c[SMIN]);s.q(HP)[j]=e<(int)c[ENEMIES]?(int64_t)c[TARGETHITS]:0;
   float dx=px-s.f(TARGET)[k],dy=py-s.f(TARGET)[k+1],length=hi(norm2(dx,dy),1e-6f);
   s.f(TVEL)[k]=dx/length*s.f(SPEED)[j];s.f(TVEL)[k+1]=dy/length*s.f(SPEED)[j];
   s.q(TIMER)[j]=s.q(RTIMER)[j];s.f(ASPEED)[j]=c[ASMIN]+s.f(RSPEED)[j]*(c[ASMAX]-c[ASMIN]);
  }
  for(int a=0;a<2;a++){s.f(VEL)[i*2+a]=0;s.f(NVEL)[i*2+a]=0;}
  for(int b=0;b<8;b++)s.f(LIFE)[i*8+b]=0;
  for(int b=0;b<32;b++)s.f(HLIFE)[i*32+b]=0;
  s.f(COOLDOWN)[i]=0;s.q(STEPS)[i]=0;s.q(IMMUNITY)[i]=0;s.q(LAST)[i]=0;s.q(CLOCK)[i]=0;s.f(EYE)[i]=1;
 }
 if(!c[MIXED])return;
 if(reset&&s.q(SHOOTERS)[i]<4){
  float px=20+s.f(RPOS)[i*2]*(c[WIDTH]-40),py=20+s.f(RPOS)[i*2+1]*(c[HEIGHT]-40);
  float tx[4],ty[4];bool validroom=true;
  for(int e=0;e<4;e++){
   bool found=false;
   for(int b=0;b<64;b++){
    int k=(e*s.n+i)*64+b;float a=s.f(RANGLE)[k]*6.283185307f,r=90+20*s.f(RRADIUS)[k];
    float x=px+cosf(a)*r,y=py+sinf(a)*r;bool valid=x>=20&&y>=20&&x<=c[WIDTH]-20&&y<=c[HEIGHT]-20;
    for(int prev=0;prev<e;prev++)valid=valid&&norm2(x-tx[prev],y-ty[prev])>=35;
    if(b==0){tx[e]=x;ty[e]=y;}if(valid&&!found){tx[e]=x;ty[e]=y;found=true;}
   }
   validroom=validroom&&found;
  }
  if(validroom){s.f(POS)[i*2]=px;s.f(POS)[i*2+1]=py;for(int e=0;e<4;e++){s.f(TARGET)[(i*4+e)*2]=tx[e];s.f(TARGET)[(i*4+e)*2+1]=ty[e];}}
 }
 for(int e=0;e<4;e++){
  int j=i*4+e;bool chaser=s.b(CHASER)[j],active=e<s.q(COUNTS)[i];
  if(reset){s.q(HP)[j]=(chaser?s.q(CHITS)[i]:12)*(active?1:0);s.f(SPEED)[j]=(60+40*s.f(RCHSPEED)[j])*(chaser?1.f:0.f);}
  if(chaser||!active)s.q(TIMER)[j]=1000000;
 }
 // Original suite does this for *all* lanes, even if mask is entirely false.
 target_velocity(s,i);
}
FN void prepare_tick(S s,int i){
 for(int a=0;a<2;a++)s.f(OLDPOS)[i*2+a]=s.f(POS)[i*2+a];
 for(int e=0;e<4;e++){s.b(ALIVE)[i*4+e]=s.q(HP)[i*4+e]>0;for(int a=0;a<2;a++)s.f(OLDTARGET)[(i*4+e)*2+a]=s.f(TARGET)[(i*4+e)*2+a];}
}
