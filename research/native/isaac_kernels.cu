#include "isaac_kernels.h"
extern "C" __global__ void isaac_motion(void** p,const float* c,int n,const int64_t* action,int masked){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)motion(S{p,c,n},i,action[i],masked!=0);
}
extern "C" __global__ void isaac_attacks(void** p,const float* c,int n,int masked){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)attacks(S{p,c,n},i,masked!=0);
}
extern "C" __global__ void isaac_projectiles(void** p,const float* c,int n,const int64_t* action,int masked,int damage_ends){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)projectiles(S{p,c,n},i,action[i],masked!=0,damage_ends!=0);
}
__device__ void projectiles_warp(S s,int i,int lane,int64_t command,int masked,int damage_ends){
 if(masked&&!s.b(ACTIVE)[i])return;
 int fired=lane==0?emit(s,i,command):0;fired=__shfl_sync(0xffffffff,fired,0);
 __syncwarp();int enemy=lane<8?friendly(s,i,lane):-1;
 bool phit=hostile(s,i,lane);
 unsigned hitmask=__ballot_sync(0xffffffff,phit);int counts[4];
 for(int e=0;e<4;e++)counts[e]=__popc(__ballot_sync(0xffffffff,enemy==e));
 __syncwarp();if(lane==0)finish_contacts(s,i,counts,hitmask!=0,fired!=0,damage_ends!=0);
}
extern "C" __global__ void isaac_projectiles_warp(void** p,const float* c,int n,const int64_t* action,int masked,int damage_ends){
 int tid=blockIdx.x*blockDim.x+threadIdx.x,i=tid/32,lane=tid%32;if(i<n)projectiles_warp(S{p,c,n},i,lane,action[i],masked,damage_ends);
}
extern "C" __global__ void isaac_tick(void** p,const float* c,int n,const int64_t* action,int masked,int damage_ends){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;S s{p,c,n};
 attacks(s,i,masked!=0);motion(s,i,action[i],masked!=0);projectiles(s,i,action[i],masked!=0,damage_ends!=0);
}
extern "C" __global__ void isaac_decision(void** p,const float* c,int n,const int64_t* action,int damage_ends){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;S s{p,c,n};
 for(int t=0;t<3;t++){s.random_offset=t*n*4;prepare_tick(s,i);int64_t cmd=t<c[DELAY]?s.q(LAST)[i]:action[i];attacks(s,i,true);motion(s,i,cmd,true);projectiles(s,i,cmd,true,damage_ends!=0);}
}
extern "C" __global__ void isaac_decision_warp(void** p,const float* c,int n,const int64_t* action,int damage_ends){
 int tid=blockIdx.x*blockDim.x+threadIdx.x,i=tid/32,lane=tid%32;if(i>=n)return;S s{p,c,n};
 for(int t=0;t<3;t++){
  s.random_offset=t*n*4;int64_t cmd=t<c[DELAY]?s.q(LAST)[i]:action[i];
  if(lane==0){prepare_tick(s,i);attacks(s,i,true);motion(s,i,cmd,true);}__syncwarp();
  projectiles_warp(s,i,lane,cmd,1,damage_ends);__syncwarp();
 }
}
extern "C" __global__ void isaac_observation(void** p,const float* c,int n){
 int k=blockIdx.x*blockDim.x+threadIdx.x;if(k<n*236)((float*)p[OBS])[k]=observation_value(S{p,c,n},k/236,k%236);
}
extern "C" __global__ void isaac_reset(void** p,const float* c,int n){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)reset_room(S{p,c,n},i);
}
