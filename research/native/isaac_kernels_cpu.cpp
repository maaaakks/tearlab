#include "isaac_kernels.h"
#ifdef _WIN32
#define API extern "C" __declspec(dllexport)
#else
#define API extern "C"
#endif
API void cpu_motion(void** p,const float* c,int n,const int64_t* a,int masked){for(int i=0;i<n;i++)motion(S{p,c,n},i,a[i],masked!=0);}
API void cpu_attacks(void** p,const float* c,int n,int masked){for(int i=0;i<n;i++)attacks(S{p,c,n},i,masked!=0);}
API void cpu_projectiles(void** p,const float* c,int n,const int64_t* a,int masked,int damage_ends){for(int i=0;i<n;i++)projectiles(S{p,c,n},i,a[i],masked!=0,damage_ends!=0);}
API void cpu_observation(void** p,const float* c,int n){for(int i=0;i<n;i++)observation(S{p,c,n},i);}
API void cpu_reset(void** p,const float* c,int n){for(int i=0;i<n;i++)reset_room(S{p,c,n},i);}
API void cpu_tick(void** p,const float* c,int n,const int64_t* a,int masked,int damage_ends){for(int i=0;i<n;i++){S s{p,c,n};attacks(s,i,masked!=0);motion(s,i,a[i],masked!=0);projectiles(s,i,a[i],masked!=0,damage_ends!=0);}}
API void cpu_decision(void** p,const float* c,int n,const int64_t* a,int damage_ends){for(int i=0;i<n;i++){S s{p,c,n};for(int t=0;t<3;t++){s.random_offset=t*n*4;prepare_tick(s,i);int64_t cmd=t<c[DELAY]?s.q(LAST)[i]:a[i];attacks(s,i,true);motion(s,i,cmd,true);projectiles(s,i,cmd,true,damage_ends!=0);}}}
