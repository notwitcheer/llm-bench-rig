// paper-code-faithful reference for arXiv 2608.10844 (Masoumi, Kargaran, Jafari), reading the authors' Zenodo 10.5281/zenodo.20793454 (heat_map.cpp / main_functions.h).
// rule P: m_i = mean heading of self + all j within R (d^2 < R^2); h_i = mean over MUTUAL TRIANGLES (i,j,k all pairwise d^2 < R^2, self excluded from j,k)
// of 0.5*[(vk.vi) vj + (vj.vi) vk] (unit vectors); u = a*m + (1-a)*h; theta' = atan2(u) + eta*U(-1/2,1/2) (zero u keeps heading); then x += v*(cos,sin)(theta') (move with the NEW heading).
// usage: vicp N rho alpha seed eta start(0 random|1 ordered) steps window [mode]   mode 0 = paper-code rule P (default); 1 = text eq (1): star pairs within N_i (self incl), v0-normalised
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
static uint64_t s[4];
static uint64_t rotl(uint64_t x,int k){return (x<<k)|(x>>(64-k));}
static uint64_t nxt(void){uint64_t r=rotl(s[1]*5,7)*9,t=s[1]<<17;s[2]^=s[0];s[3]^=s[1];s[1]^=s[2];s[0]^=s[3];s[2]^=t;s[3]=rotl(s[3],45);return r;}
static void seed(uint64_t x){for(int i=0;i<4;i++){x+=0x9e3779b97f4a7c15ULL;uint64_t z=x;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;s[i]=z^(z>>31);}}
static double U(void){return (nxt()>>11)*(1.0/9007199254740992.0);}
#define MAXN 12000
static double X[MAXN],Y[MAXN],C[MAXN],S_[MAXN],MX[MAXN],MY[MAXN],HX[MAXN],HY[MAXN]; static int HC[MAXN],NB[MAXN][160],NN[MAXN];
static int N; static double L,V0=0.03,R2=1.0;
static inline double dd(double a,double b){double d=a-b; d-=L*round(d/L); return d;}
int main(int argc,char**argv){
  N=atoi(argv[1]);double rho=atof(argv[2]),al=atof(argv[3]);seed(atoll(argv[4]));double eta=atof(argv[5]);int ord=atoi(argv[6]),steps=atoi(argv[7]),win=atoi(argv[8]);int mode=argc>9?atoi(argv[9]):0;
  L=sqrt(N/rho);
  for(int i=0;i<N;i++){X[i]=U()*L;Y[i]=U()*L;double th=ord?0.0:(U()*2-1)*M_PI;C[i]=cos(th);S_[i]=sin(th);}
  double acc=0,acc2=0;int cnt=0;
  const char*sch=getenv("SCHED");double sch_eta[16];int sch_n[16],ns=0;int trace=getenv("TRACE")?atoi(getenv("TRACE")):0;
  if(sch){char buf[256];snprintf(buf,256,"%s",sch);for(char*tok=strtok(buf,",");tok;tok=strtok(NULL,",")){sscanf(tok,"%lf:%d",&sch_eta[ns],&sch_n[ns]);ns++;}steps=0;for(int q=0;q<ns;q++)steps+=sch_n[q];}
  for(int t=0;t<steps;t++){
    if(sch){int tt=t,q=0;while(tt>=sch_n[q]){tt-=sch_n[q];q++;}eta=sch_eta[q];}
    if(trace&&t%trace==0){double c=0,sn=0;for(int i=0;i<N;i++){c+=C[i];sn+=S_[i];}printf("t=%d eta=%.2f order=%.3f\n",t,eta,hypot(c,sn)/N);}
    for(int i=0;i<N;i++){NN[i]=0;}
    for(int i=0;i<N;i++)for(int j=i+1;j<N;j++){double dx=dd(X[i],X[j]),dy=dd(Y[i],Y[j]);if(dx*dx+dy*dy<R2){if(NN[i]<160&&NN[j]<160){NB[i][NN[i]++]=j;NB[j][NN[j]++]=i;}}}
    for(int i=0;i<N;i++){double sx=C[i],sy=S_[i];for(int a=0;a<NN[i];a++){sx+=C[NB[i][a]];sy+=S_[NB[i][a]];}MX[i]=sx/(NN[i]+1);MY[i]=sy/(NN[i]+1);HX[i]=0;HY[i]=0;HC[i]=0;}
    if(al<1.0){
      if(mode==0){
        for(int i=0;i<N;i++){int n=NN[i];for(int a=0;a<n;a++){int j=NB[i][a];if(j<i)continue;for(int b=a+1;b<n;b++){int k=NB[i][b];if(k<i)continue;
          double dx=dd(X[j],X[k]),dy=dd(Y[j],Y[k]);if(dx*dx+dy*dy>=R2)continue;
          double sij=C[i]*C[j]+S_[i]*S_[j],sik=C[i]*C[k]+S_[i]*S_[k],sjk=C[j]*C[k]+S_[j]*S_[k];
          HX[i]+=0.5*(sik*C[j]+sij*C[k]);HY[i]+=0.5*(sik*S_[j]+sij*S_[k]);HC[i]++;
          HX[j]+=0.5*(sij*C[k]+sjk*C[i]);HY[j]+=0.5*(sij*S_[k]+sjk*S_[i]);HC[j]++;
          HX[k]+=0.5*(sik*C[j]+sjk*C[i]);HY[k]+=0.5*(sik*S_[j]+sjk*S_[i]);HC[k]++;}}}
        for(int i=0;i<N;i++)if(HC[i]>0){HX[i]/=HC[i];HY[i]/=HC[i];}
      } else if(mode==2){ // GPU-code rule (Zenodo Main-Fig4 FSE.cu): star pairs j<k among the neighbours of i, SELF EXCLUDED, jk distance test commented out in the authors' code
        for(int i=0;i<N;i++){int n=NN[i];double ax=0,ay=0;long cnt=0;for(int a=0;a<n;a++){int j=NB[i][a];for(int b=a+1;b<n;b++){int k=NB[i][b];
          double sij=C[i]*C[j]+S_[i]*S_[j],sik=C[i]*C[k]+S_[i]*S_[k];ax+=0.5*(sik*C[j]+sij*C[k]);ay+=0.5*(sik*S_[j]+sij*S_[k]);cnt++;}}
          if(cnt>0){HX[i]=ax/cnt;HY[i]=ay/cnt;}}
      } else { // text eq (1): N_i includes i; pairs j<k in N_i (no mutual-distance test); h = sum[(vj.vi)vk+(vk.vi)vj]/|T|
        for(int i=0;i<N;i++){int n=NN[i]+1;int idx[170];idx[0]=i;for(int a=0;a<NN[i];a++)idx[a+1]=NB[i][a];
          double Sx=0,Sy=0,sd=0,swx=0,swy=0;for(int a=0;a<n;a++){int j=idx[a];double d=C[j]*C[i]+S_[j]*S_[i];Sx+=C[j];Sy+=S_[j];sd+=d;swx+=d*C[j];swy+=d*S_[j];}
          double nT=n*(n-1)/2.0;if(n>1){HX[i]=(sd*Sx-swx)/nT;HY[i]=(sd*Sy-swy)/nT;}}
      }
    }
    for(int i=0;i<N;i++){double ux=al*MX[i]+(1-al)*HX[i],uy=al*MY[i]+(1-al)*HY[i];
      double th=(fabs(ux)+fabs(uy)<1e-12)?atan2(S_[i],C[i]):atan2(uy,ux);th+=eta*(U()-0.5);C[i]=cos(th);S_[i]=sin(th);
      X[i]+=V0*C[i];Y[i]+=V0*S_[i];X[i]-=L*floor(X[i]/L);Y[i]-=L*floor(Y[i]/L);}
    if(t>=steps-win){double c=0,sn=0;for(int i=0;i<N;i++){c+=C[i];sn+=S_[i];}double v=hypot(c,sn)/N;acc+=v;acc2+=v*v;cnt++;}
  }
  printf("%.4f %.4f\n",acc/cnt,sqrt(fmax(0,acc2/cnt-(acc/cnt)*(acc/cnt))));
  return 0;}
