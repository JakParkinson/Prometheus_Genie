struct MuonParam {
    float energy_center, track_center, mu, sigma;
};

vector<MuonParam> muon_params;
bool muon_params_loaded = false;

void load_muon_params() {
    if (muon_params_loaded) return;
    
    ifstream file("/groups/icecube/jackp/prometheus_genie_cleaned/harvard-prometheus/resources/PPC_executables/PPC_UPGRADE/muon_secondary_params_fine.txt");
    if (!file.is_open()) {
        cerr << "Warning: muon_secondary_params_fine.txt not found, using original parameterization" << endl;
        muon_params_loaded = true;
        return;
    }
    
    string line;
    getline(file, line);
    
    MuonParam param;
    while (file >> param.energy_center >> param.track_center >> param.mu >> param.sigma) {
        muon_params.push_back(param);
    }
    file.close();
    
    muon_params_loaded = true;
    cerr << "Loaded " << muon_params.size() << " muon secondary parameters" << endl;
}

float sample_muon_secondary(float energy_gev, float track_distance_m) {
    if (!muon_params_loaded) load_muon_params();
    if (muon_params.empty()) return -1.0f;
    
    // closest bin
    float min_distance = 1e30f;
    const MuonParam* closest = nullptr;
    
    for (const auto& param : muon_params) {
        float log_energy_dist = fabsf(logf(energy_gev) - logf(param.energy_center));
        float log_track_dist = fabsf(logf(track_distance_m) - logf(param.track_center));
        float distance = sqrtf(log_energy_dist*log_energy_dist + log_track_dist*log_track_dist);
        
        if (distance < min_distance) {
            min_distance = distance;
            closest = &param;
        }
    }
    
    if (!closest) return -1.0f;
    
    cerr << "energy_gev " << energy_gev << ", track_distance_m" << track_distance_m 
         << "closest mu : " << closest->mu << ", closest sigma: " << closest->sigma << endl;
    
    float normal_sample = grnd(); // grnd() samples from a N(0,1)
    return expf(closest->mu + closest->sigma * normal_sample);
}





float xrnd(){
  unsigned int rnd;
  do rnd=rand_r(&sv); while(rnd==0);
  const float a=1.0f/(1ll+RAND_MAX);
  return a*rnd;
}

float grnd(){  // gaussian distribution
  return sqrtf(-2*logf(xrnd()))*sinf(2*FPI*xrnd());
}

static const float rho=0.9216f;      // density of ice [mwe]
static const float m0=0.105658389f;  // muon rest mass [GeV]

photon p, pfl;

float yield(float E, float dr, int type){
  p.n.w=dr;
  float nph=0;

  /**
   * a,b describe the longitudinal profile of cascades.
   * The energy dependence of a is given by p1+p2*logE and b is constant.
   * Add comment about validity of parameterization for low-energy cascades.
   *
   * Total light yield of cascades:
   * For e-,e+,gamma the light yield is given by p*simulated_density/ice_density.
   * For hadrons the light yield is scaled down by Fi where i denotes the particle type.
   * The parameterizations have been derived from fits in an energy range
   * from 30 GeV to 10 TeV. Below approximately 10 GeV the F is not described
   * by F = 1-(E/E0)^(-m)*(1-f0) and is therefore a rather crude approximation.
   * Fluctuations have been parameterized with sigma/F = rms0*ln(E[GeV])^(-gamma).
   * For antiprotons the annihilation cross section has to be taken into account
   * at small energies. At the moment F for protons is used but with the fluctuations
   * of antiprotons.
   *
   * Reference: icecube/201210001 (for hadrons the values are recalculated)
   * The type are as following:
   * type  particle
   * 11 standard em (same as e-) is default if no other type is given
   * 12 e-
   * 13 e+
   * 14 gamma
   * 101 standard hadron (same as pi+)
   * 102 pi+
   * 103 pi-
   * 104 kaon0L
   * 105 proton
   * 106 neutron
   * 107 anti_proton Use F for protons because parameterization fails for small
   *     energies due to annihilation cross section. However sigma_F is used from
   *     anti_proton parameterization.
   **/

  if(type<0){  // LED light
    nph=E/dppm;
  }
  else{
    float logE=logf(max(m0, E));



    if(type<10){  // bare muon (0) endpoint (1)
      float secondary_ratio;
      
      // Try to use log-normal lookup for muons (type 0 = amu-)
      if(type == 0) {
        float sampled = sample_muon_secondary(E, dr);
        cerr << "sampled: " << sampled << endl;
        if(sampled >= 0) {
          // Use sampled value, ensuring it's non-negative
          secondary_ratio = max(0.0f, sampled);
        } else {
          // Fallback to original parameterization
          secondary_ratio = max(0.0f, 0.1880f+0.0206f*(logE-type)) * 0.910f/rho;
        }
      } else {
        // Use original parameterization for endpoint (type 1) or if lookup failed
        secondary_ratio = max(0.0f, 0.1880f+0.0206f*(logE-type)) * 0.910f/rho;
      }
      
      float extr = 1 + secondary_ratio;
      nph = dr>0 ? dr*extr : 0;
      p.f = 1/extr;
      p.a = 0, p.b = 0;
    }
    else{  // cascades
      const float Lrad=0.39652*0.910f/rho;
      const float em=5.321*0.910f/rho;  // 0.910 density used in simulation

      nph=em*E;

      if(type<100){
	switch(type){
	default:
	case 10:   // em shower
	case 11:   // delta/brems/epair
	case 12:   // e-
	  p.a=2.01849f+0.63176f*logE, p.b=Lrad/0.63207f;
	  break;

	case 13:   // e+
	  p.a=2.00035f+0.63190f*logE, p.b=Lrad/0.63008f;
	  break;

	case 14:   // gamma
	  p.a=2.83923f+0.58209f*logE, p.b=Lrad/0.64526f;
	  break;
	}
      }
      else{
	float E0, m, f0, rms0, gamma;

	switch(type){
	default:
	case 101: // standard hadron (same as pi+)
	case 102: // pi+
	  p.a=1.58357292f+0.41886807f*logE, p.b=Lrad/0.33833116f;
	  E0=0.18791678f;
	  m =0.16267529f;
	  f0=0.30974123f;
	  rms0 =0.95899551f;
	  gamma=1.35589541f;
	  break;

	case 103: // pi-
	  p.a=1.69176636f+0.40803489f*logE, p.b=Lrad/0.34108075f;
	  E0=0.19826506f;
	  m =0.16218006f;
	  f0=0.31859323f;
	  rms0 =0.94033488f;
	  gamma=1.35070162f;
	  break;

	case 104: // kaon0L
	  p.a=1.95948974f+0.34934666f*logE, p.b=Lrad/0.34535151f;
	  E0=0.21687243f;
	  m =0.16861530f;
	  f0=0.27724987f;
	  rms0 =1.00318874f;
	  gamma=1.37528605f;
	  break;

	case 105: // proton
	  p.a=1.47495778f+0.40450398f*logE, p.b=Lrad/0.35226706f;
	  E0=0.29579368f;
	  m =0.19373018f;
	  f0=0.02455403f;
	  rms0 =1.01619344f;
	  gamma=1.45477346f;
	  break;

	case 106: // neutron
	  p.a=1.57739060f+0.40631102f*logE, p.b=Lrad/0.35269455f;
	  E0=0.66725124f;
	  m =0.19263595f;
	  f0=0.17559033f;
	  rms0 =1.01414337f;
	  gamma=1.45086895f;
	  break;

	case 107: // anti_proton
	  p.a=1.92249171f+0.33701751f*logE, p.b=Lrad/0.34969748f;
	  E0=0.29579368f;
	  m =0.19373018f;
	  f0=0.02455403f;
	  rms0 =1.01094637f;
	  gamma=1.50438415f;
	  break;
	}

	{
	  float e=max(2.71828183f, E), f;
	  float F=1-powf(e/E0, -m)*(1-f0);
	  float dF=F*rms0*powf(logf(e), -gamma);
	  do f=F+dF*grnd(); while(f<0 || 1.1<f);
	  nph*=f;
	}
      }

      p.f=0;
      if(dr>0) p.a=0, p.b=0;
    }
  }

  return q.eff*nph;
}

#ifdef XLIB
struct mcid:pair<int,unsigned long long>{
  int frame;
  double t0;

  bool operator< (const mcid & rhs) const {
    return
      frame!=rhs.frame ? frame<rhs.frame :
      first!=rhs.first ? first<rhs.first :
      second!=rhs.second ? second<rhs.second :
      t0<rhs.t0;
  }
};

struct ihit{
  mcid track;
  ikey omkey;
  int pmt;
  float dir;
  float time;

  bool operator< (const ihit & rhs) const {
    return
      track!=rhs.track ? track<rhs.track :
      omkey!=rhs.omkey ? omkey<rhs.omkey :
      pmt!=rhs.pmt ? pmt<rhs.pmt :
      dir!=rhs.dir ? dir<rhs.dir :
      time<rhs.time;
  }
} tmph;

float trz=0;

void set_res(float t){
  trz=t;
}

float crz=0;

void set_res(float t, float c){
  trz=t; crz=c;
}

float res(float val, float bin){
  return bin>0?(floor(val/bin)+0.5)*bin:val;
}

deque<mcid> flnz;

struct pout{
  float r[4], n[4];
};
map<ihit, vector<pout> > hitz;
#else
deque<string> flnz;
#endif

unsigned int flnb=0, flne=0;
#ifdef XCPU
unsigned int & flnd = flne;
#else
unsigned int flnd=0;
#endif

#ifdef DTMN
int sign(float x){ return x<0?-1:x>0?1:0; }
#endif

int hcmp(const void *a, const void *b){
  hit & ah = * (hit *) a;
  hit & bh = * (hit *) b;
#ifdef DTMN
  return ah.n!=bh.n ? (int)(ah.n-bh.n) : ah.i!=bh.i ? (int)(ah.i-bh.i) : ah.t!=bh.t ? sign(ah.t-bh.t) : sign(ah.z-bh.z);
#else
  return (int) ( ah.n - bh.n );
#endif
}

void print(){
#ifndef DTMN
  if((int) ( flnb - flnd ) < 0)
#endif
    qsort(q.hits, d.hidx, sizeof(hit), hcmp);

  for(unsigned int i=0; i<d.hidx; i++){
    hit & h = q.hits[i];
    if((int)(h.n-flnb)<0 || (int)(h.n-flnd)>0){
      cerr<<"Internal Error: "<<h.n<<" is not between "<<flnb<<" "<<flnd<<" "<<flne<<endl;
      continue;
    }

    for(; (int) ( flnb - h.n ) < 0; flnb++){
#ifdef XLIB
      tmph.track=flnz.front();
#else
      cout<<flnz.front()<<endl;
#endif
      flnz.pop_front();
    }

    name n=q.names[h.i];

    bool flag=n.hv>0;

    if(flag){
      irde & mt = irdes[make_pair(n.omt,n.type)];
      flag=mt.rde>0&&xrnd()<mt.rat[h.z]*n.rde/mt.rmax;
    }

    float nx, ny, nz, rx, ry, rz;

    if(flag){
      float F=q.oms[h.i].F;
      float ptc=  cosf(h.pth), pts=      sinf(h.pth), ppc=cosf(h.pph), pps=sinf(h.pph);
      float dtc=F*cosf(h.dth), dts=F<0?1:sinf(h.dth), dpc=cosf(h.dph), dps=sinf(h.dph);
      rx=dts*dpc, ry=dts*dps, rz=dtc;
      nx=pts*ppc, ny=pts*pps, nz=ptc;
    }

    int pmt;

    if(flag){
      map<int, itype>::iterator it=types.find(n.omt);
      if(it!=types.end()){
	V<3> dir; dir[0]=nx, dir[1]=ny, dir[2]=nz;
	V<3> pos; pos[0]=rx, pos[1]=ry, pos[2]=rz;
	pmt=it->second.getPMT(dir, pos, n.tilt, xrnd(), n.azi);
	flag=pmt>=0;
      }
      else flag=false;
    }

    if(flag){
      float omr=q.oms[h.i].R;

      float rc=0.023f;
      float dc=omr+rc;

      float ph=n.azi;
      if(ph>=0){
	float drx=-omr*rx-dc*cos(fcv*ph);
	float dry=-omr*ry-dc*sin(fcv*ph);
	float a=nx*nx+ny*ny;
	if(a>0){
	  float b=-2*(nx*drx+ny*dry);
	  float c=drx*drx+dry*dry-rc*rc;
	  float D=b*b-4*a*c;
	  if(D>=0){
	    float h1=(-b+sqrt(D))/(2*a);
	    if(h1>0) flag=false;
	  }
	}
      }
    }

    if(flag){
      float wv=q.wvs[h.z].x();
#ifdef XLIB
      tmph.omkey=n;
      tmph.pmt=pmt;
      tmph.time=res(h.t, trz);
      tmph.dir=res(1+cos(h.pth), crz)-1;

      pout p;
      float omr=q.oms[h.i].R;

      p.r[0]=-omr*rx;
      p.r[1]=-omr*ry;
      p.r[2]=-omr*rz;
      p.r[3]=h.t;
      p.n[0]=nx;
      p.n[1]=ny;
      p.n[2]=nz;
      p.n[3]=wv;
      hitz[tmph].push_back(p);
#else
	printf("HIT %d %d %f %f %f %f %f %f\n", n.str, n.dom, h.t, wv, h.pth, h.pph, h.dth, h.dph);
  //     if(nextgen){
	// printf("HIT %d %d_%d %f %f %f %f %f %f\n", n.str, n.dom, pmt, h.t, wv, h.pth, h.pph, h.dth, h.dph);
  //     }
  //     else{
	// printf("HIT %d %d %f %f %f %f %f %f\n", n.str, n.dom, h.t, wv, h.pth, h.pph, h.dth, h.dph);
  //     }
#endif
    }
  }

  for(; (int) ( flnb - flnd ) < 0; flnb++){
#ifdef XLIB
    tmph.track=flnz.front();
#else
    cout<<flnz.front()<<endl;
#endif
    flnz.pop_front();
  }
}

void output(){
#ifdef USE_I3_LOGGING
  log_info_stream(pn << " ("<<pmax<<") photons from " << pk << " ("<<pmxo<<") tracks");
#endif

  kernel(pn); pn=0; pk=0;

#ifndef XCPU
  flnd=flne;
#endif
}

void addh(unsigned long long num){
  do{
    p.num=min((unsigned long long) (pmax-pn), num);
    num-=p.num; pn+=p.num; q.pz[pk++]=p;
    if(pn==pmax || pk==pmxo) output();
  } while(num>0);

  if(pn>0 && pk==0) q.pz[pk++]=p;
}

double gammln(double xx){  // gamma function
  static const double stp = 2.5066282746310005;
  static const double cof[6] = {76.18009172947146, -86.50532032941677, 24.01409824083091,
			  -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5};

  double ser=1, x=xx-1;
  double tmp=x+5.5;

  tmp-=(x+0.5)*log(tmp);
  for(int j=0; j<=5; j++){
    x++;
    ser+=cof[j]/x;
  }
  return -tmp+log(stp*ser);
}

unsigned long long poidev(double xm){  // sample Poisson distribution with mean xm
  double sq, alxm, g, t, y;
  unsigned long long um;

  if(xm<12){
    g=exp(-xm);
    um=-1ULL;
    t=1;
    do{
      um++;
      t*=xrnd();
    } while(t>g);
  }
  else{
    sq=sqrt(2*xm);
    alxm=log(xm);
    g=xm*alxm-gammln(xm+1);

    do{
      double em;
      do{
	y=tan(FPI*xrnd());
	em=sq*y+xm;
      } while(em<0);
      um=floor(em);
      t=0.9*(1+y*y)*exp(um*alxm-gammln(um+1)-g);
    } while(xrnd()>t);
  }

  return um;
}

unsigned long long bnldev(unsigned long long n, double pp){  // down-sample with binomial distribution
  unsigned long long j, bnl;
  double am, em, g, angle, p, sq, t, y;
  double pc, plog, pclog, en, oldg;

  p=(pp<=0.5 ? pp : 1-pp);
  am=n*p;

  if(n<25){
    bnl=0;
    for(j=1; j<=n; j++) if(xrnd()<p) bnl++;
  }
  else if(am<1){
    g=exp(-am);
    t=1;
    for(j=0; j<=n; j++){
      t*=xrnd();
      if(t<g) break;
    }
    bnl=(j<=n ? j : n);
  }
  else{
    en=n;
    oldg=gammln(en+1);
    pc=1-p;
    plog=log(p);
    pclog=log(pc);
    sq=sqrt(2*am*pc);
    do{
      do{
	angle=FPI*xrnd();
	y=tan(angle);
	em=sq*y+am;
      } while(em<0 || em>=(en+1));
      em=floor(em);
      t=1.2*sq*(1+y*y)*exp(oldg-gammln(em+1)-gammln(en-em+1)+em*plog+(en-em)*pclog);
    } while(xrnd()>t);
    bnl=em;
  }
  if(p!=pp) bnl=n-bnl;
  return bnl;
}

void addf(float rx, float ry, float rz, float t, unsigned long long num){
  p.r.x=rx; p.r.y=ry; p.r.z=rz; p.r.w=t; p.type=1;
  p.ka=0, p.up=0, p.fldr=-1; p.fla=-1; p.ofla=-1;
  p.n.w=1; addh(num);
}

void addp(float rx, float ry, float rz, float t, float E, float dr, int type, float scale = 1){
  p.r.x=rx; p.r.y=ry; p.r.z=rz; p.r.w=t;
  if(type<0){
    p.c=pfl.c;
    switch(type){
    case -1: dr=0; p.ka=-2; break; // linear cone around n
    case -2: dr=0;          break; // 2d gaussian around n
    case -3: dr=0; p.ka=-1; break; // isotropic
    case -4: // flset-configured flasher
    default: dr=pfl.n.w;
      p.type=pfl.type; p.r=pfl.r; p.n=pfl.n;
    }
  }
  else{
    p.type=-type;
    p.beta=1; p.tau=0;
  }

  unsigned long long num=poidev(scale*yield(E, dr, type)/ovr);
  cerr << "num photons: " << num << endl;
  //addh(0);
  addh(num);
}

#ifdef XLIB
void addp_clst(float rx, float ry, float rz, float t, unsigned long long n, float dr, float beta){
  p.type=0;
  p.r.w=t; p.r.x=rx; p.r.y=ry; p.r.z=rz;
  p.n.w=dr; p.f=1.f; p.a=0, p.b=0;
  p.beta=beta; p.tau=0;

  // when 0<qmas<1 assume n is given for DOM w/o the maximum of the OM sensitivy curve rescaling (i.e. as if qmas=1)
  // if other types of optimization downsampling are needed (i.e., other parts of q.eff), this is where they would go
  float qmas=1.f;
  unsigned long long num=qmas>0&&qmas<1?bnldev(n, qmas):n;
  addh(num);
}

void addp_mopo(float rx, float ry, float rz, float t, float light, float length, float beta, float tau, float fr){
  p.type=0;
  p.r.w=t; p.r.x=rx; p.r.y=ry; p.r.z=rz;
  p.n.w=length; p.f=fr; p.a=0, p.b=0;
  p.beta=beta; p.tau=tau;

  unsigned long long num=poidev(light*q.eff/ovr);
  addh(num);
}
#endif

void finc(){
  flne++;
  if((int) ( flne - flnb ) < 0){ cerr<<"Error: too many event segments"<<endl; exit(3); }
}

void eout(){
  output();
#ifndef XCPU
  output();
#endif
}

#ifdef XLIB
void efin(){
  hitz.erase(hitz.begin(), hitz.end());
}

void sett(float nx, float ny, float nz, pair<int,unsigned long long> id, int frame, double t0 = 0){
  mcid ID; ID.first=id.first; ID.second=id.second; ID.frame=frame; ID.t0=t0;
  flnz.push_back(ID); finc();
  p.q=flne; p.n.x=nx; p.n.y=ny; p.n.z=nz;
}
#else

void f2k(){
  ini();

  string in;
  while(getline(cin, in)){
    cerr << in << endl;
    flnz.push_back(in); finc();
    char name[32];
    int gens, igen, type=-1;
    float x, y, z, th, ph, l, E, t;
    const char * str = "TR %d %d %31s %f %f %f %f %f %f %f %f";

    if(sscanf(in.c_str(), str, &gens, &igen, name, &x, &y, &z, &th, &ph, &l, &E, &t)==11){
   //   th=fcv*(180-th); ph=fcv*(ph<180?ph+180:ph-180);
      // th=180-th; ph=ph<180?ph+180:ph-180;
      // th*=FPI/180; ph*=FPI/180;
      
      float costh=cosf(th), sinth=sinf(th), cosph=cosf(ph), sinph=sinf(ph);
      p.q=flne; p.n.x=sinth*cosph; p.n.y=sinth*sinph; p.n.z=costh;
      if(0==strcmp(name, "amu+") || 0==strcmp(name, "amu-") || 0==strcmp(name, "amu")) type=0;
      else if(0==strcmp(name, "delta") || 0==strcmp(name, "brems") ||
	      0==strcmp(name, "epair") || 0==strcmp(name, "e")) type=11;
      else if(0==strcmp(name, "e-")) type=12;
      else if(0==strcmp(name, "e+")) type=13;
      else if(0==strcmp(name, "munu") || 0==strcmp(name, "hadr")) type=101;
      else if(0==strcmp(name, "ph")) addf(x, y, z, t, E);
      if(type>=0) addp(x, y, z, t, E, l, type);
    }
  }
  eout();

  fin();
}
#endif

void flone(unsigned long long num){
  addh(num*(q.eff/dppm)/ovr);
}

void flshift(float m[], float n[], float *r = NULL){
  if(isnan(n[2])){
    float xi;
    sincosf(p.up, &n[2], &xi);
    n[0]*=xi; n[1]*=xi;
  }

  float q[3], o[3]={0,0,1};

  q[0]=m[1]*o[2]-m[2]*o[1]; // m[1]
  q[1]=m[2]*o[0]-m[0]*o[2]; //-m[0]
  q[2]=m[0]*o[1]-m[1]*o[0]; // 0

  float norm=sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]);
  if(norm>0){
    float cs=0;
    for(int i=0; i<3; i++) q[i]/=norm, cs+=o[i]*m[i];
    float sn=sin(acos(cs)); //norm

    float R[3][3]={{0}};
    for(int i=0; i<3; i++)
      for(int j=0; j<3; j++)
	R[i][j]=(i==j?cs:sn*q[3-i-j]*((j-i+3)%3==1?1:-1))+(1-cs)*q[i]*q[j];

    for(int i=0; i<3; i++){
      q[i]=0;
      for(int j=0; j<3; j++) q[i]+=R[i][j]*n[j];
    }
    for(int i=0; i<3; i++) n[i]=q[i];

    if(r!=NULL){
      for(int i=0; i<3; i++){
	q[i]=0;
	for(int j=0; j<3; j++) q[i]+=R[i][j]*r[j];
      }
      for(int i=0; i<3; i++) r[i]=q[i];
    }
  }
}

const DOM& flset(int str, int dom){
  {
    p.fla=-1, p.ofla=-1;
    p.ka=0, p.up=0, p.fldr=-1;
    p.n.x=0, p.n.y=0, p.n.z=1, p.n.w=0;
  }

  int type=1;
  static DOM om;

  {
    float r[3]={0, 0, 0};

    if(str<0){ type=2; str=-str; }
    if(str==0) switch(dom){ // standard candles
      case 1: type=3; r[0]=544.07; r[1]=55.89; r[2]=136.86; break;
      case 2: type=4; r[0]=11.87; r[1]=179.19; r[2]=-205.64; break;
      }
    else for(int n=0; n<d.gsize; n++) if(q.names[n].str==str && q.names[n].dom==dom){
	  p.fla=n;
	  for(int m=0; m<3; m++) r[m]=q.oms[n].r[m]; break;
	}
    cerr<<"Flasher configured at "<<r[0]<<", "<<r[1]<<", "<<r[2]<<endl;

    p.type=type;
    p.r.x=r[0], p.r.y=r[1], p.r.z=r[2], p.r.w=0;
    for(int m=0; m<3; m++) om.r[m]=r[m];
  }

  float fzcr=0.f;
  {
    char * FZCR=getenv("FZCR");
    if(FZCR!=NULL){ fzcr=atof(FZCR);
      cerr<<"Adjusting flasher LED tilt by "<<fzcr<<" degrees"<<endl;
    }
  }

  switch(type){
  case 1: p.up=(-0.2f+fzcr)*fcv; break;
  case 2: p.up=(48.1f+fzcr)*fcv; break;
  case 3: p.up=(90.0f-41.13f)*fcv; break;
  case 4: p.up=(41.13f-90.0f)*fcv; break;
  }

  {
    char * ELEV=getenv("ELEV");
    if(ELEV!=NULL){
      float elev=atof(ELEV);
      cerr<<"Setting LED beam elevation angle to "<<elev<<" degrees"<<endl;
      p.up=elev*fcv;
    }
  }

  {
    p.ofla=p.fla;
    char * ofla=getenv("OFLA");
    if(ofla!=NULL) if(*ofla!=0 && atoi(ofla)==0){
	p.ofla=-1;
	cerr<<"Flasher DOM will register photons!"<<endl;
      }
  }

  {
    char * FLDR=getenv("FLDR");
    p.fldr=FLDR==NULL?-1:atof(FLDR);
    if(p.fldr>=0){
      float fold=int(p.fldr/360);
      float dir=p.fldr-360*fold++;
      cerr<<"Flasher LEDs are in a "<<fold<<"-fold pattern with LED #0 at "<<dir<<" degrees"<<endl;
    }
  }

  float2 v;
  {
    float xi=p.fldr*fcv;
    sincosf(xi, &v.y, &v.x);
    sincosf(p.up, &p.n.z, &xi);
    p.n.x=xi*v.x; p.n.y=xi*v.y;
  }

  float fwid=9.7f;
  {
    char * FWID=getenv("FWID");
    if(FWID!=NULL){ fwid=atof(FWID);
      cerr<<"Setting flasher beam width to "<<fwid<<" degrees"<<endl;
    }
  }

  if(fwid>999.f){
    p.ka=fwid;

    if(p.fla>=0 && p.fldr>=0 && p.fldr<360){
      char * bmp=getenv("FTLT");
      if(bmp!=NULL){
	float m[3]={0,0,1};
	float nx, ny, nz;
	int num=sscanf(bmp, "%f %f %f", &nx, &ny, &nz);
	if(num==3){
	  cerr<<"Applying tilt to flasher DOM of ";
	  m[0]=nx, m[1]=ny, m[2]=nz;
	}
	else{
	  ikey om; om.str=str, om.dom=dom;
	  if(cx.find(om)!=cx.end()){
	    V<3>& n = cx[om];
	    for(int i=0; i<3; i++) m[i]=n[i];
	    cerr<<"Setting flasher tilt from file cx.dat to ";
	  }
	  else cerr<<"Leaving flasher tilt at default of ";
	}
	cerr<<m[0]<<", "<<m[1]<<", "<<m[2]<<endl;

	p.n.w=1;
	const float FLZ=0.07735f, FLR=0.119f-0.008f;
	p.r.x=FLR*v.x, p.r.y=FLR*v.y, p.r.z=FLZ;

	float n[3]={p.n.x, p.n.y, p.n.z};
	float r[3]={p.r.x, p.r.y, p.r.z};
	flshift(m, n, r);
	p.r.x=r[0], p.r.y=r[1], p.r.z=r[2];
	p.n.x=n[0], p.n.y=n[1], p.n.z=n[2];
      }
    }
  }
  else if(fwid>0) p.ka=square(fcv*fwid);
  else p.ka=fwid;

  pfl=p;
  return om;
}

#ifdef XLIB
const float * fldir(){
  static float dir[3] = {pfl.n.x, pfl.n.y, pfl.n.z};
  return dir;
}

#else
void flasher(int str, int dom, unsigned long long num, int itr){
  ini();
  flset(str, dom);

  for(int j=0; j<max(1, itr); j++){
    p.q=flne;
    flone(num);
    if(itr>0){
      flnz.push_back(""); finc();
    }
  }

  eout();

  fin();
}
#endif