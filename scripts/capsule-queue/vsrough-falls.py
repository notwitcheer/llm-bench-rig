import glob,re,os,sys
S=re.compile(r't=\s*([\d.]+)s step=\s*\d+ trunk_z=\s*([-\d.]+)mm grav_b=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\).*?r=([\d.]+)m')
R=re.compile(r'RESET at t=\s*([\d.]+)')
d=os.path.expanduser(sys.argv[1]); groups={}
for f in sorted(glob.glob(d+'/*-t*.log')):
    if f.endswith('.full.log'): continue
    t=open(f,errors='replace').read(); v=[(float(m[1]),float(m[2]),float(m[5]),float(m[6])) for m in S.finditer(t) if float(m[1])<=19.9]
    falls=[];st='up'
    for (tt,z,gz,r) in v:
        if st=='up' and gz>-0.5: falls.append(tt); st='down'
        elif st=='down' and gz<-0.9: st='up'
    rs=[float(x) for x in R.findall(t) if float(x)<19.9]
    name=os.path.basename(f)[:-4]; g=name.rsplit('-t',1)[0]
    rmin=min(x[3] for x in v); rmax=max(x[3] for x in v)
    groups.setdefault(g,[]).append(len(falls)==0)
    print(f"{name:30s} falls={len(falls)} at {[round(x,1) for x in falls]} resets={rs} r={rmin:.2f}-{rmax:.2f}m end_gz={v[-1][2]:+.2f}")
for g,l in groups.items(): print(f"{g:28s} {sum(l)}/{len(l)} takes with no fall")
