import glob,re,os
S=re.compile(r't=\s*([\d.]+)s step=\s*\d+ trunk_z=\s*([-\d.]+)mm grav_b=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\) r=([\d.]+)m')
for f in sorted(glob.glob(os.path.expanduser('~/Pollen/clips/clip-z7-fresh*-2026-10-22/zc*-t*.full.log'))):
    t=open(f,errors='replace').read()
    rw=re.search(r'after reset:.*root_w=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\)',t)
    res=[float(x) for x in re.findall(r'RESET at t=\s*([\d.]+)',t)]
    v=[(float(m[1]),float(m[2]),float(m[6]),float(m[5])) for m in S.finditer(t)]
    pts=[x for x in v if abs(round(x[0]*10)%20)==0]
    print(os.path.basename(f),'spawn',rw.groups() if rw else '?','RESETS',res,'last t',v[-1][0] if v else None,'worst z',max(x[3] for x in v) if v else None)
    print('   ',' '.join(f'{x[0]:.0f}:{x[1]:.0f}/{x[2]:.1f}' for x in pts))
