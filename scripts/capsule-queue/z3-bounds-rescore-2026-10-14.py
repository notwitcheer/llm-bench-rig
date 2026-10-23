import re,glob,os
from math import comb
S=re.compile(r"t=\s*([\d.]+)s step=\s*\d+ trunk_z=\s*([-\d.]+)mm grav_b=\(([-+\d.]+),([-+\d.]+),([-+\d.]+)\).*?r=([\d.]+)m")
def one(a,n1,b,n2):
    K=a+b;N=n1+n2;return sum(comb(n1,x)*comb(n2,K-x) for x in range(a,min(n1,K)+1))/comb(N,K)
res={}
for c in "abc":
    for f in sorted(glob.glob(os.path.expanduser(f"~/Pollen/clips/proof-z3-{c}-2026-10-14/*-t*.log"))):
        if f.endswith(".full.log"): continue
        t=open(f,errors="replace").read(); n=os.path.basename(f)[:-4]; pol=n.split("-")[0]
        v=[(float(m[1]),float(m[5]),float(m[6])) for m in S.finditer(t) if float(m[1])<=19.9]
        rs=[float(y) for y in re.findall(r"RESET at t=\s*([\d.]+)",t) if float(y)<19.9]
        falls=[];st="up"
        for tt,gz,r in v:
            if st=="up" and gz>-0.5: falls.append(tt); st="down"
            elif st=="down" and gz<-0.9: st="up"
        upr=[x for x in rs if (lambda w: w and max(s[1] for s in w)<-0.9)([s for s in v if x-0.5<=s[0]<=x])]
        first_upr=min(upr) if upr else None
        # take truncated by an upright reset: count as held if no fall/real reset BEFORE that reset
        real=[x for x in rs if x not in upr]
        before=[x for x in falls+real if first_upr is None or x<first_upr]
        held_orig = not falls and not rs
        held_trunc = not before
        res.setdefault((c,pol),[]).append((n,held_orig,held_trunc,first_upr))
for c in "abc":
    for pol in ("z3","v2"):
        l=res[(c,pol)]; print(c,pol,"held orig %d/%d, if upright resets are truncations (held up to the truncation) %d/%d"%(sum(x[1] for x in l),len(l),sum(x[2] for x in l),len(l)), [ (x[0],x[3]) for x in l if x[3] is not None])
for lab,cs in (("a+b","ab"),):
    a=sum(x[2] for c in cs for x in res[(c,"z3")]); b=sum(x[2] for c in cs for x in res[(c,"v2")])
    print(lab,"rescored z3 %d/30 vs v2 %d/30 one-sided p %.4f"%(a,b,one(a,30,b,30)))
