import json,os
os.chdir('/home/witcheer/benchmark-rig')
rows=[("qwen3-6-35b-a3b-ud-q5-k-m","Qwen3.6-35B-A3B UD-Q5_K_M"),("qwen3-6-27b-q6-k","Qwen3.6-27B Q6_K"),("qwable-27b-q4-k-m","Qwable-27B Q4_K_M")]
ROWS_PATHS=[("qwen3-6-35b-a3b-ud-q5-k-m","/home/witcheer/models/qwen36-35b-base/Qwen3.6-35B-A3B-UD-Q5_K_M.gguf"),("qwen3-6-27b-q6-k","/home/witcheer/models/qwen36-mtp-redl/Qwen3.6-27B-Q6_K.gguf"),("qwable-27b-q4-k-m","/home/witcheer/models/Qwable/Qwable-27b_Q4_K_M.gguf")]
out=[]
for s,label in rows:
    r={"slug":s,"label":label}
    for x,key in (("thinkon","16k"),("thinkon-32k","32k")):
        d=f"results/{s}-{x}"
        det=f"{d}/math500_detail.json"
        c=json.load(open(f"{d}/math500_progress.json"))["completed"]; v=list(c.values())
        seg={"n":len(v),"correct":sum(1 for i in v if i.get("correct")),
             "capped_wrong":sum(1 for i in v if i.get("capped") and not i.get("correct")),
             "capped":sum(1 for i in v if i.get("capped"))}
        seg["finished_wrong"]=seg["n"]-seg["correct"]-seg["capped_wrong"]
        if os.path.exists(det):
            j=json.load(open(det)); seg.update(final=True,score=j["score"],capped_rate=j["capped_rate"],pf=j["parse_failures"],tpc=j["tokens_per_correct"],detail={k:v for k,v in j.items() if not isinstance(v,(list,dict))})
        else: seg["final"]=False
        r[key]=seg
    r["file_bytes"]=os.stat(os.path.realpath(dict(ROWS_PATHS)[s])).st_size if s in dict(ROWS_PATHS) else None
    out.append(r)
print(json.dumps(out))
