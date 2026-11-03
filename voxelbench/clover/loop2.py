import json, sys, time, urllib.request, subprocess, os, re
port=sys.argv[1]; out=sys.argv[2]; cache=sys.argv[3]=="1"; logf=sys.argv[4]
W="/tmp/clover-loop"; os.makedirs(W,exist_ok=True)
tools=[{"type":"function","function":{"name":n,"description":d,"parameters":{"type":"object","properties":p,"required":list(p)}}} for n,d,p in [
 ("write_file","Write content to a file",{"path":{"type":"string"},"content":{"type":"string"}}),
 ("read_file","Read a file",{"path":{"type":"string"}}),
 ("terminal","Run a shell command and return its output",{"command":{"type":"string"}})]]
def run_tool(n,a):
    try:
        if n=="write_file": open(a["path"],"w").write(a["content"]); return json.dumps({"bytes_written":len(a["content"]),"ok":True})
        if n=="read_file": return open(a["path"]).read()[:4000]
        if n=="terminal":
            r=subprocess.run(a["command"],shell=True,capture_output=True,text=True,timeout=30,cwd=W); return json.dumps({"output":(r.stdout+r.stderr)[:3000],"exit_code":r.returncode})
    except Exception as e: return json.dumps({"error":str(e)})
    return json.dumps({"error":"unknown tool"})
msgs=[{"role":"system","content":"You are a coding agent. Use your tools. Work step by step; verify your work by running it."},
 {"role":"user","content":os.environ.get("TASK") or f"In {W} write primes.py that prints the first 10 primes one per line, run it with the terminal tool, then write test_primes.py that asserts the 10th prime is 29 and run it too. Report the result in one sentence."}]
URL=f"http://127.0.0.1:{port}/v1/chat/completions"; res=[]; 
for turn in range(1,9):
    b={"model":"x","messages":msgs,"tools":tools,"max_tokens":int(os.environ.get("MT","1200")),"cache_prompt":cache}
    b.update(json.loads(os.environ.get("SAMP","{}")))
    t=time.time()
    try: d=json.load(urllib.request.urlopen(urllib.request.Request(URL,json.dumps(b).encode(),{"Content-Type":"application/json"}),timeout=900))
    except Exception as e:
        print(json.dumps({"turn":turn,"error":str(e)[:200],"body":getattr(e,"read",lambda:b"")().decode()[:300]}),flush=True); break
    s=time.time()-t
    ch=d["choices"][0]; m=ch["message"]; tc=m.get("tool_calls") or []; tm=d.get("timings",{})
    rec={"turn":turn,"s":round(s,1),"finish":ch["finish_reason"],"completion_tokens":d["usage"]["completion_tokens"],"prompt_tokens":d["usage"]["prompt_tokens"],"tok_s":round(tm.get("predicted_per_second",0),1),"draft_n":tm.get("draft_n"),"draft_acc":tm.get("draft_n_accepted"),"calls":[c["function"]["name"] for c in tc],"reasoning_chars":len(m.get("reasoning_content") or ""),"content":(m.get("content") or "")[:160]}
    rec["tail"]=(tc[0]["function"]["arguments"] if tc else (m.get("content") or ""))[-240:]
    print(json.dumps(rec),flush=True); res.append(rec); json.dump({"turns":res,"msgs":msgs,"last":m},open(out,"w"))
    if not tc: break
    msgs.append({"role":"assistant","content":m.get("content") or "","reasoning_content":m.get("reasoning_content") or "","tool_calls":tc})
    for c in tc:
        try: a=json.loads(c["function"]["arguments"])
        except Exception as e: a={}; r=json.dumps({"error":"bad args "+str(e)})
        else: r=run_tool(c["function"]["name"],a)
        msgs.append({"role":"tool","tool_call_id":c["id"],"content":r})
json.dump({"turns":res,"msgs":msgs},open(out,"w"))
print("LOOP-DONE turns",len(res),"final_has_calls",bool(tc))
