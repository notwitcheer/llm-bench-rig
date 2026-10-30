import ast, glob, os, sys
import pandas as pd
R=os.path.expanduser("~/esoteric/Honesty"); os.chdir(R); sys.path.insert(0,R+"/src")
src=open("scripts/56_xm_replay_extract.py").read()
tree=ast.parse(src)
ns={}
exec("import json, re\nimport numpy as np\nimport pandas as pd\nfrom cape.envs.base import TOOLS\nREADS = {\"list_files\", \"read_file\", \"open_mock_url\"}\n", ns)
for n in tree.body:
    if isinstance(n, ast.FunctionDef) and n.name=="action_events":
        exec(compile(ast.Module([n],[]),"x","exec"), ns)
ae=ns["action_events"]
keep=pd.read_parquet("outputs/v57/conf_episodes.parquet")
tot_act=tot_meta=0; act_meta=act_all=0
for m in sorted(glob.glob("outputs/xm_feats/xm_gptoss20b_*_meta.parquet")):
    st=os.path.basename(m).replace("_meta.parquet","")
    E=pd.read_parquet(f"outputs/rollouts/{st}.parquet")
    if st.endswith(("6789","10111213")):
        E=E[E.episode_id.isin(set(keep[keep.set==st].episode_id))]
    n_act_eps=0; n_acts=0
    for ep in E.itertuples():
        a=ae(ep)
        if a: n_act_eps+=1; n_acts+=len(a)
    M=pd.read_parquet(m); got=M.episode_id.nunique()
    tot_act+=n_act_eps; tot_meta+=got; act_all+=n_acts; act_meta+=len(M)
    print(f"{st}: eps_with_actions={n_act_eps} kept={got} skipped={n_act_eps-got} actions {len(M)}/{n_acts} max_ntok_kept={M.n_tokens.max()}")
print(f"TOTAL eps_with_actions={tot_act} kept={tot_meta} skipped={tot_act-tot_meta} actions_kept={act_meta}/{act_all}")
