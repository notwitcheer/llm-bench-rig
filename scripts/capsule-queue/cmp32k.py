import json,sys,os
os.chdir('/home/witcheer/benchmark-rig')
def load(d):
    p=f'results/{d}/math500_progress.json'
    c=json.load(open(p))['completed']
    return {str(k):v for k,v in c.items()}
for s in sys.argv[1:]:
    a=load(f'{s}-thinkon'); b=load(f'{s}-thinkon-32k')
    common=[k for k in a if k in b]
    capA=[k for k in common if a[k].get('capped')]
    unc=[k for k in common if not a[k].get('capped')]
    same_tok=sum(1 for k in unc if a[k].get('completion_tokens')==b[k].get('completion_tokens'))
    same_corr=sum(1 for k in unc if a[k].get('correct')==b[k].get('correct'))
    gainA=sum(1 for k in capA if b[k].get('correct'))
    stillcap=sum(1 for k in capA if b[k].get('capped'))
    lost=sum(1 for k in unc if a[k].get('correct') and not b[k].get('correct'))
    won_unc=sum(1 for k in unc if not a[k].get('correct') and b[k].get('correct'))
    print(s,'common',len(common),'16k-capped',len(capA),'-> correct at 32k',gainA,'still capped',stillcap,'| uncapped',len(unc),'same tokens',same_tok,'same correctness',same_corr,'lost',lost,'won',won_unc)
    print('  16k correct',sum(1 for k in common if a[k]['correct']),'32k correct',sum(1 for k in common if b[k]['correct']))
