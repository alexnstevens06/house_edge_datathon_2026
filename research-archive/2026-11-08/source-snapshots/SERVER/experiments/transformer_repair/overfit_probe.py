import os,sys,torch,time
sys.path.insert(0,'source/src')
from house_edge.data import load_split
from house_edge.windows import sliding_examples,WindowCollator
from house_edge.models import NextSpinTransformer,masked_ce_loss
torch.manual_seed(0);torch.set_num_threads(2)
d=load_split(sys.argv[1]+'/public/train',with_labels=True);ex=sliding_examples(d.episodes)[:256]
for arch in ['dense','sparse']:
  b={k:v.cuda() for k,v in WindowCollator(d.episodes,arch,64)(ex).items()}
  m=NextSpinTransformer(arch,dim_ff=256,dropout=0.0).cuda();o=torch.optim.AdamW(m.parameters(),lr=1e-3)
  for s in range(601):
    l=masked_ce_loss(m(b),b['labels'],b['label_mask']);o.zero_grad();l.backward();o.step()
    if s%150==0:print(os.environ.get('HE_POS'),arch,s,round(l.item(),4),flush=True)
