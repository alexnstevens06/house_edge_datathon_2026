import sys,os,torch,json,hashlib,time
sys.path.insert(0,os.path.expanduser("~/Documents/code/datathon_2026/house-edge/experiments/mlp_cnn_hpo/source"))
import train as T, base_cnn as B
from house_edge.scoring import write_submission_csv
torch.set_num_threads(2)
for run in sys.argv[1:]:
    ck=torch.load(run+"/checkpoint.pt",map_location="cpu"); a=ck["config"]
    m=B.CNN(a["h"],a["layers"],3,a["drop"],a["W"],use_signal=False) if a["model"]=="cnn" else T.MLP(a["width"],a["depth"],a["drop"],a["W"])
    m.load_state_dict(ck["state"]); m.eval()
    te=B.load_split(os.path.join(B.PUB,"test")); hs=[]
    for k in range(2):
        t0=time.time(); Q,e=B.build_query(te,a["W"]); p=B.to_preds(B.predict(m,Q,"cpu"),e,te); dt=time.time()-t0
        write_submission_csv(f"/tmp/rep_{k}.csv",p); hs.append(hashlib.sha256(open(f"/tmp/rep_{k}.csv","rb").read()).hexdigest())
    saved=hashlib.sha256(open(run+"/test_predictions_cpu.csv","rb").read()).hexdigest()
    print(run.split("/")[-1],"params",sum(x.numel() for x in m.parameters()),"ckptKB",os.path.getsize(run+"/checkpoint.pt")//1024,"cpu2_s %.2f"%dt,"repeat_identical",hs[0]==hs[1],"matches_saved",hs[0]==saved,saved[:16])
