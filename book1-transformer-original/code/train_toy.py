"""自建32句词典组合任务：只检验缩小模型的学习与因果生成。"""
from pathlib import Path
import argparse,itertools,json,time,hashlib
import torch
from model import Translator
from paths import RESEARCH as R
# PAD=0 BOS=1 EOS=2；四名词3..6，两动词7..8；两侧语言共用数值标签，嵌入参数分离。
DATA=[(s,v,o) for s in range(3,7) for v in range(7,9) for o in range(3,7)]
TEST=[i for i,(s,v,o) in enumerate(DATA) if (s+o+v)%4==0]
TRAIN=[i for i in range(len(DATA)) if i not in TEST]

def run(seed,pe,steps):
    torch.manual_seed(seed);model=Translator(use_positions=pe)
    source=torch.tensor(DATA);labels=torch.cat([source,torch.full((len(DATA),1),2)],1)
    inp=torch.cat([torch.ones(len(DATA),1,dtype=torch.long),labels[:,:-1]],1)
    optim=torch.optim.Adam(model.parameters(),lr=.003,betas=(.9,.98),eps=1e-9)
    history=[];start=time.perf_counter()
    for step in range(1,steps+1):
        model.train();optim.zero_grad();logits=model(source[TRAIN],inp[TRAIN])
        loss=torch.nn.functional.cross_entropy(logits.reshape(-1,9),labels[TRAIN].reshape(-1),label_smoothing=.1)
        loss.backward();optim.step()
        if step==1 or step%50==0:
            history.append({'step':step,'smoothed_train_loss':loss.item()})
            print('progress',seed,pe,step,loss.item(),flush=True)
    elapsed=time.perf_counter()-start;model.eval()
    with torch.no_grad():
        logits=model(source,inp);pred=model.generate(source)
        nll=torch.nn.functional.cross_entropy(logits[TEST].reshape(-1,9),labels[TEST].reshape(-1)).item()
        result=dict(seed=seed,positions=pe,steps=steps,train_seconds=elapsed,history=history,test_nll=nll,
                    train_exact=(pred[TRAIN]==labels[TRAIN]).all(-1).double().mean().item(),
                    test_exact=(pred[TEST]==labels[TEST]).all(-1).double().mean().item(),
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    examples=[dict(source=source[i].tolist(),expected=labels[i].tolist(),generated=pred[i].tolist()) for i in TEST])
        model(source[:1],inp[:1]);result['cross_attention_first_example']=model.decoder[-1].cross.last_weights[0].tolist()
    return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=400);args=ap.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    report=dict(task='人工词典组合；不是自然语言翻译/WMT',torch=torch.__version__,device='CPU',dtype='float32',
                train_indices=TRAIN,test_indices=TEST,data=DATA,settings=dict(d=32,h=4,layers=2,ff=64,drop=.1,lr=.003),results=[])
    for pe,seed in [(True,0),(True,1),(True,2),(False,0),(False,1),(False,2)]:
        result=run(seed,pe,args.steps);report['results'].append(result)
        report['code_hashes']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('model.py')]}
        (R/'toy-training.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
        print({k:result[k] for k in ['seed','positions','train_exact','test_exact','train_seconds']},flush=True)
