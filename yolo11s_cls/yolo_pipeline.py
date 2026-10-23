"""YOLO11s-cls on Teddy's fixed manifest. Smoke/train never open PlantDoc images."""
import argparse, csv, hashlib, json, random, time, platform
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from ultralytics import YOLO
from ultralytics.nn.tasks import ClassificationModel


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_manifest(path, mapping):
    labels = [c['label'] for c in json.loads(Path(mapping).read_text())['classes']]
    assert len(labels) == 12 and len(set(labels)) == 12
    rows = list(csv.DictReader(Path(path).open()))
    assert len({r['image_id'] for r in rows}) == len(rows), 'Duplicate image IDs'
    expected = {'train':12174, 'val':2609, 'test':2609, 'external':1097}
    assert {s:sum(r['split']==s for r in rows) for s in expected} == expected
    for r in rows:
        assert labels[int(r['y'])] == r['label']
        assert r['split'] in expected
        assert (r['root'],r['source']) == (('pd','plantdoc') if r['split']=='external' else ('pv','plantvillage'))
        assert r['image_id'] == r['root'] + '/' + r['relpath']
        assert not Path(r['relpath']).is_absolute() and '..' not in Path(r['relpath']).parts
    pv = [r for r in rows if r['root']=='pv']
    assert len({r['md5'] for r in pv}) == len(pv), 'Duplicate PV contents in shared manifest'
    return rows, labels


def subset(rows, split, per_class):
    selected, counts = [], {}
    for r in rows:
        if r['split'] != split: continue
        c = r['label']; counts[c] = counts.get(c,0)+1
        if not per_class or counts[c] <= per_class: selected.append(r)
    return selected


class Leaves(Dataset):
    def __init__(self, rows, root, names, transform, audit=True):
        self.rows, self.tf = rows, transform
        self.paths = [(Path(root)/r['relpath']).resolve() for r in rows]
        base = Path(root).resolve()
        for r,p in zip(rows,self.paths):
            assert p.is_relative_to(base) and p.is_file(), f'Missing/invalid path: {p}'
            if audit:
                assert digest(p,'md5') == r['md5'], f'Content differs from team manifest: {p}'
                with Image.open(p) as im: im.convert('RGB').load()
        self.targets = [names.index(r['label']) for r in rows]
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        with Image.open(self.paths[i]) as im: x = self.tf(im.convert('RGB'))
        return x,self.targets[i]


def transform(train=False):
    # YOLO11 pretrained checkpoint: zero mean, unit std; no ImageNet mean/std.
    if train:
        return T.Compose([T.RandomResizedCrop(224,scale=(0.6,1.0)),T.RandomHorizontalFlip(),
                          T.RandomVerticalFlip(),T.RandomRotation(20),
                          T.ColorJitter(0.3,0.3,0.3,0.05),T.ToTensor()])
    return T.Compose([T.Resize(224),T.CenterCrop(224),T.ToTensor()])


def probabilities(net, x):
    out = net(x)
    return out[0] if isinstance(out,tuple) else out


def metrics(truth, pred, k=12):
    cm = np.zeros((k,k),dtype=np.int64)
    np.add.at(cm,(truth,pred),1)
    tp=cm.diagonal(); support=cm.sum(1); predicted=cm.sum(0)
    precision=np.divide(tp,predicted,out=np.zeros(k,float),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros(k,float),where=support>0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(k,float),where=(support+predicted)>0)
    return {'accuracy':float(tp.sum()/cm.sum()),'macro_f1':float(f1.mean()),
            'macro_precision':float(precision.mean()),'macro_recall':float(recall.mean()),
            'confusion_matrix':cm.tolist(),'per_class_f1':f1.tolist(),'support':support.tolist()}


@torch.inference_mode()
def infer(net,loader,device):
    net.eval(); probs=[]; truth=[]
    for x,y in loader:
        p=probabilities(net,x.to(device))
        assert p.ndim==2 and p.shape[1]==12 and torch.isfinite(p).all()
        assert torch.allclose(p.sum(1),torch.ones(len(p),device=device),atol=1e-4)
        probs.append(p.cpu().numpy()); truth.extend(y.tolist())
    return np.concatenate(probs),np.asarray(truth)


def export_predictions(path,rows,p,head_names,team_labels):
    assert len(rows)==len(p)
    # Probability columns, y_true and y_pred all use the team's order.
    order=[head_names.index(label) for label in team_labels]
    q=p[:,order]; pred=q.argmax(1)
    with Path(path).open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['image_id','y_true','y_pred']+[f'p_{i}' for i in range(12)])
        for r,y,prob in zip(rows,pred,q): w.writerow([r['image_id'],r['y'],int(y),*map(float,prob)])
    return metrics(np.asarray([int(r['y']) for r in rows]),pred)


def write_json(path,obj): Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['smoke','train','evaluate'])
    ap.add_argument('--manifest',required=True);ap.add_argument('--mapping',required=True)
    ap.add_argument('--pv-root');ap.add_argument('--pd-root')
    ap.add_argument('--weights',default='yolo11s-cls.pt');ap.add_argument('--checkpoint')
    ap.add_argument('--out',required=True);ap.add_argument('--epochs',type=int,default=None)
    ap.add_argument('--warmup',type=int,default=None)
    ap.add_argument('--batch',type=int,default=8);ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--threads',type=int,default=4)
    a=ap.parse_args();torch.set_num_threads(a.threads)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(a.seed)
    torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True,warn_only=True)
    rows,labels=load_manifest(a.manifest,a.mapping);names=sorted(labels)
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    import ultralytics,torchvision
    config=vars(a)|{'device':str(device),'torch':torch.__version__,'torchvision':torchvision.__version__,
         'ultralytics':ultralytics.__version__,'python':platform.python_version(),
         'manifest_sha256':digest(a.manifest),'mapping_sha256':digest(a.mapping),
         'team_labels':labels,'head_names':names,'eval_transform':'RGB -> Resize(short edge 224) -> CenterCrop(224) -> ToTensor; mean=0,std=1',
         'selection':'PV validation macro-F1 only','gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None}
    write_json(out/'config.json',config)
    wrapper=YOLO(a.weights);net=wrapper.model
    if a.mode=='evaluate':
        assert a.checkpoint and a.pv_root and a.pd_root
        state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
        assert state['manifest_sha256']==config['manifest_sha256']
        assert state['mapping_sha256']==config['mapping_sha256']
        assert state['head_names']==names
        assert state['mode']=='train', 'Smoke checkpoint forbidden for final external evaluation'
        ClassificationModel.reshape_outputs(net,12);net.load_state_dict(state['state_dict']);net.to(device)
        results={'checkpoint_sha256':digest(a.checkpoint),'selected_using':'PV val only'}
        for split,root,tag in [('test',a.pv_root,'internal'),('external',a.pd_root,'external')]:
            selected=subset(rows,split,0);ds=Leaves(selected,root,names,transform())
            p,y=infer(net,DataLoader(ds,batch_size=a.batch,shuffle=False,num_workers=0),device)
            results[tag]=export_predictions(out/f'predictions_{tag}.csv',selected,p,names,labels)
        write_json(out/'evaluation.json',results);print(json.dumps(results,indent=2));return
    assert a.pv_root, '--pv-root is required'
    assert not a.pd_root, 'Do not supply PlantDoc during smoke/train'
    train_rows=subset(rows,'train',4 if a.mode=='smoke' else 0)
    val_rows=subset(rows,'val',2 if a.mode=='smoke' else 0)
    tr=Leaves(train_rows,a.pv_root,names,transform(True));va=Leaves(val_rows,a.pv_root,names,transform())
    write_json(out/'selected_rows.json',{'train':[r['image_id'] for r in train_rows],'val':[r['image_id'] for r in val_rows]})
    net.to(device).eval();x=torch.stack([va[i][0] for i in range(min(4,len(va)))]).to(device)
    with torch.inference_mode():
        pre=probabilities(net,x);assert pre.shape==(len(x),1000)
    # Check actual pretrained inference API on real PV images, including RGB/BGR handling.
    pred=wrapper.predict(source=[str(p) for p in va.paths[:len(x)]],imgsz=224,device=str(device),verbose=False)
    assert all(tuple(r.probs.data.shape)==(1000,) for r in pred)
    wrapper.model.to(device).eval()
    with torch.inference_mode():
        tensor_probs=probabilities(net,x)
    api_probs=torch.stack([r.probs.data for r in pred]).to(device)
    assert torch.allclose(tensor_probs,api_probs,atol=2e-5,rtol=2e-4),'Preprocessing/API discrepancy'
    # A fresh model avoids predictor fusion side effects during training.
    net=YOLO(a.weights).model
    ClassificationModel.reshape_outputs(net,12)
    net.model[-1].drop.p=0.3;net.names=dict(enumerate(names));net.to(device)
    # Official checkpoints load with gradients disabled; explicitly enable fine-tuning.
    net.requires_grad_(True)
    initial_backbone=next(net.parameters()).detach().clone()
    warm=[];net.eval()
    with torch.inference_mode():
        probabilities(net,x)
        for _ in range(5):
            if device.type=='cuda':torch.cuda.synchronize()
            t=time.perf_counter();p=probabilities(net,x)
            if device.type=='cuda':torch.cuda.synchronize()
            warm.append(time.perf_counter()-t)
    smoke={'pretrained_weights_sha256':digest(a.weights),'input_shape':list(x.shape),'input_range':[float(x.min()),float(x.max())],
           'pretrained_output_shape':list(pre.shape),'new_head_shape':list(p.shape),
           'head_to_team_index':[labels.index(n) for n in names],
           'forward_batch_seconds':warm,'timing_note':'5 warm measurements, forward only; input tensors already prepared',
           'api_tensor_max_abs_difference':float((tensor_probs-api_probs).abs().max()),
           'train_count':len(tr),'val_count':len(va),'plantdoc_images_opened':0,
           'trainable_parameters':sum(v.numel() for v in net.parameters() if v.requires_grad)}
    write_json(out/'smoke.json',smoke)
    train_dl=DataLoader(tr,batch_size=a.batch,shuffle=True,num_workers=0,generator=torch.Generator().manual_seed(a.seed))
    val_dl=DataLoader(va,batch_size=a.batch,shuffle=False,num_workers=0)
    epochs=a.epochs if a.epochs is not None else (1 if a.mode=='smoke' else 20)
    head_ids={id(p) for p in net.model[-1].parameters()}
    head=[p for p in net.parameters() if id(p) in head_ids]
    body=[p for p in net.parameters() if id(p) not in head_ids]
    opt=torch.optim.AdamW([{'params':head,'lr':1e-3},{'params':body,'lr':1e-4}],weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(epochs,1))
    criterion=torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    best=-1.;stale=0;history=[];start=time.perf_counter()
    warmup=a.warmup if a.warmup is not None else (0 if a.mode=='smoke' else 2)
    assert epochs>0 and warmup>=0
    for epoch in range(warmup+epochs):
        epoch_start=time.perf_counter()
        warming=epoch<warmup
        net.requires_grad_(not warming)
        for v in net.model[-1].parameters():v.requires_grad_(True)
        if warming:
            net.eval();net.model[-1].train()
        else:
            net.train()
        total=0.;count=0
        for bx,by in train_dl:
            bx,by=bx.to(device),by.to(device);opt.zero_grad(set_to_none=True)
            logits=net(bx);assert logits.shape==(len(bx),12)
            loss=criterion(logits,by);assert torch.isfinite(loss)
            loss.backward()
            if not warming: assert next(net.parameters()).grad is not None, 'Backbone gradients missing'
            assert torch.stack([torch.isfinite(v.grad).all() for v in net.parameters() if v.grad is not None]).all()
            opt.step();total+=float(loss.detach())*len(by);count+=len(by)
        if not warming:scheduler.step()
        vp,vy=infer(net,val_dl,device);scores=metrics(vy,vp.argmax(1))
        history.append({'epoch':epoch+1,'stage':'warmup' if warming else 'finetune','train_loss':total/count,'val_macro_f1':scores['macro_f1'],'val_accuracy':scores['accuracy'],'epoch_seconds':time.perf_counter()-epoch_start})
        print(history[-1],flush=True);write_json(out/'history.json',history)
        if scores['macro_f1']>best:
            best=scores['macro_f1'];stale=0
            torch.save({'state_dict':{k:v.detach().cpu() for k,v in net.state_dict().items()},
                        'head_names':names,'team_labels':labels,'epoch':epoch+1,'mode':a.mode,
                        'manifest_sha256':config['manifest_sha256'],'mapping_sha256':config['mapping_sha256'],
                        'val_macro_f1':best},out/'best_model.pt')
        else: stale+=int(not warming)
        if not warming and stale>=5:break
    if history: assert not torch.equal(initial_backbone,next(net.parameters()).detach()), 'Backbone not updated'
    smoke.update(backbone_updated=bool(history),training_seconds=time.perf_counter()-start,epochs_completed=len(history),best_pv_val_macro_f1=best,
                 limitation='Small-subset smoke metrics are pipeline diagnostics, not final performance' if a.mode=='smoke' else 'PV val selected model; no external evaluation yet')
    write_json(out/'smoke.json',smoke)
    if history:
        state=torch.load(out/'best_model.pt',map_location=device,weights_only=True);net.load_state_dict(state['state_dict'])
        vp,vy=infer(net,val_dl,device)
        export_predictions(out/'predictions_validation.csv',val_rows,vp,names,labels)
    print(json.dumps(smoke,indent=2))

if __name__=='__main__': main()
