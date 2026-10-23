"""Audit the fixed external set and produce figures/tables from measured predictions."""
import csv, hashlib, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha256(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def extract_external(archive,manifest,target):
    """Resolve Windows-renamed paths, while requiring exact original image bytes."""
    import re, unicodedata
    rows=list(csv.DictReader(open(manifest)));ext=[r for r in rows if r['split']=='external']
    assert len(ext)==1097 and len({r['image_id'] for r in ext})==1097
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    pv_hashes={r['md5'] for r in rows if r['root']=='pv'}
    def canonical(name):
        name=unicodedata.normalize('NFC',name.replace('\\','/'))
        return '/'.join(re.sub(r'[<>:"|?*]','_',part) for part in name.split('/'))
    renamed=[]
    with zipfile.ZipFile(archive) as z:
        index={}
        for info in z.infolist():
            if not info.is_dir():index.setdefault(canonical(info.filename),[]).append(info.filename)
        first=canonical(ext[0]['relpath'])
        candidates=[name[:-len(first)] for name in index if name==first or name.endswith('/'+first)]
        prefixes=[prefix for prefix in candidates if all(prefix+canonical(r['relpath']) in index for r in ext)]
        if not prefixes:
            prefix=max(candidates,key=lambda p:sum(p+canonical(r['relpath']) in index for r in ext),default='')
            missing=[r['image_id'] for r in ext if prefix+canonical(r['relpath']) not in index]
            raise AssertionError(f'Archive still differs from team manifest: {len(missing)} missing paths. First 10: {missing[:10]}')
        prefix=sorted(prefixes,key=lambda p:(len(p),p))[0]
        for r in ext:
            hits=index[prefix+canonical(r['relpath'])]
            matches=[]
            for name in sorted(hits):
                data=z.read(name)
                if hashlib.md5(data).hexdigest()==r['md5']:matches.append((name,data))
            assert matches,f"External bytes differ from manifest: {r['image_id']}"
            name,data=matches[0]
            assert r['md5'] not in pv_hashes,'Cross-dataset duplicate'
            path=(target/r['relpath']).resolve();assert path.is_relative_to(target.resolve())
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            if name!=prefix+r['relpath']:renamed.append({'image_id':r['image_id'],'zip_member':name})
    report={'external_images':1097,'md5_matches':1097,'cross_dataset_exact_overlaps':0,
            'archive_prefix':prefix,'filename_mappings':len(renamed),'renamed_files':renamed}
    return report


def bootstrap(truth,pred,k=12,n=1000,seed=2026):
    rng=np.random.default_rng(seed);values=[]
    for _ in range(n):
        idx=rng.integers(0,len(truth),len(truth))
        values.append(f1_score(truth[idx],pred[idx],labels=range(k),average='macro',zero_division=0))
    return np.quantile(values,[.025,.975]).tolist()


def summarize(run_root,manifest,mapping):
    run_root=Path(run_root);summary=run_root/'summary';summary.mkdir(exist_ok=True)
    labels=[r['label'] for r in json.load(open(mapping))['classes']]
    manifest=pd.read_csv(manifest);records=[]
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for seed in [42,43,44]:
        hist=pd.read_json(run_root/f'train_s{seed}/history.json')
        axs[0].plot(hist.epoch,hist.train_loss,label=f'Seed {seed}')
        axs[1].plot(hist.epoch,hist.val_macro_f1,label=f'Seed {seed}')
        for tag,split in [('internal','test'),('external','external')]:
            df=pd.read_csv(run_root/f'eval_s{seed}'/f'predictions_{tag}.csv')
            expected=manifest.loc[manifest.split==split,['image_id','y']]
            assert df.image_id.is_unique and set(df.image_id)==set(expected.image_id)
            joined=expected.merge(df,on='image_id',validate='one_to_one')
            assert (joined.y==joined.y_true).all()
            truth=joined.y_true.to_numpy();pred=joined.y_pred.to_numpy()
            assert np.isin(pred,range(12)).all()
            report=classification_report(truth,pred,labels=range(12),target_names=labels,output_dict=True,zero_division=0)
            pd.DataFrame(report).T.to_csv(summary/f'per_class_{tag}_s{seed}.csv')
            macro=report['macro avg'];ci=bootstrap(truth,pred) if tag=='external' else [None,None]
            records.append({'seed':seed,'set':tag,'n':len(df),'accuracy':float((truth==pred).mean()),'macro_f1':macro['f1-score'],
                            'macro_precision':macro['precision'],'macro_recall':macro['recall'],'ci_low':ci[0],'ci_high':ci[1]})
            cm=confusion_matrix(truth,pred,labels=range(12));pd.DataFrame(cm,index=labels,columns=labels).to_csv(summary/f'confusion_{tag}_s{seed}.csv')
            normalized=np.divide(cm,cm.sum(1,keepdims=True),out=np.zeros_like(cm,dtype=float),where=cm.sum(1,keepdims=True)>0)
            f,ax=plt.subplots(figsize=(11,9));im=ax.imshow(normalized,vmin=0,vmax=1,cmap='Blues');f.colorbar(im,ax=ax)
            ax.set(xticks=range(12),yticks=range(12),xticklabels=labels,yticklabels=labels,xlabel='Predicted class',ylabel='True class',title=f'{tag.title()} confusion matrix — seed {seed}')
            plt.setp(ax.get_xticklabels(),rotation=65,ha='right');f.tight_layout();f.savefig(summary/f'confusion_{tag}_s{seed}.png',dpi=150);plt.close(f)
    for ax,title in zip(axs,['Training loss','PlantVillage validation macro-F1']):ax.set(xlabel='Epoch',title=title);ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(summary/'training_curves.png',dpi=150);plt.close(fig)
    results=pd.DataFrame(records);results.to_csv(summary/'results_per_seed.csv',index=False)
    stats=results.groupby('set')[['accuracy','macro_precision','macro_recall','macro_f1']].agg(['mean','std'])
    stats.to_csv(summary/'results_mean_sd.csv')
    pivot=results.pivot(index='seed',columns='set',values='macro_f1');pivot['gap']=pivot.internal-pivot.external
    pivot['retention']=pivot.external/pivot.internal.replace(0,np.nan);pivot.to_csv(summary/'domain_gap.csv')
    means=results.groupby('set').macro_f1.mean()
    text=f'''Across three prespecified training seeds, mean internal macro-F1 was {means['internal']:.4f} and mean external macro-F1 was {means['external']:.4f}. The mean absolute domain gap was {pivot.gap.mean():.4f}. These values describe the fixed shared dataset only; they do not establish field deployment reliability. Check the saved per-class reports and confusion matrices before attributing errors to a specific visual mechanism.\n\nSeed standard deviations describe training variation. Per-run percentile bootstrap intervals use 1,000 resamples of the same external images and describe test-set sampling uncertainty. No three-model superiority claim is made by this notebook.\n'''
    (summary/'measured_results_note.md').write_text(text)
    return summary,results,stats,pivot
