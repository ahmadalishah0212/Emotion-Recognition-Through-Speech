"""Ablate added prosody and training-only mild noise using fixed actor CV."""
import sys
from pathlib import Path
import json
import time
import numpy as np
import soundfile as sf
from io import BytesIO
from sklearn.model_selection import GroupKFold

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.experiment_accuracy import ROOT,LABELS,build,metrics
from src.dataset_sources import scan_ravdess,scan_tess,scan_emodb,validate_rows
from src.feature_extraction import load_audio,extract_features_v2
from src.prosody import prosody_summary
from src.train import TEST_ACTORS

def main():
    rows,errors=validate_rows(scan_ravdess(ROOT/'data/raw')[0]+scan_tess(ROOT/'data/external/tess')[0]+scan_emodb(ROOT/'data/external/emodb')[0])
    if errors: raise ValueError(errors[:2])
    cache=np.load(ROOT/'data/cache/expanded_features_v2.npz')
    assert list(cache['hashes'])==[r['sha256'] for r in rows]
    x=cache['features']; y=np.array([r['emotion'] for r in rows]); g=np.array([r['speaker'] for r in rows])
    train=np.array([i for i,r in enumerate(rows) if r['dataset']=='ravdess' and r['speaker'].split(':')[-1] not in TEST_ACTORS])
    rng=np.random.default_rng(42)
    tess=np.concatenate([rng.choice([i for i,r in enumerate(rows) if r['speaker']=='tess:OAF' and y[i]==label],70,replace=False) for label in LABELS if label!='calm'])
    needed=np.concatenate([train,tess]); extra=np.zeros((len(rows),8),dtype=np.float32); noise={}
    extra_cache=ROOT/'data/cache/expanded_features_v3.npz'
    cached_extra=extra_cache.exists()
    if cached_extra:
        saved=np.load(extra_cache);assert list(saved['hashes'])==list(cache['hashes'])
        extra=saved['features'][:,130:]
    chosen=set(rng.choice(train,size=270,replace=False).tolist())
    for n,i in enumerate(needed,1):
        if cached_extra and i not in chosen:continue
        signal,sr=load_audio(rows[i]['path'])
        if not cached_extra:extra[i]=prosody_summary(signal,sr)
        if i in chosen:
            # 25 dB additive Gaussian noise; retain source actor and original label.
            altered=signal+rng.normal(0,np.sqrt(np.mean(signal**2))/10**(25/20),len(signal))
            buff=BytesIO();sf.write(buff,altered,sr,format='WAV',subtype='FLOAT')
            noise[i]=extract_features_v2(buff.getvalue())
        if n%250==0: print('training features',n,len(needed),flush=True)
    folds=list(GroupKFold(3).split(train,groups=g[train])); results=[]
    for name in ['baseline_lr','prosody_lr','prosody_svm','noise_lr']:
        data=np.concatenate([x,extra],axis=1) if name.startswith('prosody') else x
        pred=np.empty(len(train),dtype=object); fold=[];start=time.perf_counter()
        for tr,va in folds:
            fit=np.concatenate([train[tr],tess]); val=train[va]
            xx=data[fit]; yy=y[fit]
            if name=='noise_lr':
                aug=[i for i in fit if i in noise]
                xx=np.concatenate([xx,np.stack([noise[i] for i in aug])]); yy=np.concatenate([yy,y[aug]])
            model=build('svm_10' if name=='prosody_svm' else 'lr_1');model.fit(xx,yy)
            pred[va]=model.predict(data[val]);fold.append(dict(train_samples=len(yy),validation_samples=len(val),metrics=metrics(y[val],pred[va])))
        results.append(dict(id=name,seed=42,features=data.shape[1],folds=fold,pooled_validation=metrics(y[train],pred),seconds=time.perf_counter()-start))
        print(name,results[-1]['pooled_validation']['macro_f1'],flush=True)
    (ROOT/'results/accuracy/feature_augmentation.json').write_text(json.dumps(dict(protocol='Same training-only actor folds; prosody and augmentation never touch historical test data',augmentation='270 original RAVDESS train clips, 25 dB Gaussian noise; copies only in source fit folds',extra_features=['f0_median','f0_std','f0_p90_p10','duration','rms_median','rms_p90_p10','rms_delta_std','active_frame_fraction'],experiments=results),indent=2))

if __name__=='__main__':main()
