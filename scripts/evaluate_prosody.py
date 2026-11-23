"""Fit the CV-selected prosody candidate and compare historical test corpora."""
import sys
from pathlib import Path
import json
import joblib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.experiment_accuracy import ROOT,LABELS,build,metrics
from src.dataset_sources import scan_ravdess,scan_tess,scan_emodb,validate_rows
from src.feature_extraction import load_audio
from src.prosody import prosody_summary
from src.train import TEST_ACTORS

def main():
    rows,errors=validate_rows(scan_ravdess(ROOT/'data/raw')[0]+scan_tess(ROOT/'data/external/tess')[0]+scan_emodb(ROOT/'data/external/emodb')[0])
    assert not errors
    old=np.load(ROOT/'data/cache/expanded_features_v2.npz');hashes=[r['sha256'] for r in rows]
    assert list(old['hashes'])==hashes
    cache=ROOT/'data/cache/expanded_features_v3.npz'
    if cache.exists():
        saved=np.load(cache);assert list(saved['hashes'])==hashes;x=saved['features']
    else:
        extras=[]
        for n,row in enumerate(rows,1):
            extras.append(prosody_summary(*load_audio(row['path'])))
            if n%500==0:print('prosody',n,len(rows),flush=True)
        x=np.concatenate([old['features'],np.stack(extras)],axis=1)
        np.savez_compressed(cache,hashes=hashes,features=x)
    y=np.array([r['emotion'] for r in rows]);groups=np.array([r['speaker'] for r in rows])
    train=np.array([i for i,r in enumerate(rows) if r['dataset']=='ravdess' and r['speaker'].split(':')[-1] not in TEST_ACTORS])
    rng=np.random.default_rng(42);tess=np.concatenate([rng.choice([i for i,r in enumerate(rows) if r['speaker']=='tess:OAF' and y[i]==label],70,replace=False) for label in LABELS if label!='calm'])
    fit=np.concatenate([train,tess]);model=build('lr_1');model.fit(x[fit],y[fit])
    baseline=joblib.load(ROOT/'models/logistic_expanded_v2.joblib');tests={}
    for corpus in ['ravdess','tess','emodb']:
        ix=np.array([i for i,r in enumerate(rows) if (corpus=='ravdess' and r['dataset']==corpus and r['speaker'].split(':')[-1] in TEST_ACTORS) or (corpus=='tess' and r['speaker']=='tess:YAF') or (corpus=='emodb' and r['dataset']==corpus)])
        assert not set(groups[fit])&set(groups[ix])
        source_labels=sorted(set(y[ix]))
        tests[corpus]=dict(samples=len(ix),source_labels=source_labels,baseline=metrics(y[ix],baseline.predict(old['features'][ix]),source_labels),candidate=metrics(y[ix],model.predict(x[ix]),source_labels))
        print(corpus,tests[corpus]['baseline']['macro_f1'],tests[corpus]['candidate']['macro_f1'],flush=True)
    joblib.dump(model,ROOT/'data/cache/prosody_candidate.joblib')
    (ROOT/'results/accuracy/prosody_historical.json').write_text(json.dumps(dict(train_samples=len(fit),features=138,model='balanced Logistic Regression C=1',historical_comparisons=tests,warning='Previously inspected test corpora; no newly untouched speech test claim.'),indent=2))

if __name__=='__main__':main()
