"""Training-actor CV search; historical tests evaluated only after selection."""
from pathlib import Path
import json
import sys
import time
from collections import Counter
import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.dataset_sources import scan_ravdess, scan_tess, scan_emodb, validate_rows
from src.train import TEST_ACTORS

ROOT=Path(__file__).resolve().parents[1]
LABELS=['angry','calm','disgust','fearful','happy','neutral','sad','surprised']

def metrics(y,p,source_labels=None):
    report=classification_report(y,p,labels=source_labels or LABELS,output_dict=True,zero_division=0)
    return dict(accuracy=float(accuracy_score(y,p)),macro_f1=report['macro avg']['f1-score'],
                weighted_f1=report['weighted avg']['f1-score'],per_class=report,
                confusion_labels=LABELS,confusion_matrix=confusion_matrix(y,p,labels=LABELS).tolist())

def build(name):
    if name.startswith('lr_'):
        return make_pipeline(StandardScaler(),LogisticRegression(C=float(name[3:]),max_iter=2500,class_weight='balanced',random_state=42))
    if name=='select80_lr':
        return make_pipeline(StandardScaler(),SelectKBest(f_classif,k=80),LogisticRegression(max_iter=2500,class_weight='balanced',random_state=42))
    if name=='pca_svm':
        return make_pipeline(StandardScaler(),PCA(n_components=.95),SVC(C=10,kernel='rbf'))
    if name.startswith('svm_'):
        return make_pipeline(StandardScaler(),SVC(C=float(name[4:]),kernel='rbf'))
    if name=='extra_trees':
        return ExtraTreesClassifier(n_estimators=250,min_samples_leaf=2,class_weight='balanced',n_jobs=2,random_state=42)
    if name=='hist_gradient':
        return HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=1,random_state=42)
    raise ValueError(name)

def main():
    sources=[scan_ravdess(ROOT/'data/raw')[0],scan_tess(ROOT/'data/external/tess')[0],scan_emodb(ROOT/'data/external/emodb')[0]]
    rows,errors=validate_rows(sum(sources,[]))
    if errors: raise ValueError(errors[:2])
    cache=np.load(ROOT/'data/cache/expanded_features_v2.npz')
    if list(cache['hashes']) != [r['sha256'] for r in rows]: raise ValueError('Feature provenance mismatch')
    x=cache['features']; y=np.array([r['emotion'] for r in rows]); groups=np.array([r['speaker'] for r in rows])
    train=np.array([i for i,r in enumerate(rows) if r['dataset']=='ravdess' and r['speaker'].split(':')[-1] not in TEST_ACTORS])
    rng=np.random.default_rng(42)
    tess=np.concatenate([rng.choice([i for i,r in enumerate(rows) if r['speaker']=='tess:OAF' and y[i]==label],70,replace=False) for label in LABELS if label!='calm'])
    folds=list(GroupKFold(3).split(train,groups=groups[train]))
    experiments=[]
    for name in ['lr_1','lr_0.1','lr_10','svm_1','svm_10','svm_30','select80_lr','pca_svm','extra_trees','hist_gradient']:
        predictions=np.empty(len(train),dtype=object); details=[]; start=time.perf_counter()
        for tr,va in folds:
            fit=np.concatenate([train[tr],tess]); val=train[va]
            assert not set(groups[fit]) & set(groups[val])
            model=build(name); model.fit(x[fit],y[fit]); predictions[va]=model.predict(x[val])
            details.append(dict(train_samples=len(fit),validation_samples=len(val),validation_speakers=sorted(set(groups[val])),metrics=metrics(y[val],predictions[va])))
        result=dict(id=name,features='v2_130',seed=42,parameters={k:str(v) for k,v in build(name).get_params().items()},
                    folds=details,pooled_validation=metrics(y[train],predictions),seconds=time.perf_counter()-start)
        experiments.append(result)
        print(name,result['pooled_validation']['accuracy'],result['pooled_validation']['macro_f1'],flush=True)
        out=ROOT/'results/accuracy';out.mkdir(parents=True,exist_ok=True)
        (out/'search.json').write_text(json.dumps(dict(protocol='Training-only three-fold actor CV; TESS OAF fit only; pooled OOF macro F1 selection',experiments=experiments),indent=2))
    winner=max(experiments,key=lambda r:r['pooled_validation']['macro_f1'])['id']
    final=build(winner);final.fit(x[np.concatenate([train,tess])],y[np.concatenate([train,tess])])
    joblib.dump(final,ROOT/'data/cache/accuracy_candidate.joblib')
    historical={}
    for corpus in ('ravdess','tess','emodb'):
        ix=np.array([i for i,r in enumerate(rows) if (corpus=='ravdess' and r['dataset']==corpus and r['speaker'].split(':')[-1] in TEST_ACTORS) or (corpus=='tess' and r['speaker']=='tess:YAF') or (corpus=='emodb' and r['dataset']==corpus)])
        historical[corpus]=dict(samples=len(ix),speakers=sorted(set(groups[ix])),metrics=metrics(y[ix],final.predict(x[ix])))
    (out/'selected.json').write_text(json.dumps(dict(winner=winner,train_samples=len(train)+len(tess),historical_comparisons=historical,warning='Historical comparisons are previously inspected development benchmarks, not fresh final tests.'),indent=2))
    print('CV winner',winner,flush=True)

if __name__=='__main__': main()
