"""Freeze fusion choice on Actor 05 before opening Actors 10 and 15 results."""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.fusion_experiments import fuse_scores
from scripts.experiment_accuracy import metrics

ROOT=Path(__file__).resolve().parents[1]

def evaluate(rows,config):
    valid=[r for r in rows if r['truth']!='calm' and r.get('voice_scores') and r.get('face_full_scores')]
    scores=[fuse_scores(r['voice_scores'],r['face_full_scores'],**config) for r in valid]
    truth=[r['truth'] for r in valid];pred=[max(p,key=p.get) for p in scores]
    confidence=np.array([max(p.values()) for p in scores]);correct=np.array([a==b for a,b in zip(pred,truth)])
    ece=0.
    for low in np.linspace(0,.9,10):
        mask=(confidence>low)&(confidence<=low+.1)
        if mask.any(): ece+=float(mask.mean()*abs(confidence[mask].mean()-correct[mask].mean()))
    # Eight-class helper includes Calm; seven-class reports use only the shared classes.
    from sklearn.metrics import classification_report, accuracy_score,confusion_matrix
    from src.fusion import SHARED
    report=classification_report(truth,pred,labels=list(SHARED),output_dict=True,zero_division=0)
    return dict(samples=len(valid),accuracy=float(accuracy_score(truth,pred)),macro_f1=report['macro avg']['f1-score'],weighted_f1=report['weighted avg']['f1-score'],per_class=report,confusion_labels=SHARED,confusion_matrix=confusion_matrix(truth,pred,labels=list(SHARED)).tolist(),ece_10bins=ece,nll=float(np.mean([-np.log(max(p[t],1e-12)) for p,t in zip(scores,truth)])))

def main():
    out=ROOT/'results/accuracy';out.mkdir(parents=True,exist_ok=True)
    rows=json.loads((ROOT/'results/face/ravdess_actor05_opencv_zoo.json').read_text())['rows']
    configs=[dict(weight=w,temperature=t,confidence_aware=False) for t in [1.,2.] for w in [.5,.25,.75]]
    configs.append(dict(weight=.5,temperature=1.,confidence_aware=True))
    experiments=[dict(config=c,validation=evaluate(rows,c)) for c in configs]
    winner=max(experiments,key=lambda r:r['validation']['macro_f1'])
    (out/'fusion_frozen.json').write_text(json.dumps(dict(development_actor='05',evaluation_actors=['10','15'],selection='development macro F1, ties retain first simpler configuration',experiments=experiments,selected=winner['config'],warning='Actor 05 is previously inspected development data. Do not retune after new actor results.'),indent=2))
    print(winner,flush=True)

if __name__=='__main__':main()
