"""Evaluate a frozen fusion rule on two additional speech-held-out AV actors."""
import json
from pathlib import Path
import sys
import time
import zipfile
import argparse
from hashlib import md5
import numpy as np
import joblib
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.experiment_accuracy import ROOT
from scripts.experiment_fusion import evaluate
from scripts.evaluate_ravdess_av import EMOTIONS
from src.face_opencv import load_model,aggregate_frames
from src.face import load_face_model,aggregate_frames as baseline_face
from src.predict import predict_audio
from src.video import analyze_video_payload
from src.fusion import SHARED
from src.fusion_experiments import fuse_scores
from src.prosody import extract_features_v3

def summarize(rows,key):
    valid=[r for r in rows if r['truth'] in SHARED and r[key] is not None]
    y=[r['truth'] for r in valid];p=[r[key] for r in valid]
    report=classification_report(y,p,labels=list(SHARED),output_dict=True,zero_division=0)
    return dict(samples=len(valid),accuracy=float(accuracy_score(y,p)),macro_f1=report['macro avg']['f1-score'],weighted_f1=report['weighted avg']['f1-score'],per_class=report,
                confusion_labels=list(SHARED)+['calm','contempt'],confusion_matrix=confusion_matrix(y,p,labels=list(SHARED)+['calm','contempt']).tolist())

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-for-download',type=int,default=0,help='Maximum seconds to await the checksum-verified Actor15 archive')
    args=parser.parse_args()
    frozen=json.loads((ROOT/'results/accuracy/fusion_frozen.json').read_text())
    config=frozen['selected'];rows=[];start=time.perf_counter()
    face_model=load_model();old=load_face_model()
    prosody=joblib.load(ROOT/'data/cache/prosody_candidate.joblib')
    for actor in frozen['evaluation_actors']:
        archive=ROOT/f'data/cache/Video_Speech_Actor_{actor}.zip'
        deadline=time.monotonic()+args.wait_for_download
        expected={'10':(565580163,'70199d1c6902f76e17df308d5c41fa01'),'15':(525205576,'4b3b9bc8473e86f884630f2506684887')}[actor]
        while not archive.exists() or archive.stat().st_size!=expected[0]:
            if time.monotonic()>deadline:raise ValueError('Complete official actor archive required')
            print('Waiting for actor',actor,'download',flush=True);time.sleep(10)
        if md5(archive.read_bytes()).hexdigest()!=expected[1]:raise ValueError('Archive checksum mismatch')
        with zipfile.ZipFile(archive) as package:
            names=sorted(n for n in package.namelist() if Path(n).name.startswith('01-01-') and n.endswith('.mp4'))
            assert len(names)==60
            for index,name in enumerate(names,1):
                clip=analyze_video_payload(package.read(name))
                voice=predict_audio(clip['audio']) if clip['audio'] else None
                face=aggregate_frames(clip['frames'],face_model);base=baseline_face(clip['frames'],old)
                row=dict(actor=actor,file=Path(name).name,truth=EMOTIONS[Path(name).stem.split('-')[2]],
                         voice=voice['emotion'] if voice else None,face=face['emotion'] if face else None,ferplus=base['emotion'] if base else None,
                         voice_scores=voice['probabilities'] if voice else None,face_full_scores=face['probabilities'] if face else None,
                         frames_sampled=clip['sampled'],frames_with_face=face['frames_with_face'] if face else 0)
                for key,cfg in [('equal',dict(weight=.5,temperature=1.)),('frozen',config)]:
                    scores=fuse_scores(row['voice_scores'],row['face_full_scores'],**cfg) if voice and face else None
                    row[key]=max(scores,key=scores.get) if scores else None
                p=prosody.predict_proba(extract_features_v3(clip['audio']).reshape(1,-1))[0] if voice else None
                row['voice_v3_scores']=dict(zip(prosody.classes_,map(float,p))) if p is not None else None
                row['voice_v3']=max(row['voice_v3_scores'],key=row['voice_v3_scores'].get) if p is not None else None
                for key,cfg in [('equal_v3',dict(weight=.5,temperature=1.)),('frozen_v3',config)]:
                    scores=fuse_scores(row['voice_v3_scores'],row['face_full_scores'],**cfg) if voice and face else None
                    row[key]=max(scores,key=scores.get) if scores else None
                rows.append(row)
                if index%10==0: print(actor,index,flush=True)
    keys=['voice','voice_v3','face','ferplus','equal','frozen','equal_v3','frozen_v3']
    report=dict(protocol='Frozen Actor05-developed fusion, evaluated once on new AV Actors10/15. Speech test actors previously measured; face/AV clips newly evaluated. No tuning on these outputs.',configuration=config,seconds=time.perf_counter()-start,
                pooled={k:summarize(rows,k) for k in keys},by_actor={a:{k:summarize([r for r in rows if r['actor']==a],k) for k in keys} for a in frozen['evaluation_actors']},rows=rows)
    (ROOT/'results/accuracy/new_av.json').write_text(json.dumps(report,indent=2))
    print({k:(v['accuracy'],v['macro_f1']) for k,v in report['pooled'].items()},flush=True)

if __name__=='__main__': main()
