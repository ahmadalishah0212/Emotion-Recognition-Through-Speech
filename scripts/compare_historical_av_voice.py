"""Compare new voice scores on the SAME historical Actor05 face scores."""
import json,sys,zipfile
from pathlib import Path
import joblib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.experiment_accuracy import ROOT
from scripts.experiment_fusion import evaluate
from scripts.evaluate_new_av import summarize
from src.video import analyze_video_payload
from src.prosody import extract_features_v3

def main():
    rows=json.loads((ROOT/'results/face/ravdess_actor05_opencv_zoo.json').read_text())['rows']
    model=joblib.load(ROOT/'data/cache/prosody_candidate.joblib')
    with zipfile.ZipFile(ROOT/'data/cache/Video_Speech_Actor_05.zip') as package:
        lookup={Path(n).name:n for n in package.namelist()}
        for i,row in enumerate(rows,1):
            audio=analyze_video_payload(package.read(lookup[row['file']]))['audio']
            p=model.predict_proba(extract_features_v3(audio).reshape(1,-1))[0]
            row['voice_scores']=dict(zip(model.classes_,map(float,p)))
            row['voice_v3']=max(row['voice_scores'],key=row['voice_scores'].get)
            if i%20==0:print('Actor05 voice',i,flush=True)
    result=dict(protocol='Historical Actor05 development comparison; original saved full-frame face scores reused unchanged.',voice_v3=summarize(rows,'voice_v3'),equal_v3=evaluate(rows,dict(weight=.5,temperature=1.)),softened_v3=evaluate(rows,dict(weight=.5,temperature=2.)))
    (ROOT/'results/accuracy/actor05_prosody.json').write_text(json.dumps(result,indent=2))
    print({k:(v['accuracy'],v['macro_f1']) for k,v in result.items() if isinstance(v,dict)},flush=True)

if __name__=='__main__':main()
