"""Measure CPU feature/inference latency on four fixed local demo samples."""
import json
from pathlib import Path
import sys
import time
import joblib

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.feature_extraction import extract_features_v2
from src.prosody import extract_features_v3
root=Path(__file__).resolve().parents[1]
results=[]
for name,path,extractor in [('v2',root/'models/logistic_expanded_v2.joblib',extract_features_v2),('v3',root/'data/cache/prosody_candidate.joblib',extract_features_v3)]:
    start=time.perf_counter();model=joblib.load(path);load=time.perf_counter()-start
    timings=[]
    for label in ['happy','sad','angry','neutral']:
        payload=(root/f'demo/assets/{label}.wav').read_bytes();start=time.perf_counter()
        p=model.predict_proba(extractor(payload).reshape(1,-1))
        timings.append(dict(sample=label,seconds=time.perf_counter()-start,top_class=str(model.classes_[p.argmax()])))
    results.append(dict(engine=name,model_bytes=path.stat().st_size,load_seconds=load,inference=timings))
(root/'results/accuracy/audio_runtime.json').write_text(json.dumps(results,indent=2))
print(results)
