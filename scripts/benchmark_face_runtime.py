"""Compare ORT CPU thread counts on identical local frames."""
import sys
from pathlib import Path
import json
import time
import cv2
import onnxruntime as ort
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.face_opencv import MODEL_PATH,score_frame
root=Path(__file__).resolve().parents[1]
frame=cv2.imread(str(root/'demo/assets/happy.jpg'))
ort.set_default_logger_severity(3)
results=[]
for threads in [0,2]:
    options=ort.SessionOptions();options.intra_op_num_threads=threads;options.inter_op_num_threads=1
    start=time.perf_counter()
    session=ort.InferenceSession(str(MODEL_PATH),sess_options=options,providers=['CPUExecutionProvider'])
    load=time.perf_counter()-start
    start=time.perf_counter();scores=[score_frame(frame,session) for _ in range(12)]
    results.append(dict(threads=threads,load_seconds=load,twelve_frame_seconds=time.perf_counter()-start,scores=scores[0]['probabilities']))
    print({k:v for k,v in results[-1].items() if k!='scores'},flush=True)
assert np.allclose(list(results[0]['scores'].values()),list(results[1]['scores'].values()),rtol=1e-4,atol=1e-6)
(root/'results/accuracy/face_runtime.json').write_text(json.dumps(results,indent=2))
