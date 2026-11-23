"""Check evaluator thread oversubscription with both face sessions resident."""
from pathlib import Path
import sys,time,json
import cv2
import onnxruntime as ort
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.face import MODEL_PATH as OLD,score_frame as old_score
from src.face_opencv import MODEL_PATH as NEW,score_frame as new_score
root=Path(__file__).resolve().parents[1];frame=cv2.imread(str(root/'demo/assets/happy.jpg'))
ort.set_default_logger_severity(3);results=[]
for n in [1,0]:
    options=ort.SessionOptions();options.intra_op_num_threads=n;options.inter_op_num_threads=1
    a=ort.InferenceSession(str(OLD),sess_options=options,providers=['CPUExecutionProvider'])
    b=ort.InferenceSession(str(NEW),sess_options=options,providers=['CPUExecutionProvider'])
    start=time.perf_counter();p=old_score(frame,a);q=new_score(frame,b)
    results.append(dict(threads=n,two_frames_seconds=time.perf_counter()-start,old=p['probabilities'],new=q['probabilities']))
    print(n,results[-1]['two_frames_seconds'],flush=True)
(root/'results/accuracy/dual_face_runtime.json').write_text(json.dumps(results,indent=2))
