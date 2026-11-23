import json
import os
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from src.predict import load_selected


def test_model_cache_reloads_replaced_artifact(tmp_path):
    (tmp_path/'selected.json').write_text(json.dumps({'name':'test','file':'model.joblib','feature_version':'v1'}))
    path=tmp_path/'model.joblib'
    first=DummyClassifier(strategy='most_frequent').fit(np.zeros((2,46)),['happy','happy'])
    first.n_features_in_=46
    joblib.dump(first,path)
    loaded,_=load_selected(tmp_path)
    assert load_selected(tmp_path)[0] is loaded
    old_ns=path.stat().st_mtime_ns
    second=DummyClassifier(strategy='most_frequent').fit(np.zeros((2,46)),['sad','sad'])
    second.n_features_in_=46
    joblib.dump(second,path);os.utime(path,ns=(old_ns+1000000,old_ns+1000000))
    replaced,_=load_selected(tmp_path)
    assert replaced is not loaded
    assert list(replaced.classes_)==['sad']
