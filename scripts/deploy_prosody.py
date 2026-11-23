"""Install the recorded prosody candidate while retaining all previous artifacts."""
from pathlib import Path
import sys
import json
import shutil
import argparse
from hashlib import sha256
import joblib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.prosody import extract_features_v3
ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollback-v2',action='store_true',help='Restore the preserved 130-feature voice selection')
    args=parser.parse_args()
    if args.rollback_v2:
        selection=dict(name='logistic_expanded_v2',file='logistic_expanded_v2.joblib',feature_version='v2',feature_count=130,
                       selection_metric='training_actor_cv_macro_f1',evaluation='results/expanded/evaluation.json')
        assert (ROOT/'models'/selection['file']).exists()
        (ROOT/'models/selected.json').write_text(json.dumps(selection,indent=2)+'\n')
        print('Restored v2 selection');return
    cv=json.loads((ROOT/'results/accuracy/feature_augmentation.json').read_text())['experiments']
    base=next(e for e in cv if e['id']=='baseline_lr');winner=next(e for e in cv if e['id']=='prosody_lr')
    assert all(w['metrics']['macro_f1']>b['metrics']['macro_f1'] for w,b in zip(winner['folds'],base['folds']))
    report=json.loads((ROOT/'results/accuracy/prosody_historical.json').read_text())
    assert all(r['candidate']['accuracy']>r['baseline']['accuracy'] for r in report['historical_comparisons'].values())
    artifact=ROOT/'data/cache/prosody_candidate.joblib';model=joblib.load(artifact)
    assert model.n_features_in_==138
    assert model.get_params()['logisticregression__C']==1
    saved=np.load(ROOT/'data/cache/expanded_features_v3.npz')
    lookup={value:i for i,value in enumerate(saved['hashes'])}
    for sample in sorted((ROOT/'data/raw').rglob('*.wav')):
        digest=sha256(sample.read_bytes()).hexdigest()
        if digest in lookup:
            np.testing.assert_allclose(extract_features_v3(sample),saved['features'][lookup[digest]],rtol=1e-5,atol=1e-4)
            break
    else:raise ValueError('A validated raw sample is required to verify feature parity')
    # Keep the prior v2 artifact available for an explicit rollback.
    selection=dict(name='logistic_prosody_v3',file='logistic_prosody_v3.joblib',feature_version='v3',feature_count=138,
                   selection_metric='training_actor_cv_macro_f1',evaluation='results/accuracy/prosody_historical.json',
                   fallback='logistic_expanded_v2.joblib',artifact_sha256=sha256(artifact.read_bytes()).hexdigest())
    shutil.copyfile(artifact,ROOT/'models'/selection['file'])
    (ROOT/'models/selected.json').write_text(json.dumps(selection,indent=2)+'\n')
    print(selection)

if __name__=='__main__':main()
