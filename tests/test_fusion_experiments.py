import pytest
from src.fusion import SHARED,experimental_fusion
from src.fusion_experiments import fuse_scores


def test_default_candidate_reproduces_deployed_fusion():
    voice={k:(.7 if k=='sad' else .02) for k in SHARED}|{'calm':.18}
    face={k:(.8 if k=='happy' else .2/6) for k in SHARED}
    actual=experimental_fusion({'probabilities':voice},{'probabilities':face})
    assert fuse_scores(voice,face)==pytest.approx(actual['probabilities'])
    assert sum(fuse_scores(voice,face,temperature=2).values())==pytest.approx(1)


def test_invalid_scores_do_not_create_prediction():
    p=dict.fromkeys(SHARED,0.)
    with pytest.raises(ValueError): fuse_scores(p,p)
    with pytest.raises(ValueError): fuse_scores(p,p,weight=2)
