"""Explicit experimental fusion candidates; deployed equal average is separate."""
import numpy as np
from .fusion import SHARED

def fuse_scores(voice,face,weight=.5,temperature=1.0,confidence_aware=False):
    if not 0<=weight<=1 or temperature<=0:
        raise ValueError('Invalid fusion parameters')
    def vector(scores):
        p=np.array([scores[k] for k in SHARED],dtype=float)
        if not np.isfinite(p).all() or (p<0).any() or p.sum()<=0:
            raise ValueError('Invalid modality scores')
        p=p/p.sum()
        p=np.maximum(p,1e-12)**(1/temperature)
        return p/p.sum()
    left,right=vector(voice),vector(face)
    if confidence_aware:
        confidence=lambda p: max(0.01,1+np.sum(p*np.log(p))/np.log(len(SHARED)))
        a,b=confidence(left),confidence(right)
        weight=a/(a+b)
    p=weight*left+(1-weight)*right
    return dict(zip(SHARED,map(float,p)))
