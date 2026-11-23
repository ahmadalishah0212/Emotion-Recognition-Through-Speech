"""Render recorded metrics without rerunning or selecting models."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

root=Path(__file__).resolve().parents[1]/'results/accuracy'
data=json.loads((root/'prosody_historical.json').read_text())['historical_comparisons']
fig,axes=plt.subplots(1,2,figsize=(12,5))
for ax,key,title in zip(axes,['baseline','candidate'],['130-feature LR','138-feature prosody LR']):
    result=data['ravdess'][key];matrix=np.array(result['confusion_matrix'])
    ax.imshow(matrix,cmap='Blues',vmin=0,vmax=30)
    for (i,j),v in np.ndenumerate(matrix):ax.text(j,i,str(v),ha='center',va='center',fontsize=8,color='white' if v>18 else 'black')
    ax.set(xticks=range(8),yticks=range(8),xticklabels=result['confusion_labels'],yticklabels=result['confusion_labels'],xlabel='Predicted',ylabel='Acted label',title=title)
    plt.setp(ax.get_xticklabels(),rotation=45,ha='right')
fig.suptitle('Historical RAVDESS comparison: same 360 recordings, six actors')
fig.tight_layout();fig.savefig(root/'prosody_confusion.png',dpi=130);plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4));positions=np.arange(3)
for offset,key,title in [(-.18,'baseline','130 features'),(.18,'candidate','138 features')]:
    ax.bar(positions+offset,[v[key]['accuracy']*100 for v in data.values()],width=.36,label=title)
ax.set(xticks=positions,xticklabels=['RAVDESS','TESS','EmoDB'],ylabel='Accuracy (%)',ylim=(0,65),title='Historical comparisons; previously inspected test corpora')
ax.legend();fig.tight_layout();fig.savefig(root/'historical_accuracy.png',dpi=130)
