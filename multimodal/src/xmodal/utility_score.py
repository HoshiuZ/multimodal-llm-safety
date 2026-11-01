"""Project normal-capability scoring using official VQA answer normalization."""
from pathlib import Path
import importlib.util

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('frozen_vqa_eval',ROOT/'third_party/vqa/vqaEval.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class _Empty:
    def getQuesIds(self):return []

_normalizer=module.VQAEval(_Empty(),_Empty())

def normalize(answer):
    text=answer.replace('\n',' ').replace('\t',' ').strip()
    return _normalizer.processDigitArticle(_normalizer.processPunctuation(text))

def consensus_score(answer,references):
    if len(references)!=10:raise ValueError('ten human reference answers required')
    prediction=normalize(answer);refs=[normalize(a) for a in references]
    return sum(min(1.0,sum(value==prediction for j,value in enumerate(refs) if j!=i)/3)
               for i in range(10))/10

def relative_loss(original,treated):
    return 100*(original-treated)/original if original>0 else None
