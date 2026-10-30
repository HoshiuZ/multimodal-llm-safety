"""Append-only resumable worker and strict paired aggregation."""
import math
import os
from pathlib import Path
from .common import canonical, digest, loads, read, write
from .scoring import LETTERS
from .metrics import metrics


def scopes(bundle):
    return {(scene, r['question_id']): r for scene, rows in bundle['scenes'].items() for r in rows}


def check_record(record, expected, contract, state):
    key = (record.get('scene'), record.get('question_id'))
    if key not in expected:
        raise ValueError('extra record scope')
    row = expected[key]
    if record.get('contract') != contract or record.get('state') != state:
        raise ValueError('score identity mismatch')
    if record.get('row_digest') != digest(row) or record.get('status') != 'ok':
        raise ValueError('failed/changed row')
    scores = record.get('scores')
    if not isinstance(scores, dict) or set(scores) != set(LETTERS) or any(
            type(v) not in (float, int) or not math.isfinite(v) for v in scores.values()):
        raise ValueError('invalid option scores')
    pred = max(LETTERS, key=lambda letter: scores[letter])
    if record.get('prediction') != pred or record.get('target') != LETTERS[row['answer_index']]:
        raise ValueError('prediction/target mismatch')
    if type(record.get('correct')) is not bool or record['correct'] != (pred == record['target']):
        raise ValueError('correctness mismatch')
    return key


def records(path, bundle, contract, state, complete=False):
    expected, seen = scopes(bundle), {}
    if Path(path).exists():
        with Path(path).open(encoding='utf-8') as stream:
            for line in stream:
                # A torn last line is an explicit failure; never silently discard scores.
                if not line.endswith('\n'):
                    raise ValueError('truncated output line; explicit repair required')
                record = loads(line)
                key = check_record(record, expected, contract, state)
                if key in seen:
                    raise ValueError('duplicate record scope')
                seen[key] = record
    if complete and set(seen) != set(expected):
        raise ValueError(f'incomplete coverage: {len(set(expected) - set(seen))} missing')
    return seen


def run_worker(bundle, identity, state, output, scorer, progress_every=250):
    if state not in ('base', 'cleared'):
        raise ValueError('unknown state')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(root, 0o700)
    # Cooperative lock prevents simultaneous runs from duplicating append records.
    import fcntl
    with (root / '.worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = digest(identity)
        manifest = root / 'contract.json'
        if manifest.exists():
            if read(manifest) != identity:
                raise ValueError('output belongs to different model/code/prompt/data identity')
        else:
            if any(root.glob('*.jsonl')):
                raise ValueError('score records lack original contract')
            write(manifest, identity)
        os.chmod(manifest, 0o600)
        expected = scopes(bundle)
        path = root / f'{state}.jsonl'
        seen = records(path, bundle, contract, state)
        if path.exists():
            os.chmod(path, 0o600)
        with path.open('a', encoding='utf-8') as stream:
            for offset, ((scene, qid), row) in enumerate(expected.items(), start=1):
                if (scene, qid) in seen:
                    continue
                scores = scorer(row)
                pred = max(LETTERS, key=lambda letter: scores[letter])
                rec = {'contract': contract, 'state': state, 'scene': scene, 'question_id': qid,
                       'row_digest': digest(row), 'scores': scores, 'prediction': pred,
                       'target': LETTERS[row['answer_index']], 'correct': pred == LETTERS[row['answer_index']],
                       'status': 'ok'}
                check_record(rec, expected, contract, state)
                stream.write(canonical(rec) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                if offset == 1:
                    print(f'first_scorer_returned state={state} kind={bundle["kind"]}', flush=True)
                if progress_every and (offset % progress_every == 0 or offset == len(expected)):
                    print(f'progress state={state} kind={bundle["kind"]} rows={offset}/{len(expected)}', flush=True)
        return len(records(path, bundle, contract, state, complete=True))


def aggregate(bundle, identity, output, threshold=65):
    root = Path(output)
    if read(root / 'contract.json') != identity:
        raise ValueError('aggregate contract mismatch')
    contract = digest(identity)
    states = {s: records(root / f'{s}.jsonl', bundle, contract, s, complete=True) for s in ('base', 'cleared')}
    counts = {}
    by_domain = {}
    by_subject = {}
    for scene, rows in bundle['scenes'].items():
        keys = [(scene, row['question_id']) for row in rows]
        counts[scene] = {'n': len(keys), **{s: sum(states[s][k]['correct'] for k in keys) for s in states}}
        if bundle['kind'] == 'M':
            by_subject[scene] = {}
            for subject in sorted({r['subject'] for r in rows}):
                selected = [(scene, r['question_id']) for r in rows if r['subject'] == subject]
                by_subject[scene][subject] = {'n': len(selected), **{s: sum(states[s][k]['correct'] for k in selected) for s in states}}
        else:
            by_domain[scene] = {}
            for domain in ('bio', 'cyber', 'chem'):
                selected = [(scene, r['question_id']) for r in rows if r['domain'] == domain]
                by_domain[scene][domain] = {'n': len(selected), **{s: sum(states[s][k]['correct'] for k in selected) for s in states}}
    result = {'contract': contract, 'kind': bundle['kind'], 'source': bundle['source'],
              'counts': counts, 'by_domain': by_domain}
    if bundle['kind'] == 'M':
        result['by_subject'] = by_subject
    else:
        result['metrics'] = metrics(counts, bundle['kind'], threshold)
    return result
