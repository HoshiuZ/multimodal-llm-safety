"""Strict external text dataset schema. No benchmark contents bundled."""
from collections import Counter
from pathlib import Path
from .common import read, loads, digest

COUNTS = {'C': {'bio': 1273, 'cyber': 1987, 'chem': 408},
          'S': {'bio': 255, 'cyber': 397, 'chem': 82},
          'M': None}
SCENES = ('original', 'syntax', 'paraphrase')
OTHER49_EXCLUDED_SUBJECTS = frozenset({
    'college_biology', 'high_school_biology', 'virology',
    'college_chemistry', 'high_school_chemistry', 'computer_security',
    'college_computer_science', 'high_school_computer_science',
})


def validate_rows(rows, kind=None):
    seen = set()
    for row in rows:
        inferred_kind = kind or ('M' if 'subject' in row else 'C')
        required = {'question_id', 'source_row_sha256', 'question', 'choices', 'answer_index'}
        required.add('subject' if inferred_kind == 'M' else 'domain')
        if inferred_kind == 'M':
            required.add('description')
        if not required <= row.keys():
            raise ValueError('row lacks required schema fields')
        if not isinstance(row['question_id'], str) or not row['question_id'] or row['question_id'] in seen:
            raise ValueError('empty or duplicate question_id')
        seen.add(row['question_id'])
        if inferred_kind == 'M':
            if not isinstance(row['subject'], str) or not row['subject'].strip():
                raise ValueError('empty MMLU subject')
            if not isinstance(row['description'], str) or not row['description'].strip():
                raise ValueError('empty MMLU subject description')
        elif row['domain'] not in COUNTS['C']:
            raise ValueError('unknown domain')
        if not isinstance(row['question'], str) or not row['question'].strip():
            raise ValueError('empty question')
        if (not isinstance(row['choices'], list) or len(row['choices']) != 4 or
                any(not isinstance(c, str) or not c.strip() for c in row['choices'])):
            raise ValueError('four nonempty choices required')
        if type(row['answer_index']) is not int or not 0 <= row['answer_index'] < 4:
            raise ValueError('answer_index must be integer 0..3')
        h = row['source_row_sha256']
        if not isinstance(h, str) or len(h) != 64 or any(c not in '0123456789abcdef' for c in h):
            raise ValueError('source_row_sha256 must be lowercase SHA256')
    return rows


def load_bundle(manifest_path):
    path = Path(manifest_path)
    spec = read(path)
    if spec.get('schema') != 'text-eval-v1' or spec.get('kind') not in COUNTS:
        raise ValueError('invalid schema/kind')
    if not isinstance(spec.get('source'), str) or not spec['source'].strip():
        raise ValueError('source provenance label required')
    expected = SCENES if spec['kind'] == 'S' else ('original',)
    if set(spec.get('scenes', {})) != set(expected):
        raise ValueError('missing/extra scenes')
    scenes = {}
    for name in expected:
        p = path.parent / spec['scenes'][name]
        with p.open(encoding='utf-8') as stream:
            rows = [loads(line) for line in stream if line.strip()]
        validate_rows(rows, spec['kind'])
        if spec['kind'] == 'M':
            counts = dict(Counter(r['subject'] for r in rows))
            subjects = spec.get('subjects')
            if (len(rows) != 245 or len(counts) != 49 or set(counts.values()) != {5}
                    or not isinstance(subjects, list) or len(subjects) != 49 or len(set(subjects)) != 49
                    or set(subjects) != set(counts)
                    or any(subject in OTHER49_EXCLUDED_SUBJECTS for subject in counts)):
                raise ValueError('other49 must contain exactly 49 subjects by five rows')
        elif dict(Counter(r['domain'] for r in rows)) != COUNTS[spec['kind']]:
            raise ValueError('incomplete or extra domain coverage; denominator cannot change')
        scenes[name] = rows
    base = {r['question_id']: r for r in scenes['original']}
    for rows in scenes.values():
        if set(base) != {r['question_id'] for r in rows}:
            raise ValueError('missing/extra paired IDs')
        for row in rows:
            stable = ('source_row_sha256', 'subject', 'description', 'choices', 'answer_index') if spec['kind'] == 'M' else (
                'source_row_sha256', 'domain', 'choices', 'answer_index')
            if any(row[k] != base[row['question_id']][k] for k in stable):
                raise ValueError('paired source/options/answer/domain changed')
    # Ignore transport paths and unused historical metadata; bind actual scoring inputs.
    fields = (('question_id', 'source_row_sha256', 'subject', 'description', 'question', 'choices', 'answer_index')
              if spec['kind'] == 'M' else
              ('question_id', 'source_row_sha256', 'domain', 'question', 'choices', 'answer_index'))
    normalized = {name: [{k: r[k] for k in fields} for r in rows] for name, rows in scenes.items()}
    identity = {'kind': spec['kind'], 'source': spec['source'], 'scenes': normalized}
    return {'kind': spec['kind'], 'source': spec['source'], 'scenes': normalized, 'digest': digest(identity)}
