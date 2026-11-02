"""Short entrypoints for the repository's paired C/S scoring and aggregation.

Paths and pinned model revisions are configured in score_paths.json. Actual
scoring, coverage checks and metrics remain in scripts/text_eval.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SETTINGS = ROOT / 'score_paths.json'
MODEL_IDENTITY = ROOT / 'score_model_identity.json'
TOKEN_PREFIXES = ('tokenizer', 'special_tokens', 'added_tokens', 'merges', 'vocab')


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def load_settings(family, kind):
    settings = json.loads(SETTINGS.read_text(encoding='utf-8'))
    if settings['device'] != 'cuda:0' or kind not in ('C', 'S'):
        raise ValueError('only C/S and one allocated GPU at logical cuda:0 are supported')
    info = settings['models'][family]
    cfg = json.loads((ROOT / info['config']).read_text(encoding='utf-8'))
    if (cfg['family'] != family or cfg['repository'] != info['base_repo'] or
            cfg['base_revision'] != info['base_revision']):
        raise ValueError('base identity differs from repository configuration')
    return settings, info, info['evaluations'][kind]


def verify_model_files(family, info):
    """Check pinned base sidecars and every released cleared weight shard."""
    cfg = json.loads((ROOT / info['config']).read_text(encoding='utf-8'))
    reference = json.loads(MODEL_IDENTITY.read_text(encoding='utf-8'))[family]
    restored = ROOT / info['cleared_dir'] / 'restoration_record.json'
    if restored.is_file():
        record = json.loads(restored.read_text(encoding='utf-8'))
        if (record.get('schema') != 'local-restored-model-v1' or record.get('family') != family or
                record.get('base_repository') != cfg['repository'] or
                record.get('base_revision') != cfg['base_revision'] or
                record.get('full_state_bitwise_equal') is not True or
                not record.get('files')):
            raise ValueError('invalid local restoration record')
        reference = dict(reference, cleared_repo=info['cleared_repo'],
                         cleared_revision=info['cleared_revision'], files=record['files'])
    if (reference['cleared_repo'] != info['cleared_repo'] or
            reference['cleared_revision'] != info['cleared_revision']):
        raise ValueError('cleared model identity metadata does not match settings')
    for dirname, items in ((info['base_dir'], cfg.get('base_small_files', {})),
                           (info['cleared_dir'], reference['files'])):
        directory = ROOT / dirname
        for name, expected in items.items():
            if Path(name).name != name:
                raise ValueError('unsafe model identity filename')
            path = directory / name
            if not path.is_file() or path.stat().st_size != expected['bytes'] or file_hash(path) != expected['sha256']:
                raise ValueError(f'model identity mismatch: {dirname}/{name}')
    index = json.loads((ROOT / info['cleared_dir'] / 'model.safetensors.index.json').read_text(encoding='utf-8'))
    indexed = set(index['weight_map'].values())
    expected_shards = {n for n in reference['files'] if n.endswith('.safetensors')}
    if not indexed or indexed != expected_shards:
        raise ValueError('cleared model shard index differs from pinned identity')


def commands(settings, info, evaluation, kind, action):
    common = ['--config', info['config'], '--data', settings['datasets'][kind],
              '--base-model', info['base_dir'], '--cleared-model', info['cleared_view'],
              '--preflight', evaluation['preflight']]
    exe = [sys.executable, '-u', '-B', 'scripts/text_eval.py']
    if action == 'score':
        return [exe + ['preflight'] + common,
                exe + ['run'] + common + ['--state', 'base', '--device', 'cuda:0', '--output', evaluation['output_dir']],
                exe + ['run'] + common + ['--state', 'cleared', '--device', 'cuda:0', '--output', evaluation['output_dir']]]
    if action == 'aggregate':
        return [exe + ['aggregate'] + common + ['--output', evaluation['output_dir']]]
    raise ValueError('unknown action')


def token_files(path):
    return {p.name: p for p in path.iterdir() if p.is_file() and
            p.name.startswith(TOKEN_PREFIXES) and p.suffix in ('.json', '.py', '.txt', '.model')}


def cleared_view(family, kind, settings, info):
    """Use base tokenizer only when all scoring token streams agree exactly."""
    from eval.data import load_bundle
    from eval.models import tokenizer, context_limit
    from eval.scoring import token_preflight

    base, cleared = ROOT / info['base_dir'], ROOT / info['cleared_dir']
    b, c = token_files(base), token_files(cleared)
    same_files = b.keys() == c.keys() and all(file_hash(b[k]) == file_hash(c[k]) for k in b)
    proof = {'dataset_kind': kind, 'mode': 'identical-tokenizer-view', 'token_stream': None}
    if not same_files:
        bundle = load_bundle(ROOT / settings['datasets'][kind])
        limit = context_limit(base)
        if limit != context_limit(cleared):
            raise ValueError('model context limits differ')
        left = token_preflight(tokenizer(base, family), bundle, limit)
        right = token_preflight(tokenizer(cleared, family), bundle, limit)
        if left != right:
            raise ValueError(f'base and cleared tokenization differ on evaluated {kind} inputs; stop')
        proof = {'dataset_kind': kind, 'mode': 'verified-base-tokenizer-view',
                 'token_stream': left['token_stream_sha256']}
    view = ROOT / info['cleared_view']
    desired = {p.name: p.resolve() for p in cleared.iterdir() if p.is_file() and p.name not in c}
    desired.update({name: p.resolve() for name, p in b.items()})
    if view.exists():
        existing = {p.name for p in view.iterdir()}
        if existing != set(desired) or any(not (view / n).is_symlink() or
                                          (view / n).resolve() != target for n, target in desired.items()):
            raise ValueError('existing evaluation view differs; refusing to overwrite')
    else:
        view.mkdir(parents=True, exist_ok=False)
        for name, source in desired.items():
            (view / name).symlink_to(source)
    return proof


def scorer_name(family, kind):
    return f'score_{family}{"_s" if kind == "S" else ""}.py'


def main(family, kind='C', action='score'):
    parser = argparse.ArgumentParser(description=f'{family}: {kind} {action}')
    parser.add_argument('--dry-run', action='store_true', help='show plan without model, GPU, network or writes')
    if action == 'score':
        parser.add_argument('--prepare-base', action='store_true', help='download the pinned official base to a new local directory')
    args = parser.parse_args()
    if args.dry_run and getattr(args, 'prepare_base', False):
        parser.error('--dry-run and --prepare-base cannot be combined')
    settings, info, evaluation = load_settings(family, kind)
    plan = commands(settings, info, evaluation, kind, action)
    if args.dry_run:
        print(json.dumps({'model': family, 'dataset': kind, 'action': action,
                          'settings': info, 'data': settings['datasets'][kind], 'steps': plan,
                          'note': 'No model loading, GPU use, downloads or filesystem writes.'},
                         ensure_ascii=False, indent=2))
        return
    if getattr(args, 'prepare_base', False):
        target = ROOT / info['base_dir']
        if target.exists() or target.is_symlink():
            raise SystemExit(f'基模路径已存在，拒绝覆盖: {target}')
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=info['base_repo'], revision=info['base_revision'], local_dir=target)
        print('官方基模准备完成:', target)
        return
    missing = [p for p in (ROOT / settings['datasets'][kind], ROOT / info['base_dir'], ROOT / info['cleared_dir'])
               if not p.exists()]
    if missing:
        raise SystemExit('缺少数据或模型: ' + ', '.join(map(str, missing)) +
                         f'。基模可提前运行 python {scorer_name(family, kind)} --prepare-base 获取；清除态模型由送测方提供。')
    from eval.data import load_bundle
    if load_bundle(ROOT / settings['datasets'][kind])['kind'] != kind:
        raise ValueError('dataset kind differs from requested evaluation')
    out = ROOT / evaluation['output_dir']
    ident = {'family': family, 'dataset': kind, 'settings_sha256': file_hash(SETTINGS),
             'entry_sha256': file_hash(ROOT / scorer_name(family, kind)),
             'runner_sha256': file_hash(Path(__file__).resolve()),
             'model_identity_sha256': file_hash(MODEL_IDENTITY)}
    restored_record = ROOT / info['cleared_dir'] / 'restoration_record.json'
    if restored_record.is_file():
        ident['restoration_record_sha256'] = file_hash(restored_record)
    marker = out / 'entrypoint_identity.json'
    complete = out / 'scoring_complete.json'
    if action == 'aggregate':
        if not marker.is_file() or json.loads(marker.read_text()) != ident:
            raise ValueError('matching scoring entrypoint identity is required for aggregate')
        if not complete.is_file() or json.loads(complete.read_text()) != ident:
            raise ValueError('both base and cleared scoring must finish in this test before aggregate')
        proof_path = out / 'tokenizer_view.json'
        if not proof_path.is_file() or not (ROOT / info['cleared_view']).is_dir():
            raise ValueError('matching tokenizer proof/view is required for aggregate')
        if json.loads(proof_path.read_text())['dataset_kind'] != kind:
            raise ValueError('tokenizer proof belongs to another dataset')
    else:
        if complete.exists():
            raise ValueError('scoring already completed; choose a new output directory for a new test')
        verify_model_files(family, info)
        out.mkdir(parents=True, exist_ok=True)
        if marker.exists() and json.loads(marker.read_text()) != ident:
            raise ValueError('entrypoint identity changed; use a new output directory')
        if not marker.exists():
            if any(out.glob('*.jsonl')) or (out / 'contract.json').exists():
                raise ValueError('existing scores have no entrypoint identity; use a new output directory')
            marker.write_text(json.dumps(ident, ensure_ascii=False, indent=2) + '\n')
        proof = cleared_view(family, kind, settings, info)
        proof_path = out / 'tokenizer_view.json'
        if proof_path.exists() and json.loads(proof_path.read_text()) != proof:
            raise ValueError('tokenizer proof changed; use a new output directory')
        if not proof_path.exists():
            proof_path.write_text(json.dumps(proof, ensure_ascii=False, indent=2) + '\n')
    print(f'开始 {family} 的 {kind} {"预检及配对评分" if action == "score" else "配对核验及汇总"}。', flush=True)
    print('基模文件与清除态权重身份核验遵循评分记录及预检合同。', flush=True)
    for cmd in plan:
        print('执行:', ' '.join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
    if action == 'score':
        complete.write_text(json.dumps(ident, ensure_ascii=False, indent=2) + '\n')
    print('完成。', flush=True)
