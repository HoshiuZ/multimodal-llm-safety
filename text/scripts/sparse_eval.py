#!/usr/bin/env python3
"""Score an exact sparse update on top of a pinned full multimodal checkpoint."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import code_identity, digest, read, sha, write
from eval.data import load_bundle
from eval.overlay import read_overlay, restore_overlay
from eval.paired_audit import audit_pair
from eval.s_audit import audit_s_pair
from eval.scoring import PROMPT_CONTRACT, token_preflight, score_one
from eval import models
from eval.worker import run_worker


def base_identity(model_dir, config):
    """Bind architecture/tokenizer files without rehashing multi-GB base shards."""
    root = Path(model_dir)
    if not root.is_dir():
        raise FileNotFoundError(f'full pinned model directory missing: {root}')
    actual = {}
    for name, expected in config['base_small_files'].items():
        path = root / name
        if not path.is_file() or path.stat().st_size != expected['bytes'] or sha(path) != expected['sha256']:
            raise ValueError(f'pinned base metadata/tokenizer file mismatch: {name}')
        actual[name] = expected
    model_config = read(root / 'config.json')
    if model_config.get('architectures') != [config['model_class']]:
        raise ValueError('pinned full multimodal wrapper architecture mismatch')
    return {'repository': config['repository'], 'revision': config['base_revision'],
            'architecture': config['model_class'], 'small_files': actual}


def prepare(config_path, data_path, base_model, overlay_path):
    config = read(config_path)
    environment = models.check_environment(config)
    bundle = load_bundle(data_path)
    base = base_identity(base_model, config)
    tok = models.tokenizer(base_model, config['family'])
    limit = models.context_limit(base_model)
    tokens = token_preflight(tok, bundle, limit)
    # Verify and hash the exact retained sparse bytes even for the base-state job.
    update = read_overlay(overlay_path, config)
    del update
    identity = {
        'schema': 'pkg2-sparse-text-contract-v1', 'config': config,
        'environment': environment, 'code': code_identity(), 'prompt': PROMPT_CONTRACT,
        'data': bundle['digest'], 'base': base, 'tokens': tokens,
    }
    return identity, bundle, tok, limit


def resolve_cuda_selection(available, count, visible_devices):
    """Accept only one CUDA logical device; never infer a physical index."""
    if not available or count < 1:
        raise RuntimeError('CUDA is unavailable or no device is visible')
    selected = (visible_devices or '').strip()
    if selected and len([part for part in selected.split(',') if part.strip()]) != 1:
        raise RuntimeError('multiple CUDA_VISIBLE_DEVICES entries have no verified single-card mapping')
    if count != 1:
        raise RuntimeError(f'CUDA enumerates {count} devices; a single-card mapping is required')
    return 'cuda:0'


def _device_properties(torch, index):
    properties = torch.cuda.get_device_properties(index)
    uuid = getattr(properties, 'uuid', None)
    if isinstance(uuid, bytes):
        uuid = uuid.decode('ascii', errors='replace')
    pci = getattr(properties, 'pci_bus_id', None)
    if uuid:
        return {'name': properties.name, 'uuid': str(uuid), 'pci_bus_id': None if pci is None else str(pci)}
    if pci:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=pci.bus_id,uuid', '--format=csv,noheader'],
            check=True, capture_output=True, text=True)
        normalize = lambda value: ':'.join(value.strip().lower().split(':')[-2:])
        matches = []
        for line in result.stdout.splitlines():
            fields = [part.strip() for part in line.split(',', 1)]
            if len(fields) == 2 and normalize(fields[0]) == normalize(str(pci)):
                matches.append(fields[1])
        if len(matches) == 1:
            return {'name': properties.name, 'uuid': matches[0], 'pci_bus_id': str(pci)}
    return {'name': properties.name, 'uuid': None, 'pci_bus_id': None if pci is None else str(pci)}


def probe_cuda():
    """Print scheduler-process CUDA evidence and return the unique logical device."""
    import torch
    available = bool(torch.cuda.is_available())
    count = int(torch.cuda.device_count())
    devices = [_device_properties(torch, index) for index in range(count)] if available else []
    snapshot = {'job_id': os.environ.get('LSB_JOBID'), 'hostname': __import__('socket').gethostname(),
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'cuda_is_available': available, 'cuda_device_count': count, 'visible_devices': devices,
                'scheduler_request': 'JSUB single-host, one-GPU request'}
    print(json.dumps(snapshot, ensure_ascii=False, sort_keys=True), flush=True)
    logical = resolve_cuda_selection(available, count, os.environ.get('CUDA_VISIBLE_DEVICES'))
    if not devices or not devices[0]['uuid']:
        raise RuntimeError('the unique CUDA device UUID could not be established from runtime properties')
    return logical


def validate_runtime_contract(contract_path, config_path, data_path, base_model, overlay_path, state):
    """Check run-time identity against a completed tokenizer preflight without re-tokenizing."""
    identity = read(contract_path)
    config = read(config_path)
    environment = models.check_environment(config)
    bundle = load_bundle(data_path)
    base = base_identity(base_model, config)
    if (identity.get('schema') != 'pkg2-sparse-text-contract-v1'
            or identity.get('config') != config
            or identity.get('environment') != environment
            or identity.get('code') != code_identity()
            or identity.get('prompt') != PROMPT_CONTRACT
            or identity.get('data') != bundle['digest']
            or identity.get('base') != base
            or not isinstance(identity.get('tokens'), dict)):
        raise ValueError('runtime inputs differ from the frozen preflight contract')
    if state == 'cleared':
        update = read_overlay(overlay_path, config)
    else:
        path = Path(overlay_path)
        if not path.is_file() or path.stat().st_size != config['update']['bytes']:
            raise ValueError('the hash-pinned sparse update is missing or has a different byte size')
        update = None
    tok = models.tokenizer(base_model, config['family'])
    limit = models.context_limit(base_model)
    if identity['tokens'].get('limit') != limit:
        raise ValueError('model context limit differs from tokenizer preflight')
    return identity, bundle, tok, limit, update


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    validate = sub.add_parser('validate', help='validate external frozen C or other49 bundle')
    validate.add_argument('--data', required=True, type=Path)

    sub.add_parser('device-probe', help='record CUDA visibility/UUID and require one unambiguous logical device')

    for command in ('preflight', 'run'):
        item = sub.add_parser(command)
        item.add_argument('--config', required=True, type=Path)
        item.add_argument('--data', required=True, type=Path)
        item.add_argument('--base-model', required=True, type=Path)
        item.add_argument('--overlay', required=True, type=Path)
    restore = sub.add_parser('restore-check', help='load the pinned full wrapper on CPU and compare every overlay tensor bitwise')
    restore.add_argument('--config', required=True, type=Path)
    restore.add_argument('--base-model', required=True, type=Path)
    restore.add_argument('--overlay', required=True, type=Path)
    restore.add_argument('--output', required=True, type=Path)
    sub.choices['preflight'].add_argument('--output', required=True, type=Path)
    run = sub.choices['run']
    run.add_argument('--contract', required=True, type=Path)
    run.add_argument('--state', choices=('base', 'cleared'), required=True)
    run.add_argument('--device', default='cuda:0')
    run.add_argument('--output', required=True, type=Path)
    audit = sub.add_parser('audit', help='independent CPU paired per-question audit')
    audit.add_argument('--data', required=True, type=Path)
    audit.add_argument('--base-scores', required=True, type=Path)
    audit.add_argument('--cleared-scores', required=True, type=Path)
    audit.add_argument('--base-contract', type=Path,
                       help='preflight contract JSON used to verify every base score row')
    audit.add_argument('--cleared-contract', type=Path,
                       help='preflight contract JSON used to verify every cleared score row')
    audit.add_argument('--output', required=True, type=Path)

    audit_s = sub.add_parser('audit-s', help='independent CPU audit and exact aggregation for a three-scene S bundle')
    audit_s.add_argument('--data', required=True, type=Path)
    audit_s.add_argument('--base-scores', required=True, type=Path)
    audit_s.add_argument('--cleared-scores', required=True, type=Path)
    audit_s.add_argument('--base-contract', required=True, type=Path)
    audit_s.add_argument('--cleared-contract', required=True, type=Path)
    audit_s.add_argument('--threshold', type=int, default=65)
    audit_s.add_argument('--output', required=True, type=Path)

    args = parser.parse_args()
    if args.command == 'device-probe':
        probe_cuda()
        return
    if args.command == 'validate':
        bundle = load_bundle(args.data)
        print({'kind': bundle['kind'], 'counts': {k: len(v) for k, v in bundle['scenes'].items()},
               'digest': bundle['digest']})
        return

    if args.command == 'audit':
        if (args.base_contract is None) != (args.cleared_contract is None):
            raise ValueError('provide both --base-contract and --cleared-contract, or neither')
        bundle = load_bundle(args.data)
        base_contract = None if args.base_contract is None else digest(read(args.base_contract))
        cleared_contract = None if args.cleared_contract is None else digest(read(args.cleared_contract))
        result = audit_pair(bundle, args.base_scores, args.cleared_scores,
                            base_contract=base_contract, cleared_contract=cleared_contract)
        write(args.output, result)
        print(result)
        return

    if args.command == 'audit-s':
        bundle = load_bundle(args.data)
        result = audit_s_pair(bundle, args.base_scores, args.cleared_scores,
                              args.base_contract, args.cleared_contract, args.threshold)
        write(args.output, result)
        os.chmod(args.output, 0o600)
        print(json.dumps({'kind': 'S', 'coverage': result['coverage'],
                          'counts': result['counts'], 'raw_S': result['raw_S'],
                          'capped_S': result['capped_S'], 'S': result['S']}, sort_keys=True))
        return

    if args.command == 'restore-check':
        import torch
        config = read(args.config)
        environment = models.check_environment(config)
        base_identity(args.base_model, config)
        state = read_overlay(args.overlay, config)
        torch.set_num_threads(4)
        model = models.load_full(args.base_model, config, 'cpu')
        restored = restore_overlay(model, state, config)
        write(args.output, {'status': 'passed', 'model': config['family'],
                            'base_revision': config['base_revision'], 'environment': environment,
                            'update_sha256': config['update']['sha256'], **restored})
        print({'status': 'passed', 'model': config['family'], 'restored_keys': len(restored['keys'])})
        return

    if args.command == 'run':
        import torch
        torch.set_num_threads(4)
        identity, bundle, tok, limit, update = validate_runtime_contract(
            args.contract, args.config, args.data, args.base_model, args.overlay, args.state)
        if args.device.startswith('cuda'):
            logical = probe_cuda()
            if args.device != logical:
                raise ValueError('only the unique verified logical CUDA device may be used')
        elif args.device != 'cpu':
            raise ValueError('device must be cpu or the unique verified cuda:0')
        config = identity['config']
        model = models.load_full(args.base_model, config, args.device)
        if args.state == 'cleared':
            restore_overlay(model, update, config)
        scorer = lambda row: score_one(model, tok, config['family'], row, limit, args.device)
        print({'valid_rows': run_worker(bundle, identity, args.state, args.output, scorer)})
        return

    identity, bundle, tok, limit = prepare(args.config, args.data, args.base_model, args.overlay)
    if args.command == 'preflight':
        if args.output.exists() and read(args.output) != identity:
            raise ValueError('refuse to replace a different frozen preflight')
        write(args.output, identity)
        print({'status': 'passed', 'kind': bundle['kind'], 'rows': sum(map(len, bundle['scenes'].values())),
               'token_stream_sha256': identity['tokens']['token_stream_sha256']})
        return



if __name__ == '__main__':
    main()
