#!/usr/bin/env python3
"""Portable text C/S entrypoint. Run --help without third-party dependencies."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import read, write, digest, code_identity
from eval.data import load_bundle
from eval.scoring import PROMPT_CONTRACT, token_preflight, score_one
from eval import models
from eval.worker import run_worker, aggregate


def prepare(args):
    config = read(args.config)
    environment = models.check_environment(config)
    bundle = load_bundle(args.data)
    paths = {'base': args.base_model, 'cleared': args.cleared_model}
    inventories = {s: models.inventory(p, config) for s, p in paths.items()}
    # Tokenizer content must be identical for paired base/cleared inference.
    token_files = lambda inv: {k: v for k, v in inv['files'].items() if k.startswith(('tokenizer', 'special_tokens', 'added_tokens', 'merges', 'vocab'))}
    if token_files(inventories['base']) != token_files(inventories['cleared']):
        raise ValueError('base/cleared tokenizer mismatch')
    limits = [models.context_limit(p) for p in paths.values()]
    if limits[0] != limits[1]:
        raise ValueError('base/cleared context limits differ')
    tok = models.tokenizer(paths['base'], config['family'])
    tokens = token_preflight(tok, bundle, limits[0])
    identity = {'schema': 'text-eval-contract-v1', 'config': config, 'environment': environment,
                'code': code_identity(), 'prompt': PROMPT_CONTRACT, 'data': bundle['digest'],
                'models': inventories, 'tokens': tokens}
    return identity, bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    validate = sub.add_parser('validate', help='CPU external-data schema/coverage validation')
    validate.add_argument('--data', required=True, type=Path)
    inv = sub.add_parser('model-preflight', help='CPU full local HF file inventory (hashes weights)')
    inv.add_argument('--config', required=True, type=Path)
    inv.add_argument('--model', required=True, type=Path)
    inv.add_argument('--output', required=True, type=Path)
    pre = sub.add_parser('preflight', help='CPU paired data/model/token/code identity; no model forward')
    run = sub.add_parser('run', help='score one state; resume only exact identity')
    agg = sub.add_parser('aggregate', help='strict complete-pair CPU C or raw/capped S aggregation')
    for parser in (pre, run, agg):
        parser.add_argument('--config', required=True, type=Path)
        parser.add_argument('--data', required=True, type=Path)
        parser.add_argument('--base-model', required=True, type=Path)
        parser.add_argument('--cleared-model', required=True, type=Path)
        parser.add_argument('--preflight', required=True, type=Path)
    run.add_argument('--state', choices=['base', 'cleared'], required=True)
    run.add_argument('--device', default='cuda:0', help='logical device visible to this process')
    run.add_argument('--output', required=True, type=Path)
    agg.add_argument('--output', required=True, type=Path)
    agg.add_argument('--threshold', type=int, default=65)
    args = p.parse_args()
    if args.command == 'validate':
        bundle = load_bundle(args.data)
        print({'kind': bundle['kind'], 'scene_counts': {k: len(v) for k, v in bundle['scenes'].items()}, 'digest': bundle['digest']})
        return
    if args.command == 'model-preflight':
        config = read(args.config)
        models.check_environment(config)
        write(args.output, models.inventory(args.model, config))
        return
    identity, bundle = prepare(args)
    if args.command == 'preflight':
        if args.preflight.exists():
            if read(args.preflight) != identity:
                raise ValueError('refuse to overwrite different preflight')
        else:
            write(args.preflight, identity)
        print('CPU preflight passed; no full model forward')
        return
    if read(args.preflight) != identity:
        raise ValueError('preflight identity changed; refusing reuse')
    if args.command == 'aggregate':
        result = aggregate(bundle, identity, args.output, args.threshold)
        write(args.output / 'aggregate.json', result)
        if bundle['kind'] == 'S':
            print({'raw_S': result['metrics']['S']['raw_S'],
                   'capped_S': result['metrics']['S']['capped_S'],
                   'passed': result['metrics']['S']['passed']})
        else:
            print(result['metrics'])
        return
    import torch
    torch.set_num_threads(4)
    if args.device.startswith('cuda'):
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1 or args.device != 'cuda:0':
            raise ValueError('expose exactly one allocated GPU; use logical cuda:0')
    config = identity['config']
    path = args.base_model if args.state == 'base' else args.cleared_model
    tok = models.tokenizer(path, config['family'])
    model = models.load_full(path, config, args.device)
    scorer = lambda row: score_one(model, tok, config['family'], row, identity['tokens']['limit'], args.device)
    print({'valid_rows': run_worker(bundle, identity, args.state, args.output, scorer)})


if __name__ == '__main__':
    main()
