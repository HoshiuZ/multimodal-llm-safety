"""Complete local HF checkpoint loading. No ancestor LM substitution or edits."""
import importlib.metadata
from pathlib import Path
from .common import read, sha, digest

PACKAGES = ('torch', 'torchvision', 'transformers', 'accelerate', 'safetensors',
            'huggingface-hub', 'tokenizers', 'Pillow', 'sentencepiece', 'protobuf')


def environment(family):
    import sys
    names = PACKAGES + (('einops', 'timm') if family == 'internvl' else ())
    return {'python': sys.version.split()[0], 'packages': {k: importlib.metadata.version(k) for k in names}}


def check_environment(config):
    actual = environment(config['family'])
    if actual != config['environment']:
        raise ValueError('environment differs from pinned config; use documented per-model environment')
    return actual


def inventory(model_dir, config):
    path = Path(model_dir)
    if not path.is_dir():
        raise FileNotFoundError(f'local full HF model directory missing: {path}')
    conf = read(path / 'config.json')
    if conf.get('architectures') != [config['model_class']]:
        raise ValueError('full multimodal checkpoint architecture required')
    if (path / 'model.safetensors.index.json').exists():
        index = read(path / 'model.safetensors.index.json')
        weight_names = set(index['weight_map'].values())
        if not weight_names:
            raise ValueError('empty weight index')
        names = set(weight_names) | {'model.safetensors.index.json'}
    elif (path / 'model.safetensors').exists():
        names = {'model.safetensors'}
    else:
        raise FileNotFoundError('complete safetensors checkpoint/index missing')
    # Runtime file content, not inode/device or machine paths, defines portable identity.
    names |= {p.name for p in path.iterdir() if p.suffix in ('.json', '.py', '.txt', '.model') and p.is_file()}
    files = {}
    for name in sorted(names):
        p = path / name
        if Path(name).is_absolute() or '..' in Path(name).parts or not p.is_file():
            raise ValueError('unsafe or missing checkpoint shard')
        files[name] = {'size': p.stat().st_size, 'sha256': sha(p)}
    return {'files': files, 'digest': digest(files)}


def context_limit(path):
    conf = read(Path(path) / 'config.json')
    for key in ('text_config', 'llm_config', 'language_config'):
        if isinstance(conf.get(key), dict):
            conf = conf[key]
            break
    limit = conf.get('max_position_embeddings')
    if type(limit) is not int or limit <= 0:
        raise ValueError('explicit max_position_embeddings required')
    return limit


def tokenizer(path, family):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(path), local_files_only=True,
        use_fast=family != 'internvl', trust_remote_code=family == 'internvl')


def load_full(path, config, device):
    import torch
    import transformers
    family = config['family']
    klass = transformers.AutoModel if family == 'internvl' else getattr(transformers, config['model_class'])
    kwargs = dict(local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                  device_map={'': device}, output_loading_info=True)
    if family == 'internvl':
        kwargs.update(trust_remote_code=True, use_flash_attn=False)
    else:
        kwargs['attn_implementation'] = 'sdpa'
    model, info = klass.from_pretrained(str(path), **kwargs)
    if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'incomplete or incompatible full checkpoint load: {info}')
    if type(model).__name__ != config['model_class']:
        raise ValueError('loaded wrapper class mismatch')
    if any(p.dtype != torch.bfloat16 for p in model.parameters()):
        raise ValueError('all parameters must use BF16')
    attention_config = model.language_model.config if family == 'internvl' else model.config
    if getattr(attention_config, '_attn_implementation', None) != config['attention']:
        raise ValueError('attention contract changed')
    model.eval()
    model.requires_grad_(False)
    model.config.use_cache = False
    if hasattr(model, 'language_model'):
        model.language_model.config.use_cache = False
    return model
