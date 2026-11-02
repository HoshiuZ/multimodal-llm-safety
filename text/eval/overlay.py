"""Safe restore of sparse, full-parameter HF tensor updates."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path


def read_overlay(path, config, *, require_frozen=True):
    """Hash and deserialize the same retained bytes using PyTorch weights_only."""
    import torch

    path = Path(path)
    spec = config['update']
    if not path.is_file() or (require_frozen and path.stat().st_size != spec['bytes']):
        raise ValueError('sparse update file missing or byte size differs from the frozen config')
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if require_frozen and actual != spec['sha256']:
        raise ValueError('sparse update SHA256 differs from the frozen config')
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if not isinstance(state, dict) or set(state) != set(spec['parameters']):
        raise ValueError('sparse update keys differ from the frozen parameter list')
    for name, tensor in state.items():
        metadata = spec['parameters'][name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f'non-tensor sparse value: {name}')
        if list(tensor.shape) != metadata['shape'] or str(tensor.dtype) != metadata['dtype']:
            raise ValueError(f'sparse tensor shape/dtype mismatch: {name}')
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f'non-finite sparse tensor: {name}')
    del payload
    return state


def restore_overlay(model, state, config):
    """Copy only listed parameters without changing their dtype, then compare bitwise."""
    import torch

    spec = config['update']
    parameters = dict(model.named_parameters())
    restored = {}
    with torch.no_grad():
        for name, tensor in state.items():
            parameter = parameters.get(name)
            if parameter is None:
                raise ValueError(f'sparse parameter is absent from the full wrapper: {name}')
            metadata = spec['parameters'][name]
            if list(parameter.shape) != metadata['shape'] or parameter.dtype != tensor.dtype:
                raise ValueError(f'base parameter shape/dtype mismatch: {name}')
            parameter.copy_(tensor.to(device=parameter.device))
            restored[name] = bool(torch.equal(parameter.detach().cpu(), tensor))
    if not restored or not all(restored.values()):
        raise ValueError('one or more sparse parameters differ after restore')
    return {'status': 'passed', 'keys': sorted(restored), 'all_bitwise_equal': True,
            'dtype': str(next(iter(state.values())).dtype)}
