#!/usr/bin/env python3
"""Prepare all pinned runtime downloads for the Word procedure in one command.

The environment step invokes this after both requirements files are installed.
Model files are obtained from immutable HF revisions. Images can be restored
from either a Git checkout or a source ZIP without requiring Git LFS locally.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile

import document_steps as D

DATA_COMMIT = '73fe3d48007732255b437ebff6d54d8c0933561e'
MEDIA_ROOT = f'https://media.githubusercontent.com/media/HoshiuZ/multimodal-llm-safety/{DATA_COMMIT}'


def model_downloads(cfg):
    registry = D.read_json(D.ROOT / 'configs/model_registry.json')
    root = D.path(cfg['models_root'])
    downloads = []
    for key, spec in registry['models'].items():
        base, adapter = D.MODELS[key]
        downloads.extend([
            (spec['hf_id'], spec['base_revision'], root / base),
            (spec['adapter_repo'], spec['adapter_revision'], root / 'adapters' / adapter),
        ])
    judge = registry['judge']
    downloads.append((judge['hf_id'], judge['revision'], root / 'Qwen2.5-14B-Instruct'))
    return downloads


def download_image(session, destination, relative, expected_sha256):
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.download-', dir=destination.parent)
    try:
        h = hashlib.sha256()
        with os.fdopen(fd, 'wb') as out:
            with session.get(f'{MEDIA_ROOT}/{relative}', stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                for chunk in response.iter_content(1 << 20):
                    if chunk:
                        h.update(chunk)
                        out.write(chunk)
        if h.hexdigest() != expected_sha256:
            raise ValueError(f'downloaded image SHA-256 differs: {relative}')
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def prepare_images(cfg, session):
    assets = D.path(cfg['asset_root'])
    sources = [
        (D.path(cfg['official_manifest']), D.path(cfg['official_root'])),
        (assets / 'figstep/manifest.jsonl', assets / 'figstep'),
        (assets / 'hades_local_v1/manifest.jsonl', assets / 'hades_local_v1'),
    ]
    protocol = D.read_json(D.ROOT / 'configs/eval_protocol.json')['asset_release']
    # Never fetch images on the authority of an altered input manifest.
    expected = {'figstep': protocol['figstep_manifest_sha256'], 'hades_local_v1': protocol['hades_local_v1_manifest_sha256']}
    for manifest, root in sources:
        records = D.rows(manifest)
        if root.name in expected:
            if D.digest(manifest) != expected[root.name]:
                raise ValueError(f'input manifest differs from frozen release: {manifest}')
        else:
            ids = sorted(r['sample_id'] for r in records)
            if len(ids) != 5040 or hashlib.sha256('\n'.join(ids).encode()).hexdigest() != protocol['official_mmsb_manifest_digest']:
                raise ValueError('official sample IDs differ from frozen release')
        downloaded = 0
        for row in records:
            relative = Path(row['image_path'])
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('image path must remain within its input directory')
            image = root / relative
            if image.is_file():
                with image.open('rb') as f:
                    if not f.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
                        continue
            try:
                github_path = image.relative_to(D.REPO).as_posix()
            except ValueError as exc:
                raise ValueError(f'external missing images must be supplied by the evaluator: {image}') from exc
            download_image(session, image, github_path, row['img_sha256'])
            downloaded += 1
            if downloaded % 100 == 0:
                print(f'{root.name}: downloaded {downloaded} images', flush=True)
        print(f'PASS {root.name}: restored {downloaded} missing/LFS-pointer images; full validation runs at the data step', flush=True)


def main():
    from huggingface_hub import snapshot_download
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    cfg = D.settings()
    for repo, revision, destination in model_downloads(cfg):
        print(f'Download/check {repo}@{revision} -> {destination}', flush=True)
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(destination))
    with requests.Session() as session:
        session.mount('https://', HTTPAdapter(max_retries=Retry(total=4, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
        prepare_images(cfg, session)
    print('PASS: all model, adapter, judge and image downloads prepared', flush=True)


if __name__ == '__main__':
    main()
