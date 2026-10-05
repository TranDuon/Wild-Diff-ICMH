from typing import List, Tuple, Optional
import os
import copy
import gc
import json
import math
import subprocess
import time
import zlib
from datetime import datetime, timezone
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import einops
import lightning.pytorch as pl
from PIL import Image
from omegaconf import OmegaConf

from ldm.xformers_state import disable_xformers
from model.spaced_sampler import SpacedSampler
from model.ddim_sampler import DDIMSampler
from model.diffeic import DiffEIC
from utils.image import pad
from utils.image_geometry import (
    PROCESSING_RESAMPLING,
    center_crop_image,
    resize_for_processing,
    resolve_geometry,
    restore_original_size,
)
from utils.metrics import compute_psnr, compute_ssim, LPIPS
from utils.common import instantiate_from_config, load_state_dict
from utils.checkpoint_contract import (
    migrate_legacy_entropy_bottleneck_checkpoint,
    resolve_run_model_config,
    state_dict_shape_mismatches,
    validate_entropy_bottleneck_checkpoint,
)
from utils.file import list_image_files, get_file_name_parts
from dataset.camera_trap_dataset import encode_domain_metadata, prompt_from_domain_metadata
from utils.illumination import apply_sidecar, load_sidecar


class CachedTagCodec:
    """Encode cached RAM++ ids without loading the multi-GB RAM++ network."""

    def __init__(self, vocabulary_path=None):
        tag_list_path = os.path.join(
            os.path.dirname(__file__), "src", "recognize-anything", "ram", "data", "ram_tag_list.txt"
        )
        with open(tag_list_path, "r", encoding="utf-8") as stream:
            self.tags = [line.strip() for line in stream]
        self.compact_to_original = None
        self.original_to_compact = None
        self.tag_codelength = 13
        if vocabulary_path:
            with open(vocabulary_path, "r", encoding="utf-8") as stream:
                requested = {
                    line.strip().lower() for line in stream
                    if line.strip() and not line.lstrip().startswith("#")
                }
            compact = [i for i, tag in enumerate(self.tags) if tag.lower() in requested]
            missing = sorted(requested - {self.tags[i].lower() for i in compact})
            if missing:
                raise ValueError(f"cached tag vocabulary contains unknown tags: {missing}")
            self.compact_to_original = compact
            self.original_to_compact = {original: index for index, original in enumerate(compact)}
            self.tag_codelength = max(1, math.ceil(math.log2(len(compact))))

    def encode_record(self, record):
        original = [int(value) for value in record.get("tag_ids", [])]
        if self.original_to_compact is None:
            return original
        return [self.original_to_compact[value] for value in original if value in self.original_to_compact]

    def decode_text(self, ids):
        if self.compact_to_original is not None:
            ids = [self.compact_to_original[int(value)] for value in ids]
        return ",".join(self.tags[int(value)] for value in ids)


def _torch_load(path: str):
    """Load large checkpoints without making an avoidable second RAM copy."""
    try:
        return torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_checkpoint(model: DiffEIC, path: str, label: str):
    checkpoint = _torch_load(path)
    checkpoint, migrations = migrate_legacy_entropy_bottleneck_checkpoint(checkpoint)
    if migrations:
        print(f"Migrated {len(migrations)} legacy entropy-bottleneck keys in {label} checkpoint")
    validate_entropy_bottleneck_checkpoint(model, checkpoint)
    mismatches = state_dict_shape_mismatches(model, checkpoint)
    if mismatches:
        preview = "; ".join(
            f"{key}: checkpoint={checkpoint_shape}, model={model_shape}"
            for key, checkpoint_shape, model_shape in mismatches[:8]
        )
        raise RuntimeError(
            f"{label} checkpoint has {len(mismatches)} tensor shape mismatch(es). "
            f"First mismatches: {preview}"
        )
    state_dict = checkpoint.get("state_dict", checkpoint)
    message = load_state_dict(model, state_dict, strict=False)
    missing = list(getattr(message, "missing_keys", ()))
    unexpected = list(getattr(message, "unexpected_keys", ()))
    print(
        f"Loaded {label} checkpoint {path}: "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )
    if unexpected:
        print(f"  First unexpected keys: {unexpected[:5]}")
    del state_dict, checkpoint
    gc.collect()


@torch.no_grad()
def process(
    model: DiffEIC,
    imgs: List[np.ndarray],
    sampler: str,
    steps: int,
    stream_path: str,
    anchor_prior_strength: Optional[float] = None,
    domain_rows: Optional[List[dict]] = None,
    cached_tag_records: Optional[List[dict]] = None,
    cached_tag_codec: Optional[CachedTagCodec] = None,
    domain_site_id: Optional[str] = None,
    domain_habitat: Optional[str] = None,
    timings: Optional[dict] = None,
) -> Tuple[List[np.ndarray], float]:
    """
    Apply DiffEIC model on a list of images.
    
    Args:
        model (DiffEIC): Model.
        imgs (List[np.ndarray]): A list of images (HWC, RGB, range in [0, 255])
        sampler (str): Sampler name.
        steps (int): Sampling steps.
        stream_path (str): Savedir of bitstream
        timings (dict, optional): filled with ``encode_seconds`` (bitstream
            written) and ``decode_seconds`` (bitstream read to pixels).

    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
        bpp
    """
    def _now():
        if model.device.type == "cuda":
            torch.cuda.synchronize(model.device)
        return time.perf_counter()

    encode_started = _now()
    n_samples = len(imgs)
    if sampler == "ddpm":
        sampler = SpacedSampler(model, var_type="fixed_small")
    else:
        sampler = DDIMSampler(model)
    control = torch.tensor(np.stack(imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()
    
    height, width = control.size(-2), control.size(-1)
    tag_enabled = model.preprocess_tag_model.enabled or cached_tag_codec is not None
    if cached_tag_codec is not None:
        if n_samples != 1 or not cached_tag_records or len(cached_tag_records) != 1:
            raise ValueError("cached tag inference currently requires one tag record per single-image batch")
        c_tag_ids = cached_tag_codec.encode_record(cached_tag_records[0])
        tag_codelength = cached_tag_codec.tag_codelength
    elif model.preprocess_tag_model.enabled:
        c_tag_ids, _ = model.preprocess_tag_model(control, return_ids=True)
        c_tag_ids = [each for each in c_tag_ids[0].reshape(-1)]
        tag_codelength = model.preprocess_tag_model.tag_codelength

    domain_codes = None
    if domain_rows is not None:
        if len(domain_rows) != n_samples:
            raise ValueError("domain_rows must match the image batch")
        domain_codes = [encode_domain_metadata(row) for row in domain_rows]

    if tag_enabled:
        bpp = model.apply_condition_compress(
            control,
            stream_path,
            height,
            width,
            tag_ids=c_tag_ids,
            tag_codelength=tag_codelength,
            domain_metadata_codes=domain_codes,
        )
    else:
        bpp = model.apply_condition_compress(
            control, stream_path, height, width, domain_metadata_codes=domain_codes
        )
    decode_started = _now()
    if timings is not None:
        timings["encode_seconds"] = decode_started - encode_started
    if tag_enabled:
        if domain_codes is not None:
            c_latent, c_tag_ids, decoded_domain_codes = model.apply_condition_decompress(
                stream_path, dec_tag_ids=True, dec_domain_metadata=True,
                domain_metadata_count=n_samples,
            )
        else:
            c_latent, c_tag_ids = model.apply_condition_decompress(stream_path, dec_tag_ids=True)
            decoded_domain_codes = None
        # Convert decoded tag IDs to tensor format for index2tag
        # c_tag_ids_tensor = torch.tensor(c_tag_ids).reshape(1, -1)  # Shape: [1, num_tags]
        if cached_tag_codec is not None:
            c_tag_rec = [cached_tag_codec.decode_text(c_tag_ids)]
        else:
            c_tag_ids = model.preprocess_tag_model.expand_tag_ids(c_tag_ids)
            c_tag_ids_np = np.array(c_tag_ids).reshape(-1, 1)
            c_tag_rec = model.preprocess_tag_model.model.index2tag([c_tag_ids_np])[0]
            c_tag_rec = [tag.replace(' |', ',') for tag in c_tag_rec]
    else:
        if domain_codes is not None:
            c_latent, decoded_domain_codes = model.apply_condition_decompress(
                stream_path, dec_domain_metadata=True,
                domain_metadata_count=n_samples,
            )
        else:
            c_latent = model.apply_condition_decompress(stream_path)
            decoded_domain_codes = None

    domain_prompts = (
        [
            prompt_from_domain_metadata(
                code, site_id=domain_site_id, habitat=domain_habitat
            )
            for code in decoded_domain_codes
        ]
        if decoded_domain_codes is not None else None
    )

    cond = {
        "c_latent": [c_latent],
        "c_crossattn": [model.get_learned_conditioning([""] * n_samples)]
    }
    unconditional_guidance_scale = 1.0
    unconditional_conditioning = None
    # if model.preprocess_semantic_model.enabled:
    #     import pdb; pdb.set_trace()
    #     c_semantic, bits_sem = model.preprocess_semantic_model(control)  # control: [0, 1]
    #     cond["c_crossattn"] = [c_semantic] * n_samples
    #     unconditional_guidance_scale = model.c_cfg_scale
    #     unconditional_conditioning = copy.deepcopy(cond)
    #     unconditional_conditioning["c_crossattn"] = [torch.zeros_like(cond["c_crossattn"][0])] * n_samples
    if model.preprocess_semantic_model.enabled or tag_enabled or domain_prompts is not None:
        unconditional_guidance_scale = model.c_cfg_scale
        unconditional_conditioning = copy.deepcopy(cond)
        if model.preprocess_semantic_model.enabled:
            c_semantic, bits_sem = model.preprocess_semantic_model(control)  # control: [0, 1]
            # cond["c_crossattn"] = [c_semantic] * n_samples
            cond_crossattn = c_semantic
            n_tokens = 256 // (model.preprocess_semantic_model.feature_postprocessor.postprocess_downscale**2)
            uncond_semantic = torch.zeros(n_samples, n_tokens, cond["c_crossattn"][0].shape[-1]).to(model.device)
            uncond_crossattn = uncond_semantic
        if tag_enabled:
            # c_tag, bits_tag = model.preprocess_tag_model(control)
            c_tag = c_tag_rec
            if domain_prompts is not None:
                c_tag = [
                    ", ".join(part for part in (tag, prompt) if part)
                    for tag, prompt in zip(c_tag, domain_prompts)
                ]
            cond_tag = model.cond_stage_model.encode(c_tag)
            uncond_tag = model.cond_stage_model.encode([''] * n_samples)
            if model.preprocess_semantic_model.enabled:
                cond_crossattn = torch.cat([cond_crossattn, cond_tag], 1)
                uncond_crossattn = torch.cat([uncond_crossattn, uncond_tag], 1)
            else:
                cond_crossattn = cond_tag
                uncond_crossattn = uncond_tag
        if domain_prompts is not None and not tag_enabled:
            cond_crossattn = model.cond_stage_model.encode(domain_prompts)
            uncond_crossattn = model.cond_stage_model.encode([''] * n_samples)
        cond["c_crossattn"] = [cond_crossattn]
        unconditional_conditioning["c_crossattn"] = [uncond_crossattn]
    
    shape = (n_samples, 4, height // 8, width // 8)
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
    if anchor_prior_strength is not None:
        x_T = x_T + anchor_prior_strength * cond['c_latent'][0]
    if isinstance(sampler, SpacedSampler):
        samples = sampler.sample(
            steps, shape, cond,
            # unconditional_guidance_scale=1.0,
            # unconditional_conditioning=None,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            cond_fn=None, x_T=x_T
        )
    else:
        sampler: DDIMSampler
        samples, _ = sampler.sample(
            S=steps, batch_size=shape[0], shape=shape[1:],
            conditioning=cond,
            unconditional_guidance_scale=unconditional_guidance_scale,
            unconditional_conditioning=unconditional_conditioning,
            x_T=x_T, eta=0
        )
    
    x_samples = model.decode_first_stage(samples)
    x_samples = ((x_samples + 1) / 2).clamp(0, 1)
    
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    
    preds = [x_samples[i] for i in range(n_samples)]
    if timings is not None:
        timings["decode_seconds"] = _now() - decode_started

    return preds, bpp


def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    # TODO: add help info for these options
    parser.add_argument("--ckpt_sd", default='./checkpoints/sd2p1/v2-1_512-ema-pruned.ckpt', type=str, help="checkpoint path of stable diffusion")
    parser.add_argument("--ckpt_lc", default='path to checkpoint file of lfgcm and control module', type=str, help="checkpoint path of lfgcm and control module")
    parser.add_argument("--config", default='configs/model/diffeic.yaml', type=str, help="model config path")
    
    parser.add_argument("--input", type=str, default='path to input images')
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--steps", default=50, type=int)
    
    parser.add_argument("--output", type=str, default='results/')
    parser.add_argument("--limit", type=int, default=None, help="process only the first N selected images")
    parser.add_argument(
        "--crop-size", type=int, default=None,
        help="smoke-test protocol: center-crop each image before coding (e.g. 256)",
    )
    parser.add_argument(
        "--processing-long-side", type=int, default=None,
        help="EVAL-11: shrink the long side to this before coding and restore the original size "
             "after decoding (default 1024 with --manifest; mutually exclusive with --crop-size)",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="resume an interrupted run: skip images whose reconstruction, bitstream and "
             "decode_log.jsonl record already exist",
    )
    
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument('--anchor_prior_strength', type=float, default=None)
    parser.add_argument(
        '--manifest', type=str, default=None,
        help='Camera-trap JSONL manifest; required to transmit H3 day/night+season metadata',
    )
    parser.add_argument('--split', choices=['train', 'val', 'test'], default=None)
    parser.add_argument('--site-id', default=None, help='e.g. KGA:A01; requires --manifest')
    parser.add_argument(
        '--dev-list', default=None,
        help='frozen image_id list (EVAL-16); keeps only those manifest rows, requires --manifest',
    )
    parser.add_argument('--tag-cache', default=None, help='JSONL produced by tools/precompute_ram_tags.py')
    parser.add_argument('--tag-vocabulary', default=None, help='optional restricted vocabulary for H3')
    parser.add_argument('--domain-metadata', action='store_true', help='transmit H3 illumination/season byte')
    parser.add_argument(
        '--illumination-sidecar', default='data/manifests/kgalagadi_illumination.jsonl',
        help='light-source day/night labels (tools/data/label_illumination.py) for H3 metadata',
    )
    parser.add_argument(
        '--habitat-map', default=None,
        help='optional JSON object site_id -> verified habitat label; requires --domain-metadata',
    )
    
    parser.add_argument('overrides', nargs='*', help='override model config keys')

    return parser.parse_args()


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd=os.path.dirname(os.path.abspath(__file__)), text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return 'unknown'


def _write_run_info(args: Namespace, model_config) -> None:
    """Record the decode protocol next to the archive (EVAL-12/EVAL-15)."""
    info = {
        'schema_version': 1,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'git_commit': _git_commit(),
        'checkpoint': os.path.abspath(args.ckpt_lc),
        'checkpoint_bytes': os.path.getsize(args.ckpt_lc) if os.path.isfile(args.ckpt_lc) else None,
        'protocol': 'center_crop' if args.crop_size is not None else 'original_resolution',
        'crop_size': args.crop_size,
        'processing_long_side': args.processing_long_side,
        'resampling': PROCESSING_RESAMPLING,
        'evaluation_resolution': 'crop' if args.crop_size is not None else 'original',
        'sampler': args.sampler,
        'steps': args.steps,
        'seed': args.seed,
        'seed_policy': 'seed + crc32(relative_path) per image',
        'c_cfg_scale': float(model_config.params.get('c_cfg_scale', 0.0)),
        'anchor_prior_strength': args.anchor_prior_strength,
        'tag_cache': args.tag_cache,
        'tag_vocabulary': args.tag_vocabulary,
        'domain_metadata': bool(args.domain_metadata),
        'manifest': args.manifest,
        'split': args.split,
        'site_id': args.site_id,
        'overrides': list(args.overrides or []),
    }
    path = os.path.join(args.output, 'run_info.json')
    temporary = path + '.part'
    with open(temporary, 'w', encoding='utf-8') as stream:
        json.dump(info, stream, indent=2, sort_keys=True)
        stream.write('\n')
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    if args.limit is not None and args.limit <= 0:
        raise ValueError('--limit must be positive')
    args.crop_size, args.processing_long_side = resolve_geometry(
        args.crop_size, args.processing_long_side, manifest_supplied=bool(args.manifest)
    )
    if args.crop_size is not None:
        print(f"Smoke protocol: {args.crop_size}x{args.crop_size} center crop (not for reported numbers)")
    elif args.processing_long_side is not None:
        print(
            f"Original-resolution protocol: code at long side {args.processing_long_side}, "
            "evaluate at the original size"
        )
    pl.seed_everything(args.seed)
    
    if args.device == "cpu":
        disable_xformers()

    # model: DiffEIC = instantiate_from_config(OmegaConf.load(args.config))
    resolved_config = resolve_run_model_config(args.config, args.ckpt_lc)
    if resolved_config != args.config:
        print(f"Using run-matched model config: {resolved_config}")
    model_config = OmegaConf.load(resolved_config)
    # --ckpt_sd below supplies these weights. Avoid loading the same 5+ GB
    # checkpoint once in __init__ and a second time in this CLI.
    model_config.params.sync_path = None
    model_config.params.synch_control = False
    # process overrides
    overrides = args.overrides
    if overrides:
        model_config = OmegaConf.merge(model_config, OmegaConf.from_dotlist(overrides))
        print('Merged model config')
        print(OmegaConf.to_yaml(model_config))
    if args.tag_cache:
        # The cache supplies exactly the RAM++ ids; never instantiate RAM++.
        model_config.params.preprocess_tag_config.params.enabled = False
    model: DiffEIC = instantiate_from_config(model_config)
    # Loading sequentially keeps peak host RAM low enough for Colab.  The
    # project checkpoint deliberately overwrites matching base-SD weights.
    _load_checkpoint(model, args.ckpt_sd, "Stable Diffusion")
    _load_checkpoint(model, args.ckpt_lc, "Diff-ICMH")
    # update preprocess model
    model.preprocess_model.update(force=True)
    model.freeze()
    model.to(args.device)
    anchor_prior_strength = args.anchor_prior_strength

    bpps = []
    total_bits = 0
    total_pixels = 0
    psnrs, ssims, lpips_values = [], [], []
    file_metrics = []
    
    assert os.path.isdir(args.input)
    manifest_rows = None
    selected_rows = None
    cached_tag_codec = CachedTagCodec(args.tag_vocabulary) if args.tag_cache else None
    tag_records = None
    if args.tag_cache:
        if not args.manifest:
            raise ValueError('--tag-cache requires --manifest')
        with open(args.tag_cache, 'r', encoding='utf-8') as stream:
            tag_records = {
                record['image_id']: record
                for record in (json.loads(line) for line in stream if line.strip())
            }
    if args.tag_vocabulary and not args.tag_cache:
        raise ValueError('--tag-vocabulary requires --tag-cache')
    if args.domain_metadata and not args.manifest:
        raise ValueError('--domain-metadata requires --manifest')
    if args.domain_metadata and not args.site_id:
        raise ValueError('--domain-metadata requires --site-id for the per-site decoder context')
    if args.habitat_map and not args.domain_metadata:
        raise ValueError('--habitat-map requires --domain-metadata')
    habitat_by_site = {}
    if args.habitat_map:
        with open(args.habitat_map, 'r', encoding='utf-8') as stream:
            habitat_by_site = json.load(stream)
        if not isinstance(habitat_by_site, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in habitat_by_site.items()
        ):
            raise ValueError('--habitat-map must be a JSON object of site_id -> habitat text')
        if args.site_id not in habitat_by_site:
            raise KeyError(f'No verified habitat label for {args.site_id}')
    if args.manifest:
        with open(args.manifest, 'r', encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        selected_rows = [
            row for row in rows
            if (args.split is None or row.get('split') == args.split)
            and (args.site_id is None or row.get('site_id') == args.site_id)
        ]
        if args.dev_list:
            with open(args.dev_list, 'r', encoding='utf-8') as stream:
                wanted = {line.strip() for line in stream if line.strip() and not line.startswith('#')}
            selected_rows = sorted(
                (row for row in selected_rows if row['image_id'] in wanted),
                key=lambda row: row['image_id'],
            )
        if not selected_rows:
            raise ValueError('manifest filters selected no images')
        if args.domain_metadata:
            sidecar = load_sidecar(args.illumination_sidecar)
            if not sidecar:
                raise FileNotFoundError(
                    '--domain-metadata needs light-source day/night labels: '
                    f'{args.illumination_sidecar} (tools/data/label_illumination.py)'
                )
            if apply_sidecar(selected_rows, sidecar) != len(selected_rows):
                raise KeyError('illumination sidecar does not cover every selected image')
        manifest_rows = {}
        for row in selected_rows:
            for key in (row.get('relative_path'), row.get('source_file_name')):
                if key:
                    manifest_rows[str(key).replace('\\', '/')] = row
    elif args.split or args.site_id or args.dev_list:
        raise ValueError('--split/--site-id/--dev-list require --manifest')

    os.makedirs(args.output, exist_ok=True)
    decode_log_path = os.path.join(args.output, 'decode_log.jsonl')
    logged_paths = set()
    if os.path.isfile(decode_log_path):
        with open(decode_log_path, 'r', encoding='utf-8') as stream:
            for line in stream:
                try:
                    logged_paths.add(json.loads(line)['relative_path'])
                except (json.JSONDecodeError, KeyError):
                    continue  # a line cut by a dead runtime is decoded again
    _write_run_info(args, model_config)

    # Intialize the LPIPS model
    lpips = LPIPS('alex').to(args.device)
    print(f"sampling {args.steps} steps using {args.sampler} sampler")
    if selected_rows is None:
        file_paths = list_image_files(args.input, follow_links=True)
    else:
        file_paths = [os.path.join(args.input, row['relative_path']) for row in selected_rows]
    if args.limit is not None:
        file_paths = file_paths[:args.limit]
    for file_path in file_paths:
        if not os.path.isfile(file_path):
            raise FileNotFoundError(file_path)
        relative_file_path = os.path.relpath(file_path, args.input).replace('\\', '/')
        save_path = os.path.join(args.output, relative_file_path)
        parent_path, stem, _ = get_file_name_parts(save_path)
        stream_parent_path = os.path.join(parent_path, 'data')
        save_path = os.path.join(parent_path, f"{stem}.png")
        stream_path = os.path.join(stream_parent_path, f"{stem}")
        if (
            args.skip_existing and relative_file_path in logged_paths
            and os.path.isfile(save_path) and os.path.isfile(stream_path)
        ):
            print(f"skip (already decoded): {relative_file_path}")
            continue

        # ``img`` is the evaluation reference: the original frame, or the smoke crop.
        img = Image.open(file_path).convert("RGB")
        if args.crop_size is not None:
            original_size = img.size
            img = center_crop_image(img, args.crop_size)
            print(
                f"Center crop {original_size[0]}x{original_size[1]} -> "
                f"{img.width}x{img.height}: {relative_file_path}"
            )
        coded = resize_for_processing(img, args.processing_long_side)
        x = pad(np.array(coded), scale=64)
        # Per-image seed: identical noise whether or not earlier images were skipped.
        torch.manual_seed((args.seed + zlib.crc32(relative_file_path.encode('utf-8'))) % (2 ** 31))

        os.makedirs(parent_path, exist_ok=True)
        os.makedirs(stream_parent_path, exist_ok=True)
        
        domain_rows = None
        if manifest_rows is not None:
            relative_key = os.path.relpath(file_path, args.input).replace('\\', '/')
            row = manifest_rows.get(relative_key)
            if row is None:
                row = manifest_rows.get(os.path.basename(file_path))
            if row is None:
                raise KeyError(f'No manifest row found for {relative_key}')
            if args.domain_metadata:
                domain_rows = [row]
        cached_tag_records = None
        if tag_records is not None:
            record = tag_records.get(row['image_id'])
            if record is None:
                raise KeyError(f"No cached RAM++ tags found for {row['image_id']}")
            cached_tag_records = [record]
        timings = {}
        preds, _ = process(
            model, [x], steps=args.steps, sampler=args.sampler,
            stream_path=stream_path,
            anchor_prior_strength=anchor_prior_strength,
            domain_rows=domain_rows,
            cached_tag_records=cached_tag_records,
            cached_tag_codec=cached_tag_codec,
            domain_site_id=args.site_id if args.domain_metadata else None,
            domain_habitat=habitat_by_site.get(args.site_id),
            timings=timings,
        )
        restore_started = time.perf_counter()
        pred_image = Image.fromarray(preds[0][:coded.height, :coded.width, :])
        pred_image = restore_original_size(pred_image, img.size)
        pred = np.asarray(pred_image)
        restore_seconds = time.perf_counter() - restore_started

        # The codec pads to a multiple of 64 internally.  BPP uses the
        # transmitted bytes over the pixels of the evaluation reference (the
        # original frame), never the padded or downscaled coding size.
        bitstream_bytes = os.path.getsize(stream_path)
        bpp = bitstream_bytes * 8.0 / (img.width * img.height)
        total_bits += bitstream_bytes * 8
        total_pixels += img.width * img.height

        # calculate bpp and save to list
        bpps.append(bpp)

        x_tmp = torch.tensor(np.asarray(img)).permute(2, 0, 1).unsqueeze(0).float().to(args.device) / 255
        xhat_tmp = torch.tensor(pred).permute(2, 0, 1).unsqueeze(0).float().to(args.device) / 255
        psnr = compute_psnr(x_tmp, xhat_tmp)
        ssim = compute_ssim(x_tmp, xhat_tmp)    # seems wrong results? better to use the [ pyiqa ] package
        lpips_value = lpips(x_tmp, xhat_tmp, normalize=True).mean()
        psnrs.append(psnr.item())
        ssims.append(ssim.item())
        lpips_values.append(lpips_value.item())
        file_metrics.append({
            'relative_path': relative_file_path,
            'bpp': bpp,
            'psnr': psnr.item(),
            'ssim': ssim.item(),
            'lpips': lpips_value.item()
        })
        
        pred_image.save(save_path, compress_level=1)  # lossless; level 1 is ~5x faster on 5 MP
        log_record = {
            'relative_path': relative_file_path,
            'image_id': row['image_id'] if manifest_rows is not None else None,
            'bitstream_bytes': bitstream_bytes,
            'bpp': bpp,
            'original_size': list(img.size),
            'coded_size': list(coded.size),
            'encode_seconds': timings.get('encode_seconds'),
            'decode_seconds': timings.get('decode_seconds'),
            'restore_seconds': restore_seconds,
        }
        with open(decode_log_path, 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(log_record, sort_keys=True) + '\n')
        print(
            f"save to {save_path}, bpp {bpp:.4f}, "
            f"encode {timings.get('encode_seconds', 0):.1f}s, decode {timings.get('decode_seconds', 0):.1f}s"
        )

    if not file_metrics:
        print('No new images decoded (all skipped).')
        return
    avg_bpp = total_bits / total_pixels
    avg_psnr = sum(psnrs) / len(psnrs)
    avg_ssim = sum(ssims) / len(ssims)
    avg_lpips = sum(lpips_values) / len(lpips_values)
    print(f'avg bpp: {avg_bpp:.4f}')
    print(f'avg psnr: {avg_psnr:.4f}')
    print(f'avg ssim: {avg_ssim:.4f}')
    print(f'avg lpips: {avg_lpips:.4f}')

    # write bpp to file with individual file bpp and average bpp
    bpp_file_path = os.path.join(args.output, 'bpp.txt')
    with open(bpp_file_path, 'w') as f:
        f.write(f'filename: bpp / psnr / ssim / lpips\n')
        for result in file_metrics:
            f.write(f"{result['relative_path']}: {result['bpp']:.4f} / {result['psnr']:.4f} / {result['ssim']:.4f} / {result['lpips']:.4f}\n")
        f.write(f'\navg: {avg_bpp:.4f} / {avg_psnr:.4f} / {avg_ssim:.4f} / {avg_lpips:.4f}\n')


if __name__ == "__main__":
    main()
