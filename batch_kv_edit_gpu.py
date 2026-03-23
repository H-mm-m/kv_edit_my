import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange
from PIL import Image

from flux.sampling import prepare
from flux.util import configs, load_ae, load_clip, load_t5
from models.kv_edit import Flux_kv_edit


@dataclass
class SamplingOptions:
    source_prompt: str = ""
    target_prompt: str = ""
    width: int = 1366
    height: int = 768
    inversion_num_steps: int = 2
    denoise_num_steps: int = 10
    skip_step: int = 2
    inversion_guidance: float = 1.0
    denoise_guidance: float = 5.5
    #随机种子，控制生成画面的随机性
    seed: int = 42
    #是否重新初始化，控制生成画面的随机性
    re_init: bool = False
    #是否使用注意力掩码，控制生成画面的随机性
    attn_mask: bool = False
    #掩码控制尺度
    attn_scale: float = 1.0


def _resize_to_fit(image_array: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """Resize HWC uint8 image to fit within (max_width, max_height) while keeping aspect ratio."""
    if max_width <= 0 or max_height <= 0:
        return image_array
    pil = Image.fromarray(image_array)
    w, h = pil.size
    scale = min(max_width / w, max_height / h)
    if scale >= 1:
        return image_array
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    pil = pil.resize((new_w, new_h), resample=Image.BICUBIC)
    return np.array(pil)


def _crop_to_multiple_of_16(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    h = h - (h % 16)
    w = w - (w % 16)
    return img[:h, :w, ...]


def _load_rgb(path: str) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _load_mask_alpha(path: str) -> np.ndarray:
    """
    Load a mask from file and return alpha mask as uint8 in [0,255], shape (H,W).
    Supports:
    - RGBA/LA: uses alpha channel
    - L/RGB: uses luminance (after converting to L)
    """
    img = Image.open(path)
    if img.mode in ("RGBA", "LA"):
        alpha = np.array(img.split()[-1], dtype=np.uint8)
        return alpha
    return np.array(img.convert("L"), dtype=np.uint8)

def _rle_pairs_to_mask_alpha(rle_pairs: list[int], height: int, width: int) -> np.ndarray:
    """
    PIE-Bench `mapping_file.json` stores mask as RLE pairs: [start0, len0, start1, len1, ...]
    over a flattened (row-major) image of size H*W.
    Returns uint8 mask in [0,255] with shape (H,W).
    """
    if len(rle_pairs) % 2 != 0:
        raise ValueError(f"RLE length must be even, got {len(rle_pairs)}")
    flat = np.zeros((height * width,), dtype=np.uint8)
    for start, length in zip(rle_pairs[0::2], rle_pairs[1::2]):
        if length <= 0:
            continue
        end = start + length
        if start < 0 or end > flat.size:
            raise ValueError(f"RLE segment out of bounds: start={start}, len={length}, hw={flat.size}")
        flat[start:end] = 255
    return flat.reshape((height, width))


def _make_brush_canvas(background_rgb: np.ndarray, mask_alpha: np.ndarray) -> dict:
    """
    Create a dict compatible with gradio ImageEditor output used by gradio_kv_edit_gpu.py:
    - background: HWC RGBA uint8
    - layers[0]: HWC RGBA uint8, alpha is the mask
    """
    h, w = background_rgb.shape[:2]
    if mask_alpha.shape[0] != h or mask_alpha.shape[1] != w:
        mask_alpha = np.array(
            Image.fromarray(mask_alpha).resize((w, h), resample=Image.NEAREST), dtype=np.uint8
        )

    background_rgba = np.concatenate(
        [background_rgb, np.full((h, w, 1), 255, dtype=np.uint8)], axis=-1
    )
    layer = np.zeros((h, w, 4), dtype=np.uint8)
    layer[..., 0] = 255  # red brush color (not used in math; alpha is)
    layer[..., 3] = mask_alpha
    return {"background": background_rgba, "layers": [layer]}


class KVEditBatchRunner:
    """
    A non-Gradio batch runner that reuses the same device split as gradio_kv_edit_gpu.py:
    - device0: Flux_kv_edit (main diffusion model)
    - device1: T5 / CLIP / AE (text encoders + autoencoder)
    """

    def __init__(self, name: str, device0: torch.device, device1: torch.device):
        self.name = name
        self.device0 = device0
        self.device1 = device1
        self.is_schnell = name == "flux-schnell"

        self.t5 = load_t5(device1, max_length=256 if self.is_schnell else 512).eval()
        self.clip = load_clip(device1).eval()
        #这里创建了Flux_kv_edit对象，并评估模式
        self.model = Flux_kv_edit(device0, name=name).eval()

        self.ae = load_ae(name, device=device1).eval()

    @torch.inference_mode()
    def encode(self, rgb_hwc_uint8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(rgb_hwc_uint8).permute(2, 0, 1).float() / 127.5 - 1.0
        x = x.unsqueeze(0).to(self.device1)
        self.ae.encoder.to(self.device1)
        z = self.ae.encode(x).to(torch.bfloat16)
        return z

    @torch.inference_mode()
    def decode(self, latent_bchw: torch.Tensor) -> np.ndarray:
        # latent_bchw expected on device0; decode on device1 like gradio_kv_edit_gpu.py
        self.ae.decoder.to(self.device1)
        with torch.autocast(device_type=self.device1.type, dtype=torch.bfloat16):
            x = self.ae.decode(latent_bchw.to(self.device1))
        x = x.clamp(-1, 1).float().cpu()
        x = rearrange(x[0], "c h w -> h w c")
        img = (127.5 * (x + 1.0)).byte().numpy()
        return img

    @torch.inference_mode()
    def run_one(
        self,
        *,
        image_rgb: np.ndarray,
        mask_alpha: np.ndarray,
        opts: SamplingOptions,
        max_width: int,
        max_height: int,
    ) -> np.ndarray:
        # mimic gradio resize/crop behavior
        #1.把图片resize到max_width，max_height,包括mask也resize
        image_rgb = _resize_to_fit(image_rgb, max_width=max_width, max_height=max_height)
        image_rgb = _crop_to_multiple_of_16(image_rgb)
        mask_alpha = _resize_to_fit(mask_alpha, max_width=max_width, max_height=max_height)
        if mask_alpha.ndim == 3:
            mask_alpha = mask_alpha[..., 0]
        mask_alpha = _crop_to_multiple_of_16(mask_alpha[..., None])[..., 0]

        h, w = image_rgb.shape[:2]
        opts.height = h
        opts.width = w

        # 2.设置Seed
        torch.manual_seed(int(opts.seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(opts.seed))

        # 3.构造准备 mask tensor (always needed for denoise blending)
        mask01 = (mask_alpha.astype(np.float32) / 255.0) > 0.5
        mask = torch.from_numpy(mask01.astype(np.int32)).unsqueeze(0).unsqueeze(0)
        mask = mask.to(self.device0).to(torch.bfloat16)

        # 4.Encode image -> latent on device1, then move to device0 for diffusion model
        init_latent = self.encode(image_rgb).to(self.device0)

        # 5.Inverse: only pass mask when attn_mask=True (to match original code behavior)
        inv_mask = mask if opts.attn_mask else None
        inp = prepare(self.t5, self.clip, init_latent, prompt=opts.source_prompt)
        #根据这个函数调用模型：self.model = Flux_kv_edit(device0, name=name).eval()
        z0, zt, info = self.model.inverse(inp, inv_mask, opts)

        # Edit / denoise
        inp_target = prepare(self.t5, self.clip, init_latent, prompt=opts.target_prompt)
        # #根据这个函数调用模型：self.model = Flux_kv_edit(device0, name=name).eval()
        out_latent = self.model.denoise(z0.clone(), zt, inp_target, mask, opts, info)

        # free feature cache early
        del info

        return self.decode(out_latent)


def _default_opts(job: dict) -> SamplingOptions:
    return SamplingOptions(
        source_prompt=job.get("source_prompt", ""),
        target_prompt=job.get("target_prompt", ""),
        inversion_num_steps=int(job.get("inversion_num_steps", 0)),
        denoise_num_steps=int(job.get("denoise_num_steps", 28)),
        skip_step=int(job.get("skip_step", 0)),
        inversion_guidance=float(job.get("inversion_guidance", 1.5)),
        denoise_guidance=float(job.get("denoise_guidance", 5.5)),
        seed=int(job.get("seed", 42)),
        re_init=bool(job.get("re_init", False)),
        attn_mask=bool(job.get("attn_mask", False)),
        attn_scale=float(job.get("attn_scale", 1.0)),
    )

def _parse_int_list_csv(s: str) -> list[int]:
    s = (s or "").strip()
    if not s:
        return []
    out = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def main():
    parser = argparse.ArgumentParser(description="KV-Edit batch inference (GPU split like gradio_kv_edit_gpu.py)")
    parser.add_argument("--name", type=str, default="flux-dev", choices=list(configs.keys()), help="Model name")
    # Mode A: jobs.jsonl
    parser.add_argument("--jobs", type=str, default="", help="Path to jobs.jsonl (line-delimited dicts)")
    # Mode B: PIE-Bench mapping
    parser.add_argument("--pie_root", type=str, default="", help="PIE-Bench-v1 root (contains annotation_images/)")
    parser.add_argument("--pie_mapping", type=str, default="", help="Path to PIE-Bench mapping_file.json")
    parser.add_argument(
        "--pie_type_ids",
        type=str,
        default="0,1,2,3,4,5,6,7,8",
        help="Comma-separated editing_type_id to run (default: 0-8)",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for results")
    parser.add_argument("--device0", type=str, default="cuda:0", help="Device for diffusion model (Flux_kv_edit)")
    parser.add_argument("--device1", type=str, default="cuda:1", help="Device for text encoders + AE")
    parser.add_argument("--max_width", type=int, default=1366, help="Max width for automatic resize (0 to disable)")
    parser.add_argument("--max_height", type=int, default=768, help="Max height for automatic resize (0 to disable)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device0 = torch.device(args.device0)
    device1 = torch.device(args.device1)
    #创建runner对象，用于运行KV-Edit,进入KVEditBatchRunner类，初始化函数__init__,加载所有需要用到的模型
    runner = KVEditBatchRunner(args.name, device0=device0, device1=device1)

    jobs: list[dict] = []
    if args.pie_mapping:
        # PIE-Bench batch: build jobs from mapping_file.json
        pie_root = args.pie_root.rstrip("/")
        if not pie_root:
            raise ValueError("--pie_root is required when using --pie_mapping")
        type_ids = set(_parse_int_list_csv(args.pie_type_ids))
        with open(args.pie_mapping, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        if not isinstance(mapping, dict):
            raise ValueError("PIE mapping_file.json must be a dict keyed by sample id")
        for sample_id, rec in mapping.items():
            if not isinstance(rec, dict):
                continue
            et = rec.get("editing_type_id", None)
            if et is None:
                continue
            try:
                et_int = int(et)
            except Exception:
                continue
            if type_ids and et_int not in type_ids:
                continue
            rel_path = rec.get("image_path", "")
            if not rel_path:
                continue
            # Only categories 0-8 are requested; they also correspond to folders starting with "0_"..."8_"
            if rel_path[:1].isdigit():
                if int(rel_path[:1]) not in type_ids:
                    continue

            image_path = os.path.join(pie_root, "annotation_images", rel_path)
            jobs.append(
                {
                    "id": str(sample_id),
                    "image": image_path,
                    "image_rel": rel_path,
                    # prompts
                    "source_prompt": rec.get("original_prompt", ""),
                    "target_prompt": rec.get("editing_prompt", ""),
                    # mask RLE pairs
                    "mask_rle": rec.get("mask", []),
                    # passthrough opts
                    "editing_instruction": rec.get("editing_instruction", ""),
                    "editing_type_id": et_int,
                }
            )
    else:
        if not args.jobs:
            raise ValueError("Provide either --jobs or --pie_mapping/--pie_root")
        # Read jobs.jsonl
        with open(args.jobs, "r", encoding="utf-8") as f:
            jobs = [json.loads(line) for line in f if line.strip()]

    for idx, job in enumerate(jobs):
        image_path = job["image"]
        out_path = job.get("out", "")
        if not out_path:
            # preserve PIE relative structure if available
            if args.pie_mapping and "image_rel" in job:
                # Mirror annotation_images/ relative path under out_dir
                rel = job["image_rel"].lstrip("/").replace("\\", "/")
                out_path = os.path.join(args.out_dir, rel)
                out_path = os.path.splitext(out_path)[0] + ".jpg"
            else:
                out_path = os.path.join(args.out_dir, f"{idx:05d}.jpg")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        t0 = time.perf_counter()
        image_rgb = _load_rgb(image_path)
        if "mask_rle" in job:
            # decode PIE RLE to alpha mask
            w, h = Image.open(image_path).size
            mask_alpha = _rle_pairs_to_mask_alpha(job["mask_rle"], height=h, width=w)
        else:
            mask_path = job["mask"]
            mask_alpha = _load_mask_alpha(mask_path)
        opts = _default_opts(job)
        #对每一张图片，都赋值给对象runner，运行run_one函数，输入图片，mask，opts，max_width，max_height
        out_img = runner.run_one(
            image_rgb=image_rgb,
            mask_alpha=mask_alpha,
            opts=opts,
            max_width=args.max_width,
            max_height=args.max_height,
        )
        Image.fromarray(out_img).save(out_path, quality=95, subsampling=0)
        if args.pie_mapping and "mask_rle" in job:
            mask_out_path = os.path.splitext(out_path)[0] + "_mask.png"
            Image.fromarray(mask_alpha).save(mask_out_path)

        # Try to keep VRAM stable across a long batch
        if device0.type == "cuda":
            torch.cuda.synchronize(device0)
            torch.cuda.empty_cache()
        if device1.type == "cuda":
            torch.cuda.synchronize(device1)
            torch.cuda.empty_cache()

        t1 = time.perf_counter()
        print(f"[{idx+1}/{len(jobs)}] saved {out_path} ({t1 - t0:.1f}s)")


if __name__ == "__main__":
    main()

