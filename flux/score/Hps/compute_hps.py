import argparse
import json
import os
from typing import Dict, List

from PIL import Image
from tqdm import tqdm
import torch

try:
    #from hpsv2 import HPSv2
    import hpsv2
except ImportError as e:
    raise ImportError(
        "未找到 hpsv2 包，请先在当前环境安装：\n\n"
        "    pip install hpsv2\n"
        "\n"
        "然后重新运行本脚本。"
    ) from e


def load_mapping(pie_mapping_path: str) -> Dict[str, dict]:
    with open(pie_mapping_path, "r", encoding="utf-8") as f:
        mapping = json.load(f)
    if not isinstance(mapping, dict):
        raise ValueError("PIE mapping_file.json 必须是以 sample_id 为 key 的字典。")
    return mapping


def build_image_records(
    images_root: str,
    mapping: Dict[str, dict],
    prompt_field: str,
) -> List[dict]:
    """
    根据 PIE 的 mapping_file.json 和 batch_kv_edit_gpu.py 的保存规则，
    构造每张生成图对应的记录：
      - id
      - image_rel
      - image_path
      - prompt
    """
    records: List[dict] = []
    missing = 0

    for sample_id, rec in mapping.items():
        if not isinstance(rec, dict):
            continue

        rel_path = rec.get("image_path", "")
        if not rel_path:
            continue

        # 对应 batch_kv_edit_gpu.py 中的保存逻辑：
        #   rel = image_rel.lstrip("/").replace("\\", "/")
        #   out_path = os.path.join(out_dir, rel)
        #   out_path = os.path.splitext(out_path)[0] + ".jpg"
        rel_norm = rel_path.lstrip("/").replace("\\", "/")
        img_path = os.path.join(images_root, rel_norm)
        img_path = os.path.splitext(img_path)[0] + ".jpg"

        if not os.path.exists(img_path):
            missing += 1
            continue

        prompt = rec.get(prompt_field, "")
        if prompt is None:
            prompt = ""

        records.append(
            {
                "id": str(sample_id),
                "image_rel": rel_norm,
                "image_path": img_path,
                "prompt": prompt,
            }
        )

    if missing > 0:
        print(f"[警告] 有 {missing} 条记录在 '{images_root}' 下找不到对应生成图，已跳过。")

    print(f"[信息] 有效样本数: {len(records)}")
    return records


def compute_hps_scores(
    records: List[dict],
    device: str = "cuda:0",
    batch_size: int = 16,
    hps_version: str = "v2.1",
) -> List[dict]:
    """
    使用 HPSv2 模型对每张图像打分。

    参数:
        records: build_image_records 返回的记录列表
        device:  例如 "cuda:0" 或 "cpu"
        batch_size: 推理 batch size
        hps_version: HPS 版本，通常为 "v2.1" 或 "v2.0"
    """
    if not records:
        return []

    device_t = torch.device(device)

    print(f"[信息] 初始化 HPSv2 模型 (version={hps_version}) 到 {device_t}")
    # 说明：pip 的 hpsv2 暴露的是 hpsv2.score(imgs_path, prompt, hps_version=...)
    # 但该函数每次都会 load checkpoint，逐张调用会很慢。
    # 这里直接复用其内部 open_clip 组件，做到“只加载一次 checkpoint，然后批量算分”。
    from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
    from hpsv2.utils import hps_version_map
    import huggingface_hub

    model, _, preprocess_val = create_model_and_transforms(
        "ViT-H-14",
        "laion2B-s32B-b79K",
        precision="amp" if device_t.type == "cuda" else "fp32",
        device=str(device_t),
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )

    cp = huggingface_hub.hf_hub_download("xswu/HPSv2", hps_version_map[hps_version])
    checkpoint = torch.load(cp, map_location=str(device_t))
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device_t).eval()

    tokenizer = get_tokenizer("ViT-H-14")

    all_results: List[dict] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(records), batch_size), desc="Computing HPS"):
            batch = records[i : i + batch_size]
            images_pil = [Image.open(r["image_path"]).convert("RGB") for r in batch]
            prompts = [r.get("prompt", "") for r in batch]

            image_tensor = torch.cat(
                [preprocess_val(img).unsqueeze(0) for img in images_pil], dim=0
            ).to(device=device_t, non_blocking=True)
            text_tokens = tokenizer(prompts).to(device=device_t, non_blocking=True)

            if device_t.type == "cuda":
                autocast_ctx = torch.cuda.amp.autocast()
            else:
                from contextlib import nullcontext

                autocast_ctx = nullcontext()

            with autocast_ctx:
                outputs = model(image_tensor, text_tokens)
                image_features, text_features = outputs["image_features"], outputs["text_features"]
                logits_per_image = image_features @ text_features.T
                scores = torch.diagonal(logits_per_image).float().detach().cpu().tolist()

            for r, s in zip(batch, scores):
                out = dict(r)
                out["hps"] = float(s)
                all_results.append(out)

    return all_results


def save_results_csv(results: List[dict], output_csv: str) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    import csv

    fieldnames = ["id", "image_rel", "image_path", "prompt", "hps"]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"[信息] 已保存 HPS 结果到: {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对 KV-Edit 生成图计算 HPSv2 图像质量分数。"
    )
    parser.add_argument(
        "--images_root",
        type=str,
        required=True,
        help="生成图片根目录，例如 /home/hemiao/KV-Edit/outputs_pie_0_8",
    )
    parser.add_argument(
        "--pie_mapping",
        type=str,
        required=True,
        help="PIE-Bench 的 mapping_file.json 路径",
    )
    parser.add_argument(
        "--prompt_field",
        type=str,
        default="editing_prompt",
        choices=["editing_prompt", "original_prompt", "editing_instruction"],
        help="使用哪个字段作为 prompt（默认: editing_prompt）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="计算 HPS 的设备，例如 cuda:0 或 cpu",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="HPS 前向计算的 batch size",
    )
    parser.add_argument(
        "--hps_version",
        type=str,
        default="v2.1",
        help="HPS 版本号，例如 v2.1 或 v2.0（默认: v2.1）",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="hps_scores.csv",
        help="保存结果的 CSV 路径（默认: hps_scores.csv）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mapping = load_mapping(args.pie_mapping)
    records = build_image_records(
        images_root=args.images_root,
        mapping=mapping,
        prompt_field=args.prompt_field,
    )

    if not records:
        print("[错误] 没有找到任何有效的图像记录，请检查 images_root 和 pie_mapping。")
        return

    results = compute_hps_scores(
        records=records,
        device=args.device,
        batch_size=args.batch_size,
        hps_version=args.hps_version,
    )

    save_results_csv(results, args.output_csv)


if __name__ == "__main__":
    main()

