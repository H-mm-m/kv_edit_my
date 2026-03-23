import argparse
import json
import os
from typing import List, Dict

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


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
    构造每张生成图对应的 (id, image_path, prompt) 记录。
    """
    records: List[dict] = []
    missing = 0

    for sample_id, rec in mapping.items():
        if not isinstance(rec, dict):
            continue

        rel_path = rec.get("image_path", "")
        if not rel_path:
            continue

        # batch_kv_edit_gpu.py 中的逻辑：
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


def compute_clip_sims(
    records: List[dict],
    model_name: str,
    device: str,
    batch_size: int,
) -> List[dict]:
    """
    使用 HuggingFace 的 CLIP（完全离线，本地 cache）计算 (image, text) 的 cosine 相似度。
    需要你已经离线下载好 `model_name` 对应的权重到 ~/.cache/huggingface/hub。
    """
    if not records:
        return []

    device_t = torch.device(device)

    print(f"[信息] 从本地缓存加载 CLIP 模型: {model_name} 到 {device_t}")
    # local_files_only=True 确保不发任何网络请求，只从本地 cache 读取
    model = CLIPModel.from_pretrained(model_name, local_files_only=True).to(device_t)
    processor = CLIPProcessor.from_pretrained(model_name, local_files_only=True)

    model.eval()

    all_results: List[dict] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(records), batch_size), desc="Computing CLIP Sim"):
            batch = records[i : i + batch_size]

            texts = [r["prompt"] for r in batch]
            images = [Image.open(r["image_path"]).convert("RGB") for r in batch]

            inputs = processor(
                text=texts,
                images=images,
                return_tensors="pt",
                padding=True,
            ).to(device_t)

            outputs = model(**inputs)
            img_embeds = outputs.image_embeds  # (B, D)
            txt_embeds = outputs.text_embeds   # (B, D)

            img_embeds = img_embeds / img_embeds.norm(p=2, dim=-1, keepdim=True)
            txt_embeds = txt_embeds / txt_embeds.norm(p=2, dim=-1, keepdim=True)

            sims = (img_embeds * txt_embeds).sum(dim=-1)  # cosine similarity

            for r, sim in zip(batch, sims):
                result = dict(r)
                result["clip_sim"] = float(sim.item())
                all_results.append(result)

    return all_results


def save_results_csv(results: List[dict], output_csv: str) -> None:
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    import csv

    fieldnames = ["id", "image_rel", "image_path", "prompt", "clip_sim"]
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"[信息] 已保存结果到: {output_csv}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对 KV-Edit 生成图计算 CLIP 图文相似度（使用本地缓存的 CLIP，完全离线）。"
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
        help="使用哪个字段作为文本输入（默认: editing_prompt）",
    )
    parser.add_argument(
        "--clip_model",
        type=str,
        default="openai/clip-vit-large-patch14",
        help="CLIP 模型名称（transformers 格式，需要已缓存到本地）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="计算 CLIP 的设备，例如 cuda:0 或 cpu",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="CLIP 前向计算的 batch size",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="clip_sims.csv",
        help="保存结果的 CSV 路径",
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
        print("[错误] 没有找到任何有效的 (图片, 文本) 记录，请检查 images_root 和 pie_mapping。")
        return

    results = compute_clip_sims(
        records=records,
        model_name=args.clip_model,
        device=args.device,
        batch_size=args.batch_size,
    )

    save_results_csv(results, args.output_csv)


if __name__ == "__main__":
    main()

