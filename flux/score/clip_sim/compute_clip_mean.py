import argparse
import csv
from pathlib import Path


def compute_mean_clip_sim(csv_path: str, column: str = "clip_sim") -> float:
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV 文件不存在: {path}")

    total = 0.0
    count = 0

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if column not in reader.fieldnames:
            raise ValueError(f"CSV 中找不到列 '{column}'，实际列为: {reader.fieldnames}")
        for row in reader:
            val_str = row.get(column, "").strip()
            if not val_str:
                continue
            try:
                val = float(val_str)
            except ValueError:
                continue
            total += val
            count += 1

    if count == 0:
        raise ValueError(f"列 '{column}' 中没有有效的数值。")

    return total / count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取 clip_sims.csv 中的 clip_sim 列并计算平均值。"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="clip_sims.csv",
        help="输入 CSV 文件路径（默认: clip_sims.csv）",
    )
    parser.add_argument(
        "--column",
        type=str,
        default="clip_sim",
        help="要计算平均值的列名（默认: clip_sim）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mean_val = compute_mean_clip_sim(args.csv, args.column)
    print(f"Mean of '{args.column}' in '{args.csv}': {mean_val}")


if __name__ == "__main__":
    main()

