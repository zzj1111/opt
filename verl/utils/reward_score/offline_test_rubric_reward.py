#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import json

# 这里假设你的文件名就叫 CudaForge.py
# 如果你的文件名不同，把 CudaForge 改成你的文件名（不带 .py）
import CudaForge as cf


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser("Offline test for rubric reward integration")
    parser.add_argument("--data_source", type=str, default="CudaForgeImprovement",
                        choices=["CudaForge", "CudaForgeImprovement"])
    parser.add_argument("--improvement_txt", type=str, required=False,
                        help="Path to improvement txt file (only for CudaForgeImprovement)")
    parser.add_argument("--reference_py", type=str, required=True,
                        help="Path to reference code .py (used as extra_info['answer'])")
    parser.add_argument("--candidate_py", type=str, required=True,
                        help="Path to candidate code .py (ModelNew), used as solution_str")
    parser.add_argument("--save_trimmed", type=str, default="",
                        help="Optional path to save trimmed txt result")
    parser.add_argument("--max_trim_chars", type=int, default=12000,
                        help="Max chars for safe_trim_improvement_txt")

    # Optional: skip bench to test rubric+aggregation faster
    parser.add_argument("--skip_bench", action="store_true",
                        help="Skip bench() and use manual correctness/speedup")
    parser.add_argument("--manual_correctness", type=int, default=1,
                        help="Manual correctness when --skip_bench is enabled (0/1)")
    parser.add_argument("--manual_speedup", type=float, default=1.0,
                        help="Manual speedup when --skip_bench is enabled")

    args = parser.parse_args()

    data_source = args.data_source

    reference_code = read_text(Path(args.reference_py))
    candidate_code = read_text(Path(args.candidate_py))

    # solution_str 在你的 compute_score 里会走 _extract_python_code，
    # 所以我们这里直接构造成 ```python ... ``` 也行
    solution_str = f"```python\n{candidate_code}\n```"

    extra_info = {"answer": reference_code}

    if data_source == "CudaForgeImprovement":
        if not args.improvement_txt:
            raise ValueError("--improvement_txt is required for CudaForgeImprovement")
        improvement_txt_raw = read_text(Path(args.improvement_txt))
        extra_info["question"] = improvement_txt_raw

        # 1) 打印裁剪结果
        trimmed = cf.safe_trim_improvement_txt(improvement_txt_raw, max_chars=args.max_trim_chars)

        print("\n" + "=" * 80)
        print("[Trimmed improvement TXT]")
        print("=" * 80)
        print(trimmed)
        print("=" * 80 + "\n")

        if args.save_trimmed:
            Path(args.save_trimmed).write_text(trimmed, encoding="utf-8")
            print(f"[Saved trimmed txt] -> {args.save_trimmed}\n")

    # 2) 计算 reward
    if args.skip_bench:
        # 跳过 bench：手工指定 correctness/speedup，然后只跑 rubric + 聚合
        # 这里直接复用 CudaForge.py 内部函数链条（确保跟训练一致）

        correctness = int(args.manual_correctness)
        speedup = float(args.manual_speedup)

        if correctness != 1:
            print("[Skip bench] manual correctness=0 => reward=0")
            print("FINAL_REWARD=0.0")
            return

        rubric_obj = cf._run_rubric_judge(
            data_source=data_source,
            reference_code=reference_code,
            candidate_code=cf._extract_python_code(solution_str),
            extra_info=extra_info,
        )

        shaped_reward, dbg = cf._compute_final_reward(
            correctness=correctness,
            speedup=speedup,
            rubric_obj=rubric_obj,
            data_source=data_source,
            speedup_clip_max=5.0,
        )
        final_reward = min(float(shaped_reward), 3.0)

        print("\n" + "=" * 80)
        print("[OFFLINE RESULT: SKIP BENCH]")
        print("=" * 80)
        print(f"manual correctness={correctness}")
        print(f"manual speedup={speedup}")
        print(f"rubric_obj={json.dumps(rubric_obj, ensure_ascii=False)}")
        print(f"dbg={json.dumps(dbg, ensure_ascii=False)}")
        print(f"FINAL_REWARD={final_reward}")
        print("=" * 80 + "\n")

    else:
        # 正常路径：走 compute_score（会执行 bench + rubric + 聚合）
        reward = cf.compute_score(data_source, solution_str, ground_truth=None, extra_info=extra_info)

        print("\n" + "=" * 80)
        print("[OFFLINE RESULT: WITH BENCH]")
        print("=" * 80)
        print(f"FINAL_REWARD={reward}")
        print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
