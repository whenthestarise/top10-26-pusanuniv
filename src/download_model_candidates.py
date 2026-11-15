"""등록한 모델의 고정 버전만 내려받습니다. 기업 자료는 가져오지 않습니다."""

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", default="all")
    parser.add_argument("--hf-home", type=Path)
    args = parser.parse_args()
    if args.hf_home:
        os.environ["HF_HOME"] = str(args.hf_home.expanduser().resolve())
    from huggingface_hub import snapshot_download
    registry = json.loads((ROOT / "configs/model_candidates.json").read_text())
    candidates = [r for r in registry["candidates"] if args.candidate in {"all", r["candidate_id"]}]
    if not candidates:
        parser.error("등록된 모델 후보가 아닙니다.")
    for candidate in candidates:
        snapshot_download(candidate["model_id"], revision=candidate["revision"],
            allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "README.md"])
        print("고정 버전 모델 저장 완료:", candidate["candidate_id"])


if __name__ == "__main__":
    main()
