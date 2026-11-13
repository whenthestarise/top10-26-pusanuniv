from pathlib import Path
from importlib.metadata import version

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open


def main():
    for package in (
        "torch",
        "transformers",
        "safetensors",
        "bitsandbytes",
    ):
        print(f"{package}: {version(package)}", flush=True)

    # 이미 다운로드한 캐시만 사용합니다.
    # 인터넷 다운로드나 모델 파일 변경은 하지 않습니다.
    model_path = Path(
        snapshot_download(
            repo_id="Qwen/Qwen3-1.7B",
            local_files_only=True,
        )
    )

    files = sorted(model_path.glob("*.safetensors"))

    if not files:
        raise FileNotFoundError(
            f"캐시에서 safetensors 파일을 찾지 못했습니다: {model_path}"
        )

    print(f"가중치 파일 수: {len(files)}", flush=True)
    print("CPU에서 가중치 읽기 시작", flush=True)

    total = 0

    for path in files:
        print(f"\n읽는 파일: {path.name}", flush=True)

        count = 0

        with safe_open(
            str(path),
            framework="pt",
            device="cpu",
        ) as file:
            for key in file.keys():
                tensor = file.get_tensor(key)

                # 실제 메모리 읽기도 확인합니다.
                if tensor.numel():
                    flat = tensor.reshape(-1)
                    _ = flat[0].item()
                    _ = flat[-1].item()
                    del flat

                del tensor
                count += 1

        total += count
        print(f"읽기 완료: {count}개 텐서", flush=True)

    print(f"\nCPU 가중치 읽기 통과: 총 {total}개 텐서", flush=True)


if __name__ == "__main__":
    main()