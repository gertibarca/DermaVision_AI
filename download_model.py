from __future__ import annotations

import os
import sys
import urllib.request

from dermavision.config import MODEL_PATH, MODEL_URL

MINIMUM_MODEL_BYTES = 40_000_000


def progress(blocks: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    percent = min(100, blocks * block_size * 100 / total_size)
    print(f"\rDuke shkarkuar modelin: {percent:5.1f}%", end="", flush=True)


def main() -> int:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size >= MINIMUM_MODEL_BYTES:
        print(f"Modeli ekziston: {MODEL_PATH}")
        return 0

    temporary = MODEL_PATH.with_suffix(".download")
    print("Po shkarkohet ResNet18 i trajnuar në HAM10000 (~43 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, temporary, progress)
        print()
        if temporary.stat().st_size < MINIMUM_MODEL_BYTES:
            raise RuntimeError("Skedari i modelit është i paplotë.")
        os.replace(temporary, MODEL_PATH)
    except (OSError, RuntimeError) as error:
        temporary.unlink(missing_ok=True)
        print(f"Gabim: {error}", file=sys.stderr)
        return 1

    print(f"Modeli u ruajt: {MODEL_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
