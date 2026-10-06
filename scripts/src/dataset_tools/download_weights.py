import argparse
import urllib.request
from pathlib import Path

WEIGHTS_URL = (
    "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth"
)


def download(out: Path) -> bool:
    """Download the YOLOX-Tiny checkpoint unless `out` exists. Returns True if downloaded."""
    if out.exists():
        return False
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".part")
    urllib.request.urlretrieve(WEIGHTS_URL, tmp)
    tmp.replace(out)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(prog="download-weights")
    parser.add_argument("--out", type=Path, default=Path("pretrained/yolox_tiny.pth"))
    args = parser.parse_args()
    print(f"downloaded {args.out}" if download(args.out) else f"{args.out} already exists")


if __name__ == "__main__":
    main()
