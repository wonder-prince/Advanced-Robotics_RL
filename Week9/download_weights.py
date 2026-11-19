"""Download and verify the original CenterMask-Lite MobileNetV2 1x checkpoint."""

import argparse
import hashlib
from pathlib import Path
import subprocess
from urllib.request import getproxies

FILENAME = "centermask-lite-M-v2-ms-bs32-1x.pth"
URL = "https://drive.usercontent.google.com/download?id=1okKyI9rYahM_OIKVRtTf7u_cqbLU4KSW&export=download&confirm=t"
SHA256 = "b3b2490756cb8c3c866c0a39ec44bf40c28f1c39a6d22c11995be2e4be54eac2"
SIZE = 29604748


def verify(path):
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    if path.stat().st_size != SIZE or digest != SHA256:
        raise ValueError(f"Checkpoint size/SHA256 mismatch: {path}; expected {SIZE} bytes and {SHA256}")
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "weights" / FILENAME)
    parser.add_argument("--proxy", default=getproxies().get("https"), help="optional HTTP proxy; defaults to system HTTPS proxy")
    args = parser.parse_args()
    if args.output.exists():
        digest = verify(args.output)
        print(f"Already verified: {args.output}\nSHA256: {digest}")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".part")
    command = ["curl", "--fail", "--location", "--retry", "2", "--connect-timeout", "15", "--max-time", "300"]
    if args.proxy:
        command.extend(["--proxy", args.proxy])
    command.extend(["--output", str(partial), URL])
    print(f"Downloading official checkpoint: {FILENAME}", flush=True)
    try:
        subprocess.run(command, check=True)
        digest = verify(partial)
        partial.replace(args.output)
    finally:
        partial.unlink(missing_ok=True)
    print(f"Verified: {args.output}\nBytes: {SIZE}\nSHA256: {digest}")


if __name__ == "__main__":
    main()
