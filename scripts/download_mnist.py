"""Download MNIST into $DATA_DIR/mnist/raw, verify checksums, write SHA256SUMS, make read-only."""

import stat
import sys
import urllib.request

from digits.data import MNIST_FILES, file_hash, raw_dir, verify_checksums

# Mirrors used by torchvision (yann.lecun.com is often unreachable)
MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "http://yann.lecun.com/exdb/mnist/",
]


def download(name: str, dest) -> None:
    for mirror in MIRRORS:
        try:
            print(f"downloading {mirror}{name}")
            urllib.request.urlretrieve(mirror + name, dest)
            return
        except OSError as e:
            print(f"  failed: {e}")
    raise RuntimeError(f"could not download {name}")


def make_read_only(raw) -> None:
    """chmod -R a-w"""
    for path in [*raw.iterdir(), raw]:
        path.chmod(path.stat().st_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def main() -> None:
    raw = raw_dir()
    if (raw / "SHA256SUMS").exists():
        verify_checksums(raw)
        print(f"{raw} already complete, checksums OK")
        return
    raw.mkdir(parents=True, exist_ok=True)

    for name, md5 in MNIST_FILES.items():
        dest = raw / name
        if not dest.exists():
            download(name, dest)
        if file_hash(dest, "md5") != md5:
            dest.unlink()
            sys.exit(f"MD5 mismatch for {name}, file removed")
        print(f"{name}: MD5 OK")

    lines = [f"{file_hash(raw / name, 'sha256')}  {name}" for name in MNIST_FILES]
    (raw / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    verify_checksums(raw)
    make_read_only(raw)
    print(f"wrote SHA256SUMS, {raw} is now read-only")


if __name__ == "__main__":
    main()
