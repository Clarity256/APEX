"""Download a dated Starlink TLE snapshot from CelesTrak."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "tle"

CELESTRAK_URLS = {
    "gp": "https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle",
    "supplemental": (
        "https://celestrak.org/NORAD/elements/supplemental/"
        "sup-gp.php?FILE=starlink&FORMAT=tle"
    ),
}


@dataclass(frozen=True)
class TLEMetadata:
    """Metadata recorded with each downloaded TLE snapshot."""

    source: str
    url: str
    downloaded_at_utc: str
    filename: str
    sha256: str
    line_count: int
    object_count_estimate: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        choices=sorted(CELESTRAK_URLS),
        default="gp",
        help="CelesTrak source. Use gp for catalog GP data; supplemental for SpaceX-derived data.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where the TLE snapshot and metadata will be saved.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def download_text(url: str, timeout: float) -> str:
    """Download UTF-8 text from a URL."""
    request = urllib.request.Request(url, headers={"User-Agent": "leo-resource-alloc/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = cast(bytes, response.read())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download TLE data from {url}: {exc}") from exc
    return data.decode("utf-8")


def validate_tle_text(text: str) -> list[str]:
    """Validate that downloaded text looks like TLE/3LE content."""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) < 3:
        raise ValueError("Downloaded content is too short to be a TLE snapshot.")
    line1_count = sum(line.startswith("1 ") for line in lines)
    line2_count = sum(line.startswith("2 ") for line in lines)
    if line1_count == 0 or line1_count != line2_count:
        raise ValueError(
            f"Downloaded content does not look like valid TLE data: "
            f"line1_count={line1_count}, line2_count={line2_count}."
        )
    return lines


def write_snapshot(source: str, out_dir: Path, text: str, url: str) -> TLEMetadata:
    """Write TLE text and sidecar metadata."""
    lines = validate_tle_text(text)
    out_dir.mkdir(parents=True, exist_ok=True)

    downloaded_at = datetime.now(UTC)
    stamp = downloaded_at.strftime("%Y%m%dT%H%M%SZ")
    filename = f"starlink_{source}_{stamp}.tle"
    path = out_dir / filename

    normalized_text = "\n".join(lines) + "\n"
    path.write_text(normalized_text, encoding="utf-8")

    digest = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
    metadata = TLEMetadata(
        source=source,
        url=url,
        downloaded_at_utc=downloaded_at.isoformat(),
        filename=filename,
        sha256=digest,
        line_count=len(lines),
        object_count_estimate=sum(line.startswith("1 ") for line in lines),
    )
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    """Download and save a Starlink TLE snapshot."""
    args = parse_args()
    url = CELESTRAK_URLS[args.source]
    text = download_text(url, timeout=args.timeout)
    metadata = write_snapshot(args.source, args.out_dir, text, url)
    print(json.dumps(asdict(metadata), indent=2))


if __name__ == "__main__":
    main()
