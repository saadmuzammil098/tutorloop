"""Extract every mermaid code block from every README.md in the repo and
render each one with mermaid-cli. A parse error here means a raw mermaid
error message would otherwise only surface later, unrendered, on GitHub.

Requires Node and network access once to fetch @mermaid-js/mermaid-cli via
npx, mmdc itself does not call out anywhere at render time.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MERMAID_BLOCK = re.compile(r"```mermaid\n(.*?)```", re.S)

# CI runners (and some sandboxes) don't have the kernel-level user namespace
# support Chromium's sandbox needs, mermaid-cli renders through headless
# Chromium under the hood, so it fails to even launch the browser without
# this, unrelated to whether the diagram source itself is valid.
PUPPETEER_CONFIG = {"args": ["--no-sandbox", "--disable-setuid-sandbox"]}


def find_readmes() -> list[Path]:
    return sorted(REPO_ROOT.rglob("README.md"))


def render(block_source: str, workdir: Path, index: int, puppeteer_config_path: Path) -> tuple[bool, str]:
    mmd_path = workdir / f"diagram_{index}.mmd"
    svg_path = workdir / f"diagram_{index}.svg"
    mmd_path.write_text(block_source)
    result = subprocess.run(
        [
            "npx", "-y", "@mermaid-js/mermaid-cli",
            "-i", str(mmd_path),
            "-o", str(svg_path),
            "-p", str(puppeteer_config_path),
        ],
        capture_output=True,
        text=True,
    )
    ok = result.returncode == 0 and svg_path.exists() and svg_path.stat().st_size > 0
    return ok, result.stdout + result.stderr


def main() -> int:
    readmes = find_readmes()
    if not readmes:
        print("no README.md files found")
        return 0

    failures = 0
    checked = 0
    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        puppeteer_config_path = workdir / "puppeteer-config.json"
        puppeteer_config_path.write_text(json.dumps(PUPPETEER_CONFIG))
        for readme in readmes:
            text = readme.read_text()
            blocks = MERMAID_BLOCK.findall(text)
            for i, block in enumerate(blocks):
                checked += 1
                ok, log = render(block, workdir, checked, puppeteer_config_path)
                relative = readme.relative_to(REPO_ROOT)
                if ok:
                    print(f"OK   {relative} (diagram {i + 1})")
                else:
                    failures += 1
                    print(f"FAIL {relative} (diagram {i + 1})")
                    print(log)

    print(f"\n{checked} mermaid diagram(s) checked, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
