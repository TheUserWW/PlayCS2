"""Official demoparser2 Node adapter; no game process or memory access."""

import shutil
import subprocess
import tempfile
from pathlib import Path

from .io import read_json, write_json


ROOT = Path(__file__).resolve().parent


def query(demo, operation, *, timeout=300, **options):
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required for demo parsing; install Node.js and rerun setup.")
    if not (ROOT / "node/node_modules/@laihoe/demoparser2").is_dir():
        raise RuntimeError("Install the parser with: npm.cmd ci --prefix pretraining/node")
    with tempfile.TemporaryDirectory(prefix="cs2-parser-") as directory:
        request, output = Path(directory) / "request.json", Path(directory) / "result.json"
        write_json(request, {"demo": str(Path(demo).resolve()), "operation": operation, "output": str(output), **options})
        result = subprocess.run([node, "--max-old-space-size=4096", str(ROOT / "node/reader.cjs"), str(request)],
                                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
        if result.returncode:
            raise RuntimeError(f"demo parser failed ({result.returncode}): {result.stderr[-3000:]}")
        return read_json(output)
