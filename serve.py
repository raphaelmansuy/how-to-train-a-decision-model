"""Load the ONNX bundle and serve POST /v1/route."""

import json
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bank_routing.paths import root
from bank_routing.service import RouterService, create_app
from bank_routing.ship import onnx_logits


def main() -> None:
    bundle = root() / "models" / "bundle"
    manifest = json.loads((bundle / "manifest.json").read_text())
    service = RouterService(
        onnx_logits(bundle),
        bundle_dir=bundle,
        audit_path=root() / "audit" / "decisions.log",
        kill_switch_file=root() / "KILL_SWITCH",
        model_version=manifest["model_version"],
    )
    uvicorn.run(create_app(service), host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
