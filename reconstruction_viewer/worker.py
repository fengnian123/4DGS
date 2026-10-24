import argparse
import base64
import contextlib
import importlib
import io
import json
import sys
import traceback
from pathlib import Path

from PIL import Image
from registry import load_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    args = parser.parse_args()
    entries = {entry["id"]: entry for entry in load_registry(args.registry)}
    adapter, active = None, None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            entry = entries[request["scene"]]
            with contextlib.redirect_stdout(sys.stderr):
                if active != entry["id"]:
                    if adapter is not None:
                        del adapter
                        import torch
                        torch.cuda.empty_cache()
                    adapter, active = None, None
                    module, class_name = entry["adapter"].split(":")
                    adapter = getattr(importlib.import_module(module), class_name)(entry)
                    active = entry["id"]
                if request["operation"] == "metadata":
                    response = {"metadata": adapter.metadata}
                elif request["operation"] == "render":
                    image = adapter.render(request)
                    buffer = io.BytesIO()
                    Image.fromarray(image).save(buffer, format="PNG")
                    response = {"image": base64.b64encode(buffer.getvalue()).decode()}
                else:
                    raise ValueError(request["operation"])
            response["ok"] = True
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            response = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
