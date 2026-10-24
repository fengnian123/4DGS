import base64
import io
import json
import os
import subprocess
import threading
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


class Backend:
    def __init__(self, entry, registry):
        (ROOT / "logs").mkdir(exist_ok=True)
        self.log = (ROOT / "logs" / f'{entry["id"]}.log').open("a")
        environment = dict(os.environ)
        environment.setdefault("MAX_JOBS", "4")
        environment.setdefault("OMP_NUM_THREADS", "4")
        self.process = subprocess.Popen([entry["python"], "-u", str(ROOT / "worker.py"), "--registry", str(registry)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, env=environment)
        self.lock = threading.Lock()

    def call(self, request):
        with self.lock:
            if self.process.poll() is not None:
                raise RuntimeError(f"Render worker exited: {self.process.returncode}")
            self.process.stdin.write(json.dumps(request) + "\n")
            self.process.stdin.flush()
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Render worker disconnected; see logs")
            response = json.loads(line)
            if not response["ok"]:
                raise RuntimeError(response["error"])
            return response

    def metadata(self, scene):
        return self.call({"operation": "metadata", "scene": scene})["metadata"]

    def render(self, scene, **query):
        response = self.call({"operation": "render", "scene": scene, **query})
        return np.asarray(Image.open(io.BytesIO(base64.b64decode(response["image"]))).convert("RGB"))

    def close(self):
        with self.lock:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.log.close()
