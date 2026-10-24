import time

import torch


def load_checkpoint(path):
    for attempt in range(5):
        try:
            return torch.load(path, map_location='cpu', weights_only=False)
        except (RuntimeError, EOFError, OSError):
            if attempt == 4:
                raise
            time.sleep(.5)
