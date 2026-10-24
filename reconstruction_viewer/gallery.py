"""Serve saved comparison videos and metrics without loading a GPU model."""
import argparse
import threading
from pathlib import Path

from media import start_media
from registry import load_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', type=Path, default=Path(__file__).with_name('scenes.json'))
    parser.add_argument('--port', type=int, default=8900)
    parser.add_argument('--viewer-port', type=int, default=8890)
    args = parser.parse_args()
    server = start_media(load_registry(args.registry), args.port, args.viewer_port, Path(__file__).parent)
    print(f'Gallery: http://127.0.0.1:{args.port}', flush=True)
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
