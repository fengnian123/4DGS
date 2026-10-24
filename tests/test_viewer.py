import json
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'reconstruction_viewer'))
from media import start_media
from registry import load_registry


class RegistryTests(unittest.TestCase):
    def test_python_symlink_retains_environment(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            executable = directory / 'venv-python'
            executable.symlink_to(sys.executable)
            registry = directory / 'scenes.json'
            registry.write_text(json.dumps([{'id': 'x', 'label': 'x', 'root': '.', 'python': 'venv-python'}]))
            self.assertEqual(load_registry(registry)[0]['python'], str(executable))

    def test_portable_registry(self):
        entries = load_registry(ROOT / 'reconstruction_viewer/scenes.json')
        self.assertEqual(len(entries), 4)
        self.assertTrue(all(Path(e['root']).is_absolute() for e in entries))
        self.assertTrue(all('agentos' not in e['id'] for e in entries))
        for entry in entries:
            metrics = Path(entry['root']) / entry['results'] / 'metrics.json'
            self.assertTrue(metrics.is_file(), metrics)
            json.loads(metrics.read_text())

    def test_duplicate_id(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'scenes.json'
            path.write_text(json.dumps([{'id': 'x', 'label': 'x'}] * 2))
            with self.assertRaises(ValueError):
                load_registry(path)

    def test_gallery_and_traversal(self):
        entries = load_registry(ROOT / 'reconstruction_viewer/scenes.json')
        server = start_media(entries, 0, 8890, ROOT / 'reconstruction_viewer')
        base = f'http://127.0.0.1:{server.server_port}'
        try:
            with urllib.request.urlopen(base) as response:
                self.assertIn('Coffee Martini', response.read().decode())
            with urllib.request.urlopen(base + '/media/4dgs-coffee/metrics.json') as response:
                self.assertEqual(json.load(response)['test_frames'], 300)
            for path in ('/media/4dgs-coffee/%2e%2e/%2e%2e/requirements-local.txt', '/scene/unknown'):
                with self.assertRaises(urllib.error.HTTPError) as caught:
                    urllib.request.urlopen(base + path)
                self.assertEqual(caught.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == '__main__':
    unittest.main()
