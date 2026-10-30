import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from unittest.mock import patch

from neural_lab.dashboard import handler_for, read_file, read_queue, snapshot
from neural_lab.queue import Queue


class DashboardTests(unittest.TestCase):
    def test_empty_observation_does_not_initialize_state(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root) / 'missing'
            with patch('neural_lab.dashboard.read_gpu', return_value={'available': False, 'devices': []}):
                result = snapshot(state)
            self.assertEqual(result['queue']['jobs'], [])
            self.assertFalse(state.exists())

    def test_live_queue_reads_results_without_recovery_or_writes(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)
            script = state / 'source.py'
            script.write_text('pass')
            queue = Queue(state)
            job_id = queue.enqueue(script, 'A testable hypothesis')
            queue.next_job()
            observed = read_queue(state)
            self.assertEqual(observed['counts'], {'running': 1})
            self.assertEqual(queue.status()['counts'], {'running': 1})
            queue.finish(job_id, {'status': 'succeeded', 'result': {
                'metrics': {'candidate_accuracy': 0.8, 'baseline_accuracy': 0.9}, 'summary': 'Hypothesis unsupported'}})
            result = read_queue(state)
            self.assertEqual(result['jobs'][0]['result']['metrics']['candidate_accuracy'], 0.8)
            self.assertNotIn('output_dir', result['jobs'][0])
            output = state / 'queue-outputs' / job_id
            output.mkdir(parents=True)
            (output / 'details.json').write_text(json.dumps({'environment': {'device': 'cuda:0'}, 'records': 'x' * 300000}))
            self.assertEqual(read_queue(state)['jobs'][0]['device'], 'cuda:0')
            queue.close()

    def test_newer_session_supersedes_historical_pause_and_private_fields_stay_private(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)
            notes = state / 'notes'
            notes.mkdir()
            (notes / 'pause-state.json').write_text(json.dumps({'updated_at': 1, 'reason': 'old pause'}))
            (notes / 'latest-session.json').write_text(json.dumps({'updated_at': 2, 'reason': 'running', 'private': 'sentinel'}))
            with patch('neural_lab.dashboard.read_gpu', return_value={'available': False, 'devices': []}):
                result = snapshot(state)
            self.assertEqual(result['scientist']['reason'], 'running')
            self.assertNotIn('sentinel', json.dumps(result))

    def test_http_routes_are_read_only_bounded_and_do_not_serve_arbitrary_files(self):
        with tempfile.TemporaryDirectory() as root:
            state = Path(root)
            (state / 'notes').mkdir()
            (state / 'notes' / 'handoff.md').write_text('<script>alert("untrusted note")</script>')
            (state / 'notes' / 'linked.md').symlink_to(state / 'secret.md')
            (state / 'secret.md').write_text('private-sentinel')
            (state / 'notes' / 'huge.md').write_text('x' * 65537)
            server = ThreadingHTTPServer(('127.0.0.1', 0), handler_for(state, ['lab.example.ts.net']))
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            base = f'http://127.0.0.1:{server.server_port}'
            try:
                with urlopen(base + '/') as response:
                    self.assertIn("script-src 'self'", response.headers['Content-Security-Policy'])
                    self.assertIn('Hypotheses meet evidence', response.read().decode())
                with urlopen(base + '/api/note?name=handoff.md') as response:
                    self.assertTrue(json.load(response)['text'].startswith('<script>'))
                    self.assertEqual(response.headers['Cache-Control'], 'no-store')
                for path, code in [('/api/note?name=../secret.md', 400), ('/api/note?name=linked.md', 404),
                                   ('/api/note?name=huge.md', 404), ('/.env', 404), ('/queue.sqlite3', 404)]:
                    with self.assertRaises(HTTPError) as error:
                        urlopen(base + path)
                    self.assertEqual(error.exception.code, code)
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(base + '/', headers={'Host': 'attacker.example'}))
                self.assertEqual(error.exception.code, 403)
                with urlopen(Request(base + '/', headers={'Host': 'lab.example.ts.net'})) as response:
                    self.assertEqual(response.status, 200)
                with self.assertRaises(HTTPError) as error:
                    urlopen(Request(base + '/api/status', data=b'{}', method='POST'))
                self.assertEqual(error.exception.code, 501)
                with self.assertRaises(ValueError):
                    read_file(state / 'notes' / 'huge.md')
            finally:
                server.shutdown()
                server.server_close()
                thread.join()


if __name__ == '__main__':
    unittest.main()
