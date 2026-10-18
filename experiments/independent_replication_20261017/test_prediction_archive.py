"""Synthetic interruption, integrity and exclusion checks; no models or datasets."""
import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

from prediction_archive import PredictionArchive, canonical, digest, make_manifest
from prediction_io import INFERENCE, prediction_record


def fixtures():
    binding = {'endpoint_id': 'synthetic_only', 'dataset_id': 'generated_metadata_only',
               'checkpoint_sha256': 'a' * 64, 'protocol_sha256': 'b' * 64,
               'source_sha256': {'synthetic.py': 'c' * 64}, 'runtime': {'purpose': 'unit_test'},
               'inference': copy.deepcopy(INFERENCE)}
    images = [{'id': i + 1, 'file_name': f'{i:06d}.png', 'width': 120, 'height': 80,
               'sha256': digest(str(i).encode())} for i in range(33)]
    records = [prediction_record(im, [] if i % 3 == 0 else [[0, 0, 0, 10, .1, 0], [1, 1, 50, 50, .7, 2], [0, 0, 20, 30, .2, 9]])
               for i, im in enumerate(images)]
    return binding, images, records


def trace(count, dtype='torch.float32'):
    return [{'shape': [count, 3, 640, 640], 'dtype': dtype, 'device': 'cpu'}]


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / 'archive'
        self.binding, self.images, self.records = fixtures()

    def tearDown(self):
        self.tmp.cleanup()

    def archive(self):
        return PredictionArchive(self.root, self.binding, self.images)

    def committed(self):
        with self.archive() as archive:
            archive.commit(0, self.records[:32], trace(32))

    def test_roundtrip_resume_empty_zero_area_other_class_and_complete(self):
        self.committed()
        before = {str(p.relative_to(self.root)): p.read_bytes() for p in (self.root / 'batches').rglob('*.json')}
        with self.archive() as archive:
            self.assertEqual(archive.verify()['committed_images'], 32)
            archive.commit(1, self.records[32:], trace(1))
            self.assertTrue(archive.finalize()['complete'])
            self.assertEqual(list(archive.iter_records()), self.records)
            self.assertTrue(archive.finalize()['complete'])
        for name, raw in before.items():
            self.assertEqual((self.root / name).read_bytes(), raw)
        with self.archive() as archive:
            self.assertTrue(archive.verify()['complete'])

    def test_preserve_interrupted_staging_but_do_not_count_it(self):
        self.committed()
        orphan = self.root / 'staging' / 'interrupted-batch1'
        orphan.mkdir()
        (orphan / 'partial.json').write_bytes(b'{')
        with self.archive() as archive:
            self.assertEqual(archive.verify()['committed_batches'], 1)
            self.assertEqual(archive.verify()['preserved_staging_entries'], [orphan.name])
            archive.commit(1, self.records[32:], trace(1))
            archive.finalize()
        self.assertEqual((orphan / 'partial.json').read_bytes(), b'{')

    def test_checkpoint_protocol_runtime_source_or_inference_change_rejected(self):
        self.committed()
        for key, value in [('checkpoint_sha256', 'd' * 64), ('protocol_sha256', 'e' * 64),
                           ('runtime', {'purpose': 'different'}), ('source_sha256', {'synthetic.py': 'f' * 64}),
                           ('inference', {**INFERENCE, 'half': True}), ('endpoint_id', 'other')]:
            changed = copy.deepcopy(self.binding)
            changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                with PredictionArchive(self.root, changed, self.images):
                    pass

    def test_image_content_metadata_or_order_change_rejected(self):
        self.committed()
        for kind in ('content', 'metadata', 'order'):
            changed = copy.deepcopy(self.images)
            if kind == 'content':
                changed[0]['sha256'] = '9' * 64
            elif kind == 'metadata':
                changed[0]['width'] += 1
            else:
                changed.reverse()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                with PredictionArchive(self.root, self.binding, changed):
                    pass

    def test_changed_prediction_bytes_rejected(self):
        self.committed()
        p = self.root / 'batches' / '000000' / '00000002.json'
        p.write_bytes(p.read_bytes().replace(b'0.7', b'0.8'))
        with self.assertRaises(ValueError):
            with self.archive():
                pass

    def test_extra_prediction_file_rejected(self):
        self.committed()
        (self.root / 'batches' / '000000' / 'unexpected.json').write_text('{}')
        with self.assertRaises(ValueError):
            with self.archive():
                pass

    def test_missing_prediction_file_rejected(self):
        self.committed()
        (self.root / 'batches' / '000000' / '00000001.json').unlink()
        with self.assertRaises(ValueError):
            with self.archive():
                pass

    def test_noncontiguous_batch_rejected(self):
        self.committed()
        (self.root / 'batches' / '000000').rename(self.root / 'batches' / '000001')
        with self.assertRaises(ValueError):
            with self.archive():
                pass

    def test_existing_commit_is_never_overwritten(self):
        self.committed()
        with self.archive() as archive, self.assertRaises(ValueError):
            archive.commit(0, self.records[:32], trace(32))

    def test_partial_or_wrong_order_results_rejected_without_commit(self):
        with self.archive() as archive:
            for records in (self.records[:31], list(reversed(self.records[:32]))):
                with self.assertRaises(ValueError):
                    archive.commit(0, records, trace(32))
            self.assertEqual(archive.verify()['committed_batches'], 0)

    def test_wrong_tensor_geometry_precision_or_forward_count_rejected(self):
        with self.archive() as archive:
            wrong = trace(32)
            wrong[0]['shape'][2] = 384
            for observed in (trace(31), trace(32, 'torch.float16'), wrong, [], trace(32) + trace(32)):
                with self.assertRaises(ValueError):
                    archive.commit(0, self.records[:32], observed)
            self.assertEqual(archive.verify()['committed_batches'], 0)

    def test_incomplete_archive_cannot_finalize_or_score(self):
        self.committed()
        with self.archive() as archive:
            with self.assertRaises(ValueError):
                archive.finalize()
            with self.assertRaises(ValueError):
                list(archive.iter_records())

    def test_mutated_completion_marker_rejected(self):
        with self.archive() as archive:
            archive.commit(0, self.records[:32], trace(32))
            archive.commit(1, self.records[32:], trace(1))
            archive.finalize()
        p = self.root / 'COMPLETE.json'
        changed = json.loads(p.read_bytes())
        changed['images'] = 32
        p.write_bytes(canonical(changed))
        with self.assertRaises(ValueError):
            with self.archive():
                pass

    def test_concurrent_writer_excluded_then_lock_released(self):
        with self.archive():
            with self.assertRaises(RuntimeError):
                with self.archive():
                    pass
        with self.archive() as archive:
            self.assertEqual(archive.verify()['committed_batches'], 0)

    def test_missing_hash_duplicate_id_or_filename_rejected(self):
        for kind in ('missing_hash', 'bad_hash', 'duplicate_id', 'duplicate_filename'):
            images = copy.deepcopy(self.images)
            if kind == 'missing_hash':
                images[0].pop('sha256')
            elif kind == 'bad_hash':
                images[0]['sha256'] = 'unknown'
            elif kind == 'duplicate_id':
                images[1]['id'] = images[0]['id']
            else:
                images[1]['file_name'] = images[0]['file_name']
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                make_manifest(self.binding, images)

    def test_manifest_copied_and_context_required(self):
        archive = self.archive()
        self.images[0]['width'] = 1000
        self.binding['runtime']['purpose'] = 'mutated'
        self.assertEqual(archive.images[0]['width'], 120)
        self.assertEqual(archive.manifest['bindings']['runtime']['purpose'], 'unit_test')
        with self.assertRaises(RuntimeError):
            archive.verify()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve previous test evidence')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ArchiveTests)
    names = [test.id().split('.')[-1] for test in suite]
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    evidence = {'status': 'passed' if result.wasSuccessful() else 'failed',
                'tests_run': result.testsRun, 'test_names': names,
                'scope': 'Synthetic archive state and records only; no model inference or real data.',
                'real_model_evaluation_complete': False,
                'source_sha256': {n: digest((Path(__file__).parent / n).read_bytes()) for n in
                                  ('prediction_archive.py', 'test_prediction_archive.py', 'prediction_io.py')}}
    args.output.write_bytes(canonical(evidence))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == '__main__':
    main()
