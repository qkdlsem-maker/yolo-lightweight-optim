"""Known-answer positive/false-positive audit; no detector accuracy claim.

The separate unmodified-inference fixture has zero true positives. Here only
the test replaces postprocessing with fabricated post-NMS boxes, while keeping
native coordinate conversion, matching, AP and the new audit recorder active.
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from ultralytics.models.yolo.detect.val import DetectionValidator

import test_native_bdd_validation as fixture
from prediction_archive import sha256_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError('Preserve previous positive-statistics proof')
    original_run = fixture.run
    captured = {}

    def controlled_run(spec, output, device='cpu'):
        if Path(output).exists():
            return original_run(spec, output, device)
        frame = 0

        def fabricated_postprocess(self, model_outputs):
            nonlocal frame
            tensor = model_outputs[0] if isinstance(model_outputs, (tuple, list)) else model_outputs
            rows = []
            for unused in range(tensor.shape[0]):
                # Fixture0 has neither ground truth nor predictions. Every
                # other frame has one high-score FP and one exact-match TP.
                boxes = [] if frame == 0 else [[0, 0, 20, 20, .95, frame % 10],
                                               [288, 302, 352, 338, .90, frame % 10]]
                rows.append(torch.tensor(boxes, dtype=torch.float32, device=tensor.device).reshape(-1, 6))
                frame += 1
            return rows

        with patch.object(DetectionValidator, 'postprocess', fabricated_postprocess):
            result = original_run(spec, output, device)
        assert frame == 33
        score = json.loads((Path(output) / 'score.json').read_bytes())
        records = [np.load(path, allow_pickle=False) for path in sorted((Path(output) / 'native_stats').glob('*.npz'))]
        tp = np.concatenate([x['tp'] for x in records])
        pred_ids = np.concatenate([x['prediction_image_id'] for x in records])
        gt_ids = np.concatenate([x['target_image_id'] for x in records])
        for x in records:
            x.close()
        assert tp.shape == (64, 10) and np.array_equal(tp.sum(axis=0), np.full(10, 32))
        assert np.array_equal(pred_ids, np.repeat(np.arange(2, 34), 2))
        assert np.array_equal(gt_ids, np.arange(2, 34))
        ap = np.asarray(score['class_AP_by_IoU'])
        assert ap.shape == (10, 10) and np.allclose(ap, .4975, atol=1e-12, rtol=0)
        assert abs(score['results']['metrics/mAP50-95(B)'] - .4975) < 1e-12
        captured.update(predictions=64, true_positives_at_every_IoU=32,
                        false_positives_at_every_IoU=32, target_boxes=32,
                        exact_prediction_and_target_image_id_alignment=True,
                        known_native_AP50_95_all_classes=.4975,
                        empty_ground_truth_and_prediction_image_retained=True)
        return result

    with tempfile.TemporaryDirectory(prefix='positive_native_bdd_proof_') as tmp:
        inner = Path(tmp) / 'integration.json'
        with patch.object(fixture, 'run', controlled_run), patch.object(sys, 'argv', [
                'test_native_bdd_validation.py', '--checkpoint', str(a.checkpoint), '--output', str(inner)]):
            fixture.main()
        proof = json.loads(inner.read_bytes())
    proof.update(scope='Synthetic known-answer positive/false-positive statistics integration only',
                 postprocessing_replaced_only_in_this_synthetic_fixture=True,
                 native_coordinate_conversion_matching_and_AP_unmodified=True,
                 known_answer=captured,
                 AP_explanation='Native101-point trapezoidal integration of the half-precision envelope gives0.4975; this is not COCO AP and not a detector result.')
    proof['source_sha256'][Path(__file__).name] = sha256_file(Path(__file__))
    a.output.write_bytes((json.dumps(proof, indent=2) + '\n').encode())
    print(json.dumps({'status': proof['status'], 'known_answer': captured}, indent=2))


if __name__ == '__main__':
    main()
