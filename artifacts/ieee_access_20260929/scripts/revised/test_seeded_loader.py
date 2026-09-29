"""Check reproducibility and between-seed differences in samples and augmentations."""
import random
import numpy as np
import torch
from seeded_loader import build_seeded_loader

class ProbeDataset(torch.utils.data.Dataset):
    def __len__(self): return 24
    def __getitem__(self, index):
        return index, random.random(), float(np.random.rand()), float(torch.rand(()))

def stream(seed):
    loader = build_seeded_loader(ProbeDataset(), 4, 2, seed)
    return [torch.cat([x.reshape(-1) for x in batch]) for _ in range(2) for batch in loader]

if __name__ == '__main__':
    a, b, c = stream(0), stream(0), stream(1)
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    assert any(not torch.equal(x, y) for x, y in zip(a, c))
    assert any(not torch.equal(x[:4], y[:4]) for x, y in zip(a, c))
    assert any(not torch.equal(x[4:], y[4:]) for x, y in zip(a, c))
    print('PASS: same seed reproduces two epochs; different seeds change order and augmentation draws')
