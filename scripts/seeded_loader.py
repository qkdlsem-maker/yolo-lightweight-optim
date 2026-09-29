"""Explicit sampler and augmentation seeds for future independent training runs.

This helper was not used to generate the September 22 fixed-data experiment.
"""
import torch
from ultralytics.data.build import seed_worker

def build_seeded_loader(dataset, batch, workers, seed):
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(
        dataset, batch_size=min(batch, len(dataset)), shuffle=True,
        num_workers=workers, pin_memory=True,
        collate_fn=getattr(dataset, 'collate_fn', None),
        worker_init_fn=seed_worker, generator=generator,
        persistent_workers=workers > 0,
    )
