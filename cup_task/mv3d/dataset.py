"""torch Dataset + workered DataLoader for the DELTA CDRNet lifter — copies YOLO's speed pattern.

YOLO trains fast because its DataLoader uses num_workers + pin_memory to load/decode data on CPU
workers in PARALLEL while the GPU trains. Our earlier loop did data+compute serially on one thread.
This wraps DeltaTrial's pre-extracted frame cache in a Dataset so a standard DataLoader with workers
feeds the GPU continuously.

A sample = one synchronized multi-view frame: variable #cams (3-5) -> can't stack into a fixed
tensor, so collate_fn keeps a LIST (like YOLO keeps variable #instances per image). The GPU loop
processes each grouped sample; workers hide the frame-load latency.
"""
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from .data_delta import DeltaTrial, load_amq, WRIST_IDX


class DeltaFrames(Dataset):
    """Flattened (trial, frame) index over several trials; __getitem__ returns one grouped sample.

    Frames come from the memmapped pre-extracted cache (fast, worker-safe: each worker opens its own
    memmap on first access). Returns cpu tensors; the train loop moves to GPU.
    """
    def __init__(self, trials, part='P07', amq=None, min_cams=3, frame_stride=1):
        self.part = part
        self.amq = amq if amq is not None else load_amq()
        self._trials = {}          # name -> DeltaTrial (lazy per worker)
        self._spec = trials        # [(name, tn), ...]
        self.index = []            # [(name, tn, frame), ...]
        for name, tn in trials:
            t = DeltaTrial(part, name, tn, self.amq)
            for f in t.valid_frames(min_cams=min_cams)[::frame_stride]:   # subsample for fast finetune
                self.index.append((name, tn, f))
            self._trials[name] = t

    def __len__(self):
        return len(self.index)

    def _trial(self, name, tn):
        # in a worker process the DeltaTrial (with its memmaps) is re-created lazily
        if name not in self._trials or self._trials[name] is None:
            self._trials[name] = DeltaTrial(self.part, name, tn, self.amq)
        return self._trials[name]

    def __getitem__(self, i):
        name, tn, f = self.index[i]
        s = self._trial(name, tn).sample(f, device='cpu')
        if s is None:                       # shouldn't happen (valid_frames filtered), guard anyway
            s = self._trial(name, tn).sample(self.index[(i + 1) % len(self)][2], device='cpu')
        s['_key'] = (name, tn, f)
        return s


def collate_group(batch):
    """Keep the list of grouped samples (variable #cams each) — no stacking."""
    return [b for b in batch if b is not None]


def make_loader(trials, part='P07', amq=None, batch=8, workers=4, shuffle=True, min_cams=3,
                frame_stride=1):
    ds = DeltaFrames(trials, part, amq, min_cams, frame_stride)
    dl = DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers,
                    pin_memory=True, collate_fn=collate_group, persistent_workers=workers > 0,
                    drop_last=False)
    return ds, dl
