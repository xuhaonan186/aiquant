import os
import h5py
import numpy as np


def mmap_h5dset(pth, key):
    """memory map shape args: pth, key"""
    with h5py.File(pth, 'r') as f:
        ds = f[key]
        offset = ds.id.get_offset()
        assert ds.chunks is None
        assert ds.compression is None
        assert offset > 0
        dtype = ds.dtype
        shape = ds.shape
    arr = np.memmap(pth, mode='r', shape=shape, offset=offset, dtype=dtype)
    return arr


def mmap_h5dset_shape(pth, key):
    """memory map shape args: pth, key"""
    with h5py.File(pth, 'r') as f:
        ds = f[key]
        shape = ds.shape
    return shape


def load_info(name):
    pth = r'K:\pip_packages\aiquant_main\users\xhn\datasets\Info.h5'
    feature_mmap = mmap_h5dset(pth, name)
    return feature_mmap


if __name__ == '__main__':
    pass
