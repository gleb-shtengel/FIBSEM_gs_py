"""
tif_stack_to_zarr.py

Convert a sequential stack of TIF files into an OME-ZARR (OME-NGFF v0.4) array
suitable for viewing in Neuroglancer, parallelised via a Dask client.

USAGE
-----
    from dask.distributed import Client
    from FIBSEM_gs_py.tif_stack_to_zarr import tif_stack_to_zarr

    client = Client(...)           # create however you like (LSF, local, SSH, …)

    tif_stack_to_zarr(
        tif_files  = sorted_list_of_tif_paths,
        output_zarr_path= "/data/out.zarr",
        client     = client,
        chunk_z=64, chunk_y=128, chunk_x=128,
        voxel_size_zyx=(8., 8., 8.),
        origin_zyx=(0., 0., 0.),       # physical origin in voxel_unit
    )

If client=None the work runs sequentially on the calling process (useful for
testing or when a single node is fast enough).

ARCHITECTURE
------------
Each (Z-slab, Y-strip) pair is one independent unit of work:
    write_strip_to_zarr(output_zarr_path, tif_files, z_start, z_end, y_start, y_end)
        1. Opens each TIF via tifffile.memmap (zero-copy file mapping)
        2. Reads ONLY rows [y_start, y_end) from each file into RAM
        3. Writes the sub-block to zarr[z_start:z_end, y_start:y_end, :]
RAM per worker: chunk_z × strip_y × nx × itemsize  (e.g. ~1 GB for strip_y=1024)
Requires uncompressed TIF files.

Multiple workers can write to the same zarr store simultaneously because
zarr chunks are independent files and each job writes a non-overlapping region.

After all slabs are written, pyramid levels (s1, s2, …) are computed from s0
by submitting one client.submit() per output zarr chunk (_write_pyramid_chunk),
mirroring the slab pattern and avoiding any large graph serialisation.

OME-ZARR output
---------------
output.zarr/
  .zattrs    ← OME-NGFF v0.4 multiscales metadata
  .zgroup
  s0/         ← full resolution   shape=(nz, ny, nx)  chunks=(chunk_z, chunk_y, chunk_x)
  s1/         ← 2× downsampled
  s2/         ← 4× downsampled
  ...

Neuroglancer:
  http://<host>:<port>/path/to/output.zarr/|zarr2:

©G.Shtengel 2026  gleb.shtengel@gmail.com
"""

import json
import os
import math
import shutil
import time
import urllib.parse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed as thread_as_completed

import numpy as np
import zarr
from dask.distributed import as_completed
from FIBSEM_gs_py.FIBSEM_help_functions_gs import check_DASK

try:
    import tifffile as tiff
except ImportError:
    try:
        import skimage.external.tifffile as tiff
    except ImportError:
        raise ImportError("tifffile is required: pip install tifffile")

warnings.filterwarnings("ignore", category=DeprecationWarning)



# ---------------------------------------------------------------------------
# TIF reading
# ---------------------------------------------------------------------------

def _read_tif(path: str) -> np.ndarray:
    """Read one TIF. Thread-safe — tifffile releases the GIL."""
    img = tiff.imread(os.path.normpath(path))
    if img.ndim == 3:
        img = img[..., 0]   # RGB → first channel
    return img


# ---------------------------------------------------------------------------
# Compressor
# ---------------------------------------------------------------------------

def _make_compressor(name: str, level: int = 1):
    if name == "blosc":
        from numcodecs import Blosc
        return Blosc(cname="lz4", clevel=level, shuffle=Blosc.BITSHUFFLE)
    if name == "gzip":
        import numcodecs
        return numcodecs.GZip(level=level)
    return None


# ---------------------------------------------------------------------------
# OME-ZARR metadata
# ---------------------------------------------------------------------------

def _translation_at_level(base_scale_zyx, base_origin_zyx, level, downsample_factor=2):
    """
    Compute the physical-space translation of a downsampled pyramid level.

    At level n (2^n downsampling), the first output pixel averages 2^n input
    pixels.  Its centre lies at half a *full-resolution* voxel beyond the
    origin of level 0.  For factor 2:

        translation_n = origin_0 + scale_0 * (2^(n-1) − 0.5)   (n ≥ 1)
        translation_0 = origin_0

    This matches the convention used by the Janelia cellmap_utils library.
    """
    if level == 0:
        return list(base_origin_zyx)
    f = downsample_factor
    return [
        orig + scale * (f ** (level - 1) - 0.5)
        for scale, orig in zip(base_scale_zyx, base_origin_zyx)
    ]


def write_ome_zarr_metadata(
    root_store: zarr.Group,
    n_levels: int,
    voxel_size_zyx: tuple = (1.0, 1.0, 1.0),
    origin_zyx: tuple = (0.0, 0.0, 0.0),
    voxel_unit: str = "nanometer",
    downsample_factor: int = 2,
    dataset_name: str = "volume",
):
    """
    Write OME-NGFF v0.4 multiscales metadata compatible with Neuroglancer.

    Stored path names are 's0', 's1', … to match the Janelia/cellmap convention.
    Each dataset entry carries both a 'scale' and a 'translation' coordinate
    transformation so that Neuroglancer can render physical-space positions
    correctly.
    """
    sz, sy, sx = voxel_size_zyx
    oz, oy, ox = origin_zyx

    base_scale = [sz, sy, sx]
    base_origin = [oz, oy, ox]

    datasets = []
    for lvl in range(n_levels):
        scale_lvl = [s * downsample_factor ** lvl for s in base_scale]
        trans_lvl = _translation_at_level(base_scale, base_origin, lvl, downsample_factor)
        datasets.append({
            "path": f"s{lvl}",
            "coordinateTransformations": [
                {"type": "scale",       "scale":       scale_lvl},
                {"type": "translation", "translation": trans_lvl},
            ],
        })

    root_store.attrs.update({
        "multiscales": [{
            "version": "0.4",
            "name": dataset_name,
            "axes": [
                {"name": "z", "type": "space", "unit": voxel_unit},
                {"name": "y", "type": "space", "unit": voxel_unit},
                {"name": "x", "type": "space", "unit": voxel_unit},
            ],
            # global identity transform (required by spec)
            "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, 1.0, 1.0]},
            ],
            "datasets": datasets,
            "type": "mean" if n_levels > 1 else "none",
        }]
    })


# ---------------------------------------------------------------------------
# zarr store creation
# ---------------------------------------------------------------------------

def create_zarr_store(
    output_zarr_path: str,
    nz: int, ny: int, nx: int,
    dtype,
    chunk_z: int = 64, chunk_y: int = 128, chunk_x: int = 128,
    n_pyramid_levels: int = 4,
    downsample_factor: int = 2,
    zarr_compressor: str = "blosc",
    zarr_compressor_level: int = 1,
    voxel_size_zyx: tuple = (8.0, 8.0, 8.0),
    origin_zyx: tuple = (0.0, 0.0, 0.0),
    voxel_unit: str = "nanometer",
    dataset_name: str = "volume",
    overwrite: bool = True,
) -> zarr.Group:
    """
    Create an empty OME-ZARR store with all array levels pre-allocated.

    Arrays are stored under paths 's0', 's1', … inside the root group.
    Returns the root zarr.Group.
    """
    compressor = _make_compressor(zarr_compressor, zarr_compressor_level)
    root = zarr.open_group(output_zarr_path, mode="w" if overwrite else "w-")

    cur_nz, cur_ny, cur_nx = nz, ny, nx
    for level in range(n_pyramid_levels):
        lvl_chunks = (min(chunk_z, cur_nz), min(chunk_y, cur_ny), min(chunk_x, cur_nx))
        root.require_dataset(
            f"s{level}",
            shape=(cur_nz, cur_ny, cur_nx),
            chunks=lvl_chunks,
            dtype=dtype,
            compressor=compressor,
            fill_value=0,
            overwrite=overwrite,
        )
        print(f"  Level s{level}: shape=({cur_nz}, {cur_ny}, {cur_nx})  chunks={lvl_chunks}")
        cur_nz //= downsample_factor
        cur_ny  //= downsample_factor
        cur_nx  //= downsample_factor

    write_ome_zarr_metadata(
        root, n_levels=n_pyramid_levels,
        voxel_size_zyx=voxel_size_zyx, origin_zyx=origin_zyx,
        voxel_unit=voxel_unit,
        downsample_factor=downsample_factor, dataset_name=dataset_name,
    )
    return root


# ---------------------------------------------------------------------------
# Strip writer  — uses tifffile.memmap for memory-efficient Y-strip reads
# ---------------------------------------------------------------------------

def write_strip_to_zarr(
    output_zarr_path: str,
    tif_files: list,
    z_start: int,
    z_end: int,
    y_start: int,
    y_end: int,
    n_read_threads: int = 8,
    level: int = 0,
) -> dict:
    """
    Read a Y-strip from TIF files [z_start, z_end) using tifffile.memmap and
    write the resulting sub-block into zarr level `level`.

    Only rows [y_start, y_end) are loaded from each TIF file, so per-worker RAM is:
        (z_end - z_start) × (y_end - y_start) × nx × itemsize

    Requires uncompressed (or single-strip) TIF files so that tifffile.memmap
    can return a numpy.memmap backed directly by the file bytes.

    Parameters
    ----------
    output_zarr_path : path to existing .zarr store
    tif_files        : full ordered list of TIF paths for the entire volume
    z_start, z_end   : half-open Z-slice range for this task
    y_start, y_end   : half-open Y-strip range for this task
    n_read_threads   : threads for parallel strip reading within this worker
    level            : pyramid level to write into (0 = full resolution)

    Returns
    -------
    dict  {z_start, z_end, y_start, y_end, elapsed_s, mb_s}
    """
    import os, time
    import numpy as np
    import zarr
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    try:
        import tifffile as _tiff
    except ImportError:
        raise ImportError("tifffile is required for memmap mode: pip install tifffile")

    t0 = time.time()
    slab_files = tif_files[z_start:z_end]
    if not slab_files:
        return {"z_start": z_start, "z_end": z_end,
                "y_start": y_start, "y_end": y_end,
                "elapsed_s": 0.0, "mb_s": 0.0}

    def _read_strip(path):
        mm = _tiff.memmap(os.path.normpath(path))   # zero-copy file mapping
        strip = np.array(mm[y_start:y_end, :])      # copy only the strip into RAM
        del mm                                       # release the file mapping
        if strip.ndim == 3:
            strip = strip[..., 0]                   # RGB → first channel
        return strip

    n = len(slab_files)
    results = [None] * n
    with ThreadPoolExecutor(max_workers=min(n_read_threads, n)) as pool:
        futs = {pool.submit(_read_strip, f): i for i, f in enumerate(slab_files)}
        for fut in _as_completed(futs):
            results[futs[fut]] = fut.result()

    strip_slab = np.stack(results, axis=0)  # (z_end-z_start, y_end-y_start, nx)
    zarr.open(output_zarr_path, mode="r+")[f"s{level}"][z_start:z_end, y_start:y_end, :] = strip_slab

    elapsed = time.time() - t0
    mb = strip_slab.nbytes / 1e6
    del strip_slab
    return {"z_start": z_start, "z_end": z_end,
            "y_start": y_start, "y_end": y_end,
            "elapsed_s": elapsed, "mb_s": mb / elapsed}


# ---------------------------------------------------------------------------
# Pyramid chunk worker  — one unit of work per output zarr chunk
# ---------------------------------------------------------------------------

def _write_pyramid_chunk(
    output_zarr_path: str,
    src_level: int,
    dst_level: int,
    out_z: tuple,
    out_y: tuple,
    out_x: tuple,
    downsample_factor: int = 2,
) -> dict:
    """
    Downsample one chunk from pyramid level src_level into dst_level.

    Reads the corresponding 2× larger block from src_level, trims it to a
    multiple of downsample_factor, reduces with a plain numpy reshape+mean
    (no dask involved), and writes the result into dst_level.

    Designed to run on a Dask worker — fully self-contained, no shared state.
    out_z/y/x are (start, stop) pairs in *destination* coordinates.
    """
    import zarr
    import numpy as np
    import time

    t0 = time.time()
    f = downsample_factor

    root = zarr.open_group(output_zarr_path, mode="r+")
    src = root[f"s{src_level}"]
    dst = root[f"s{dst_level}"]

    # Input region: 2× larger in each axis, clipped to source bounds
    in_z0 = out_z[0] * f;  in_z1 = min(out_z[1] * f, src.shape[0])
    in_y0 = out_y[0] * f;  in_y1 = min(out_y[1] * f, src.shape[1])
    in_x0 = out_x[0] * f;  in_x1 = min(out_x[1] * f, src.shape[2])

    data = src[in_z0:in_z1, in_y0:in_y1, in_x0:in_x1]

    # Trim to exact multiples of f so reshape is clean
    nz, ny, nx = data.shape
    nz_t = (nz // f) * f
    ny_t = (ny // f) * f
    nx_t = (nx // f) * f

    if nz_t == 0 or ny_t == 0 or nx_t == 0:
        return {"elapsed_s": time.time() - t0}

    downsampled = (
        data[:nz_t, :ny_t, :nx_t]
        .reshape(nz_t // f, f, ny_t // f, f, nx_t // f, f)
        .mean(axis=(1, 3, 5))
        .astype(data.dtype)
    )

    dz, dy, dx = downsampled.shape
    dst[out_z[0]:out_z[0] + dz,
        out_y[0]:out_y[0] + dy,
        out_x[0]:out_x[0] + dx] = downsampled

    return {"elapsed_s": time.time() - t0}


# ---------------------------------------------------------------------------
# Rechunker
# ---------------------------------------------------------------------------

def _rechunk_chunk(src_path: str, dst_path: str,
                   z0: int, z1: int, y0: int, y1: int, x0: int, x1: int) -> dict:
    """
    Copy one chunk region from src_path/s0 to dst_path/s0.

    Worker function for rechunk_s0. Safe to run concurrently because each
    call writes to a non-overlapping region of the destination array.

    Parameters
    ----------
    src_path : str
        Path to the source OME-ZARR store root.
    dst_path : str
        Path to the destination OME-ZARR store root.
    z0, z1, y0, y1, x0, x1 : int
        Slice bounds for the chunk to copy.

    Returns
    -------
    dict with key 'elapsed_s'.
    """
    import os, time
    import numpy as np
    import zarr

    t0 = time.time()
    src = zarr.open(src_path, mode='r')['s0']
    dst = zarr.open(dst_path, mode='r+')['s0']
    dst[z0:z1, y0:y1, x0:x1] = src[z0:z1, y0:y1, x0:x1]
    return {"elapsed_s": time.time() - t0}


def rechunk_s0(
    zarr_path: str,
    chunk_z: int,
    chunk_y: int,
    chunk_x: int,
    client=None,
    DASK_client_retries: int = 3,
):
    """
    Rechunk the s0 array inside an OME-ZARR store in-place.

    Converts s0 from chunk_z=1 (used for safe concurrent writes during
    mosaic assembly) to the target chunking (chunk_z, chunk_y, chunk_x)
    via a temporary store, without ever loading the full volume into memory.

    Each pass can be parallelised by a DASK client using the same
    chunk-per-task pattern as finalize_pyramid.

    Parameters
    ----------
    zarr_path : str
        Path to the root of the OME-ZARR store. Must contain an 's0' array.
    chunk_z : int
        Target chunk size along Z.
    chunk_y : int
        Target chunk size along Y.
    chunk_x : int
        Target chunk size along X.
    client : dask.distributed.Client or None
        If provided, chunk copies are submitted as independent DASK tasks.
        If None, copies run sequentially in the calling process.
    DASK_client_retries : int
        Number of automatic retries per failed task. Default is 3.

    Notes
    -----
    Workflow:
        Pass 1  main store (chunk_z=1)  →  temp store (chunk_z=target)
        Delete s0 from main store, pre-allocate fresh s0 with target chunking.
        Pass 2  temp store (chunk_z=target)  →  main store (chunk_z=target)
        Delete temp store.

    Peak extra disk usage : ~1× size of s0.
    Peak memory per worker: one chunk  (chunk_z × chunk_y × chunk_x × itemsize).
    """
    zarr_tmp_path = zarr_path + '_tmp_rechunk'

    # Read source metadata
    src_arr = zarr.open(zarr_path, mode='r')['s0']
    shape    = src_arr.shape
    nz, ny, nx = shape
    compressor = src_arr.compressor
    dtype      = src_arr.dtype

    cz = min(chunk_z, nz)
    cy = min(chunk_y, ny)
    cx = min(chunk_x, nx)

    # Pre-allocate temp store with target chunking
    tmp_root = zarr.open_group(zarr_tmp_path, mode='w')
    tmp_root.require_dataset(
        's0', shape=shape, chunks=(cz, cy, cx),
        dtype=dtype, compressor=compressor, fill_value=0, overwrite=True,
    )

    # Build output chunk coordinate grid
    chunk_coords = [
        (z, min(z + cz, nz), y, min(y + cy, ny), x, min(x + cx, nx))
        for z in range(0, nz, cz)
        for y in range(0, ny, cy)
        for x in range(0, nx, cx)
    ]
    n_chunks    = len(chunk_coords)
    report_every = max(1, n_chunks // 20)

    def _run_pass(src, dst, label):
        t0 = time.time()
        print(time.strftime('%Y/%m/%d  %H:%M:%S') +
              f'   Rechunk {label}: {n_chunks:,} chunks ...')
        if client is not None:
            futures = [
                client.submit(
                    _rechunk_chunk, src, dst,
                    z0, z1, y0, y1, x0, x1,
                    pure=False, retries=DASK_client_retries,
                )
                for z0, z1, y0, y1, x0, x1 in chunk_coords
            ]
            n_done = 0
            for fut in as_completed(futures):
                fut.result()
                n_done += 1
                if n_done % report_every == 0 or n_done == n_chunks:
                    elapsed = time.time() - t0
                    eta_s   = (elapsed / n_done) * (n_chunks - n_done)
                    print(time.strftime('%Y/%m/%d  %H:%M:%S') +
                          f'   Rechunk {label}: {n_done:,}/{n_chunks:,} chunks  '
                          f'{elapsed:.1f}s elapsed  ETA {eta_s/60:.1f} min')
        else:
            for n_done, (z0, z1, y0, y1, x0, x1) in enumerate(chunk_coords, 1):
                _rechunk_chunk(src, dst, z0, z1, y0, y1, x0, x1)
                if n_done % report_every == 0 or n_done == n_chunks:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S') +
                          f'   Rechunk {label}: {n_done:,}/{n_chunks:,} chunks done')
        print(time.strftime('%Y/%m/%d  %H:%M:%S') +
              f'   Rechunk {label} done in {time.time() - t0:.1f}s')

    # Pass 1: main store (chunk_z=1) → temp store (chunk_z=target)
    _run_pass(zarr_path, zarr_tmp_path, 'pass 1 (main → temp)')

    # Delete original s0 and pre-allocate fresh one with target chunking
    dst_root = zarr.open_group(zarr_path, mode='r+')
    del dst_root['s0']
    dst_root.require_dataset(
        's0', shape=shape, chunks=(cz, cy, cx),
        dtype=dtype, compressor=compressor, fill_value=0, overwrite=True,
    )

    # Pass 2: temp → main (chunks now align 1:1; fast copy)
    _run_pass(zarr_tmp_path, zarr_path, 'pass 2 (temp → main)')

    # Remove temp store
    shutil.rmtree(zarr_tmp_path)
    print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Rechunking complete.')


# ---------------------------------------------------------------------------
# Pyramid builder
# ---------------------------------------------------------------------------

def finalize_pyramid(
    output_zarr_path: str,
    n_pyramid_levels: int = 4,
    downsample_factor: int = 2,
    chunk_z: int = 64, chunk_y: int = 128, chunk_x: int = 128,
    voxel_size_zyx: tuple = (8.0, 8.0, 8.0),
    origin_zyx: tuple = (0.0, 0.0, 0.0),
    voxel_unit: str = "nanometer",
    dataset_name: str = "volume",
    client=None,
    DASK_client_retries: int = 3,
):
    """
    Build downsampled pyramid levels (s1, s2, …) from s0 in the store.

    Each output zarr chunk is submitted as an independent client.submit() call
    (_write_pyramid_chunk), identical in spirit to the slab-writing phase.
    No dask.array graph is constructed, so the scheduler never receives a
    large serialised graph — eliminating the 'Sending large graph' warning.
    """
    root = zarr.open_group(output_zarr_path, mode="a")

    for level in range(1, n_pyramid_levels):
        src_arr = root[f"s{level - 1}"]
        src_shape = src_arr.shape
        f = downsample_factor

        # Output shape and chunks (pre-allocated by create_zarr_store)
        dst_arr = root[f"s{level}"]
        dst_shape = dst_arr.shape
        cz, cy, cx = dst_arr.chunks

        # Build the full grid of (out_z, out_y, out_x) coordinate pairs
        chunk_coords = [
            ((z, min(z + cz, dst_shape[0])),
             (y, min(y + cy, dst_shape[1])),
             (x, min(x + cx, dst_shape[2])))
            for z in range(0, dst_shape[0], cz)
            for y in range(0, dst_shape[1], cy)
            for x in range(0, dst_shape[2], cx)
        ]
        n_chunks = len(chunk_coords)
        report_every = max(1, n_chunks // 20)   # ~20 progress lines per level

        print(time.strftime('%Y/%m/%d  %H:%M:%S') +
              f"     Computing pyramid level s{level}  "
              f"shape={dst_shape}  {n_chunks:,} chunks …")
        t0 = time.time()

        if client is not None:
            futures = [
                client.submit(
                    _write_pyramid_chunk,
                    output_zarr_path, level - 1, level,
                    out_z, out_y, out_x, f,
                    pure=False,
                    retries=DASK_client_retries,
                )
                for out_z, out_y, out_x in chunk_coords
            ]
            n_done = 0
            for fut in as_completed(futures):
                fut.result()   # re-raises after retries exhausted
                n_done += 1
                if n_done % report_every == 0 or n_done == n_chunks:
                    elapsed = time.time() - t0
                    eta_s = (elapsed / n_done) * (n_chunks - n_done)
                    print(time.strftime('%Y/%m/%d  %H:%M:%S') +
                          f"     s{level}: {n_done:,}/{n_chunks:,} chunks  "
                          f"{elapsed:.1f}s elapsed  ETA {eta_s/60:.1f} min")
        else:
            for n_done, (out_z, out_y, out_x) in enumerate(chunk_coords, 1):
                _write_pyramid_chunk(
                    output_zarr_path, level - 1, level,
                    out_z, out_y, out_x, f,
                )
                if n_done % report_every == 0 or n_done == n_chunks:
                    print(time.strftime('%Y/%m/%d  %H:%M:%S') +
                          f"     s{level}: {n_done:,}/{n_chunks:,} chunks done")

        print(time.strftime('%Y/%m/%d  %H:%M:%S') +
              f"     s{level} done in {time.time() - t0:.1f}s")

    write_ome_zarr_metadata(
        root, n_levels=n_pyramid_levels,
        voxel_size_zyx=voxel_size_zyx, origin_zyx=origin_zyx,
        voxel_unit=voxel_unit,
        downsample_factor=downsample_factor, dataset_name=dataset_name,
    )
    print("Pyramid complete.")


# ---------------------------------------------------------------------------
# Neuroglancer helpers
# ---------------------------------------------------------------------------

def generate_neuroglancer_link(
    zarr_path: str,
    layer_name: str = None,
    viewer_url: str = "https://neuroglancer-demo.appspot.com/",
    serve_base_url: str = "https://s3.janelia.org/hess-lab/FIBSEM",
    display_axes_order: list = None,   # e.g. ["x", "y", "z"]
) -> str:
    """
    Generate a Neuroglancer link for a local or remote OME-ZARR store.

    Parameters
    ----------
    zarr_path       : local path to the .zarr directory
    layer_name      : display name for the Neuroglancer layer (default: zarr filename)
    viewer_url      : Neuroglancer viewer URL
    serve_base_url  : base URL under which the zarr file is served, e.g.
                      "https://s3.janelia.org/hess-lab/FIBSEM" (default) or
                      "http://localhost:9000" for local serving
    display_axes_order : list with axes order, e.g. ["x", "y", "z"]. Default is None (Z-Y-X, matching storage order).

    Returns
    -------
    str — the full Neuroglancer URL
    """
    name = os.path.basename(zarr_path.rstrip("/\\"))
    if layer_name is None:
        layer_name = name
    # Neuroglancer pipe syntax: <http-url-to-zarr-root>/|zarr2:
    # The zarr2 driver receives the root store URL and reads the OME-NGFF
    # multiscales metadata directly from the store root.
    source_url = f"{serve_base_url.rstrip('/')}/{name}/"
    layer_config = {
        # layers must be a JSON array, not an object
        "layers": [
            {
                "type":   "image",
                "source": f"{source_url}|zarr2:",
                "tab":    "source",
                "name":   layer_name,
            }
        ],
        "selectedLayer": {"visible": True, "layer": layer_name},
        "layout": "4panel-alt",
    }
    if display_axes_order is not None:
        layer_config["displayDimensions"] = display_axes_order
    encoded = urllib.parse.quote(json.dumps(layer_config))
    return f"{viewer_url}#!{encoded}"


def _print_neuroglancer_info(
    zarr_path: str,
    serve_base_url: str = "https://s3.janelia.org/hess-lab/FIBSEM",
    layer_name: str = None,
    viewer_url: str = "https://neuroglancer-demo.appspot.com/",
    display_axes_order: list = None,
):
    name   = os.path.basename(zarr_path.rstrip("/\\"))
    parent = os.path.dirname(os.path.abspath(zarr_path))
    link   = generate_neuroglancer_link(
        zarr_path, layer_name=layer_name,
        viewer_url=viewer_url, serve_base_url=serve_base_url,
        display_axes_order=display_axes_order,
    )
    print("\n" + "=" * 60)
    print("Neuroglancer — how to view")
    print("=" * 60)
    print(f"  Serve:  python -m http.server 9000 --directory {parent}")
    print(f"  Source: {serve_base_url.rstrip('/')}/{name}/|zarr2:")
    print(f"\n  Link:   {link}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# OME-ZARR v2 → v3 converter
# ---------------------------------------------------------------------------

def _convert_shard_worker(
    src_path: str,
    dst_path: str,
    arr_path: str,
    src_slices: tuple,
    dst_slices: tuple,
    perm,           # None  or  list[int] — axis permutation src→dst
) -> dict:
    """
    Worker executed on a DASK node (or locally).
    Reads one shard-sized region from a zarr v2 array and writes it
    to the corresponding region in the zarr v3 array, optionally
    transposing the data.

    Parameters
    ----------
    src_path   : path to the source zarr v2 root store
    dst_path   : path to the destination zarr v3 root store
    arr_path   : relative path to the array inside both stores (e.g. 's0')
    src_slices : tuple of slice objects — region to READ  (source axis order)
    dst_slices : tuple of slice objects — region to WRITE (dest   axis order)
    perm       : None → no transpose; list[int] → axes permutation (np.transpose)

    Returns
    -------
    dict  {'arr_path', 'dst_slices', 'nbytes', 'elapsed_s'}
    """
    import time
    import numpy as np
    import zarr

    t0 = time.time()

    src_grp = zarr.open_group(str(src_path), mode='r')
    dst_grp = zarr.open_group(str(dst_path), mode='r+')

    data = np.asarray(src_grp[arr_path][src_slices])   # read shard region

    if perm is not None:
        data = data.transpose(perm)

    dst_grp[arr_path][dst_slices] = data

    elapsed = time.time() - t0
    return {
        'arr_path' : arr_path,
        'dst_slices': str(dst_slices),
        'nbytes'   : data.nbytes,
        'elapsed_s': elapsed,
    }


def convert_ome_zarr_v2_to_v3(
    src_path: str,
    dst_path: str,
    client=None,
    chunk_size: tuple = (32, 32, 32),
    shard_size: tuple = (1024, 1024, 1024),
    axis_order: str = 'xyz',
    transpose_codec: bool = True,
    compression: str = 'zstd',
    compression_level: int = 3,
    overwrite: bool = True,
    DASK_client_retries: int = 3,
    verbose: bool = True,
) -> zarr.Group:
    """
    Convert an OME-ZARR v2 store to ZARR v3 format using DASK.
    ©G.Shtengel 2026  gleb.shtengel@gmail.com

    All pyramid levels (s0, s1, …) present in the source are converted.
    Supports optional axis reordering (e.g. ZYX source → XYZ output),
    zarr v3 sharding, TransposeCodec (F-order), and configurable compression.

    Requires zarr >= 3.0.

    Parameters
    ----------
    src_path         : Path to the source OME-ZARR v2 store.
    dst_path         : Path for the new ZARR v3 store (created fresh).
    client           : dask.distributed.Client or None (local execution).
    chunk_size       : Inner (logical) chunk shape in *output* axis order.
                       Default (32, 32, 32).
    shard_size       : Shard (physical file) shape in *output* axis order.
                       Must be an integer multiple of chunk_size in every
                       dimension.  Default (1024, 1024, 1024).
    axis_order       : Axis order of the OUTPUT store, e.g. 'xyz' or 'zyx'.
                       The source axis order is read from the OME-ZARR
                       multiscales metadata; if it differs, data are
                       transposed and coordinate-transform values reordered.
                       Default 'xyz'.
    transpose_codec  : If True, adds a TransposeCodec (F-order, i.e. axes
                       reversed) to the inner codec chain of the zarr v3
                       sharded array.  This stores bytes in column-major
                       order inside each inner chunk for efficient random
                       access along the first output axis.  Default True.
    compression      : Compression codec name for inner chunks.
                       'zstd' (default), 'blosc', 'gzip', or None.
    compression_level: Compression level (codec-dependent).  Default 3.
    overwrite        : Overwrite dst_path if it already exists.  Default True.
    DASK_client_retries : Retry count for failed DASK tasks.  Default 3.
    verbose          : Print progress.  Default True.

    Returns
    -------
    zarr.Group
        The opened (and fully populated) destination v3 root store.

    Notes
    -----
    Sharding is a zarr v3 feature: each *shard* is a single file on disk that
    contains (shard_size / chunk_size) inner chunks in each dimension.  With
    shard_size=(1024,1024,1024) and chunk_size=(32,32,32) each shard file holds
    32³ = 32 768 inner chunks, reducing the number of on-disk files by ~33 000×
    compared to an un-sharded store with the same inner-chunk size.

    The task granularity for DASK is one *shard* per submitted task, keeping
    the task graph small regardless of the total number of inner chunks.
    """
    # ------------------------------------------------------------------ #
    # 0.  Imports — zarr ≥ 3.0 required                                   #
    # ------------------------------------------------------------------ #
    import math
    from dask.distributed import as_completed as dask_as_completed

    # Build the zarr v3 inner-codec pipeline
    def _make_v3_compressor():
        if compression == 'zstd':
            from zarr.codecs import ZstdCodec
            return ZstdCodec(level=compression_level)
        if compression in ('blosc', 'lz4'):
            from zarr.codecs import BloscCodec
            return BloscCodec(cname='lz4', clevel=compression_level,
                              shuffle='bitshuffle')
        if compression == 'gzip':
            from zarr.codecs import GzipCodec
            return GzipCodec(level=compression_level)
        return None   # no compression

    # ------------------------------------------------------------------ #
    # 1.  Open source (zarr 3.x reads v2 natively)                        #
    # ------------------------------------------------------------------ #
    if verbose:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')
              + '   Opening source OME-ZARR v2 store: ' + str(src_path))
    src = zarr.open_group(str(src_path), mode='r')
    src_attrs = dict(src.attrs)

    # ------------------------------------------------------------------ #
    # 2.  Determine axis ordering and permutation                          #
    # ------------------------------------------------------------------ #
    # Source axis order from OME-ZARR metadata (fall back to 'zyx')
    src_axis_order = 'zyx'
    if 'multiscales' in src_attrs:
        axes = src_attrs['multiscales'][0].get('axes', [])
        if axes:
            src_axis_order = ''.join(ax['name'] for ax in axes)

    dst_axis_order = axis_order.lower()
    if len(dst_axis_order) != len(src_axis_order):
        raise ValueError(
            f'axis_order "{dst_axis_order}" has {len(dst_axis_order)} axes but '
            f'source has {len(src_axis_order)} axes ({src_axis_order}).')

    # perm[i] = which source axis becomes output axis i
    # e.g.  src='zyx', dst='xyz'  →  perm=[2,1,0]
    perm = None
    if dst_axis_order != src_axis_order:
        try:
            perm = [src_axis_order.index(ax) for ax in dst_axis_order]
        except ValueError as exc:
            raise ValueError(
                f'axis_order "{dst_axis_order}" contains axes not present in '
                f'source "{src_axis_order}": {exc}') from exc
    ndim = len(src_axis_order)

    if verbose:
        print(f'  Source axis order : {src_axis_order}')
        print(f'  Output axis order : {dst_axis_order}')
        if perm is not None:
            print(f'  Transposition perm: {perm}')
        else:
            print('  No transposition needed.')

    # ------------------------------------------------------------------ #
    # 3.  Walk source to collect arrays                                    #
    # ------------------------------------------------------------------ #
    arrays_info = []   # [(relative_path, zarr.Array), ...]

    def _walk(grp, prefix):
        for key in sorted(grp.keys()):
            child = grp[key]
            child_path = f'{prefix}/{key}' if prefix else key
            if isinstance(child, zarr.Array):
                arrays_info.append((child_path, child))
            elif isinstance(child, zarr.Group):
                _walk(child, child_path)

    _walk(src, '')

    if verbose:
        print(f'  Found {len(arrays_info)} array(s):')
        for p, a in arrays_info:
            out_shape = tuple(a.shape[i] for i in perm) if perm else a.shape
            print(f'    [{p}]  src_shape={a.shape}  dst_shape={out_shape}'
                  f'  dtype={a.dtype}')

    # ------------------------------------------------------------------ #
    # 4.  Build updated OME-ZARR metadata                                  #
    # ------------------------------------------------------------------ #
    dst_attrs = dict(src_attrs)   # start from a copy

    if perm is not None and 'multiscales' in src_attrs:
        ms = [dict(m) for m in src_attrs['multiscales']]
        for m in ms:
            # Reorder axes list
            if 'axes' in m:
                m['axes'] = [m['axes'][i] for i in perm]
            # Reorder coordinate-transform values in every dataset entry
            if 'datasets' in m:
                new_datasets = []
                for ds in m['datasets']:
                    new_ds = dict(ds)
                    new_cts = []
                    for ct in ds.get('coordinateTransformations', []):
                        new_ct = dict(ct)
                        for key in ('scale', 'translation'):
                            if key in new_ct:
                                new_ct[key] = [new_ct[key][i] for i in perm]
                        new_cts.append(new_ct)
                    new_ds['coordinateTransformations'] = new_cts
                    new_datasets.append(new_ds)
                m['datasets'] = new_datasets
            # Reorder top-level coordinateTransformations if present
            if 'coordinateTransformations' in m:
                new_cts = []
                for ct in m['coordinateTransformations']:
                    new_ct = dict(ct)
                    for key in ('scale', 'translation'):
                        if key in new_ct:
                            new_ct[key] = [new_ct[key][i] for i in perm]
                    new_cts.append(new_ct)
                m['coordinateTransformations'] = new_cts
        dst_attrs['multiscales'] = ms

    # ------------------------------------------------------------------ #
    # 5.  Create destination zarr v3 store                                 #
    # ------------------------------------------------------------------ #
    if verbose:
        print(time.strftime('%Y/%m/%d  %H:%M:%S')
              + '   Creating destination ZARR v3 store: ' + str(dst_path))

    if overwrite and os.path.exists(str(dst_path)):
        shutil.rmtree(str(dst_path))

    dst = zarr.open_group(str(dst_path), mode='w', zarr_format=3)
    dst.attrs.update(dst_attrs)
    if verbose:
        print('  Root OME-ZARR metadata written.')

    # Prepare the zarr v3 compressor object (built once, reused per level)
    v3_compressor = _make_v3_compressor()
    v3_order = 'F' if transpose_codec else 'C'

    # ------------------------------------------------------------------ #
    # 6.  Pre-allocate all destination arrays                              #
    # ------------------------------------------------------------------ #
    for arr_path, src_arr in arrays_info:
        src_shape = src_arr.shape   # in source axis order

        # Output shape after optional axis transposition
        dst_shape = tuple(src_shape[i] for i in perm) if perm else src_shape

        # Clamp chunk / shard sizes to array dimensions
        use_chunks = tuple(min(chunk_size[i], dst_shape[i]) for i in range(ndim))
        use_shards = tuple(min(shard_size[i], dst_shape[i]) for i in range(ndim))

        # Navigate / create parent groups in dst
        parts      = arr_path.split('/')
        dst_parent = dst
        for part in parts[:-1]:
            dst_parent = dst_parent.require_group(part)

        cmp_kwargs = {}
        if v3_compressor is not None:
            cmp_kwargs['compressors'] = [v3_compressor]

        dst_parent.create_array(
            name   = parts[-1],
            shape  = dst_shape,
            dtype  = src_arr.dtype,
            chunks = use_chunks,
            shards = use_shards,
            order  = v3_order,
            fill_value = 0,
            overwrite  = True,
            **cmp_kwargs,
        )

        if verbose:
            print(f'  Pre-allocated [{arr_path}]  dst_shape={dst_shape}'
                  f'  chunks={use_chunks}  shards={use_shards}'
                  f'  order={v3_order}  compression={compression}')

        # Copy array-level attributes
        arr_attrs = dict(src_arr.attrs)
        if arr_attrs:
            dst[arr_path].attrs.update(arr_attrs)

    # Copy attributes on intermediate groups
    def _copy_group_attrs(src_grp, dst_grp_local):
        for key in src_grp.keys():
            child = src_grp[key]
            if isinstance(child, zarr.Group):
                dst_child = dst_grp_local.require_group(key)
                g_attrs   = dict(child.attrs)
                if g_attrs:
                    dst_child.attrs.update(g_attrs)
                _copy_group_attrs(child, dst_child)

    _copy_group_attrs(src, dst)

    # ------------------------------------------------------------------ #
    # 7.  Transfer data — one DASK task per output shard                   #
    # ------------------------------------------------------------------ #
    for arr_path, src_arr in arrays_info:
        src_shape = src_arr.shape
        dst_shape = tuple(src_shape[i] for i in perm) if perm else src_shape

        use_chunks = tuple(min(chunk_size[i], dst_shape[i]) for i in range(ndim))
        use_shards = tuple(min(shard_size[i], dst_shape[i]) for i in range(ndim))

        # Build list of all shard regions (output coordinates)
        shard_starts = [
            list(range(0, dst_shape[d], use_shards[d]))
            for d in range(ndim)
        ]
        import itertools
        shard_grid = list(itertools.product(*shard_starts))
        n_shards   = len(shard_grid)

        if verbose:
            print(f'\n{time.strftime("%Y/%m/%d  %H:%M:%S")}'
                  f'   [{arr_path}]  {n_shards} shard(s) to write …')

        t0 = time.time()

        def _make_shard_params(shard_origin):
            # dst_slices: region in the output array (destination axis order)
            dst_slices = tuple(
                slice(shard_origin[d],
                      min(shard_origin[d] + use_shards[d], dst_shape[d]))
                for d in range(ndim)
            )
            # src_slices: corresponding region in the source array (source axis order)
            # perm[d] = source dim for output dim d  →  inverse: iperm[s] = output dim for source dim s
            if perm is not None:
                iperm = [0] * ndim
                for d, s in enumerate(perm):
                    iperm[s] = d
                src_slices = tuple(dst_slices[iperm[s]] for s in range(ndim))
            else:
                src_slices = dst_slices
            return src_slices, dst_slices

        params_list = [_make_shard_params(origin) for origin in shard_grid]

        if client is not None:
            futures = [
                client.submit(
                    _convert_shard_worker,
                    str(src_path), str(dst_path), arr_path,
                    src_sl, dst_sl, perm,
                    pure=False, retries=DASK_client_retries,
                )
                for src_sl, dst_sl in params_list
            ]
            n_done   = 0
            n_failed = 0
            report_every = max(1, n_shards // 20)
            for fut in dask_as_completed(futures):
                try:
                    fut.result()
                    n_done += 1
                except Exception as exc:
                    n_failed += 1
                    if verbose:
                        print(f'    WARN: shard failed — {exc}')
                if verbose and n_done % report_every == 0:
                    elapsed = time.time() - t0
                    print(f'    {n_done}/{n_shards} shards done'
                          f'  ({elapsed:.0f} s elapsed)')
        else:
            # Local (sequential) fallback
            for k, (src_sl, dst_sl) in enumerate(params_list):
                _convert_shard_worker(
                    str(src_path), str(dst_path), arr_path,
                    src_sl, dst_sl, perm,
                )
                if verbose and (k + 1) % max(1, n_shards // 20) == 0:
                    print(f'    {k+1}/{n_shards} shards done')

        if verbose:
            elapsed = time.time() - t0
            total_bytes = (np.prod(dst_shape)
                           * np.dtype(src_arr.dtype).itemsize)
            print(f'  [{arr_path}] done in {elapsed:.1f} s'
                  f'  ({total_bytes / elapsed / 1e9:.2f} GB/s)')

    if verbose:
        print(f'\n{time.strftime("%Y/%m/%d  %H:%M:%S")}'
              f'   Conversion complete → {dst_path}')

    return dst


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def tif_stack_to_zarr(
    tif_files: list,
    output_zarr_path: str,
    client=None,
    chunk_z: int = 64,
    chunk_y: int = 128,
    chunk_x: int = 128,
    n_pyramid_levels: int = 4,
    downsample_factor: int = 2,
    voxel_size_zyx: tuple = (8.0, 8.0, 8.0),
    origin_zyx: tuple = (0.0, 0.0, 0.0),
    voxel_unit: str = "nanometer",
    zarr_compressor: str = "blosc",
    zarr_compressor_level: int = 1,
    n_read_threads: int = 8,
    strip_y: int = 1024,
    dataset_name: str = "volume",
    overwrite: bool = True,
    DASK_client_retries: int = 3,
    neuroglancer_serve_base_url: str = "https://s3.janelia.org/hess-lab/FIBSEM",
    neuroglancer_viewer_url: str = "https://neuroglancer-demo.appspot.com/",
    neuroglancer_display_axes_order: list = None,   # e.g. ["x","y","z"], default None = Z-Y-X
):
    """
    Convert a list of TIF files to OME-ZARR, parallelised via a Dask client.

    Writes OME-NGFF v0.4 multiscales metadata (paths 's0', 's1', …) with
    per-level scale + translation coordinate transformations, compatible with
    Neuroglancer (zarr2:// source).

    Parameters
    ----------
    tif_files        : ordered list of TIF file paths (index = Z position)
    output_zarr_path      : output .zarr path
    client           : dask.distributed.Client, or None for sequential execution.
                       Create however you like — LocalCluster, LSFCluster, SSH, …
    chunk_z/y/x      : zarr chunk dimensions (Z, Y, X)
    n_pyramid_levels : number of resolution levels including full-res (≥1)
    downsample_factor: factor between pyramid levels
    voxel_size_zyx   : physical voxel size (z, y, x) in voxel_unit
    origin_zyx       : physical origin (z, y, x) of voxel [0,0,0] in voxel_unit
    voxel_unit       : e.g. "nanometer", "micrometer"
    zarr_compressor  : "blosc" (default, fastest), "gzip", or None
    zarr_compressor_level : 1 (fastest) … 9 (best ratio)
    n_read_threads   : threads per Dask worker for parallel TIF reading;
                       rule of thumb: min(n_cores_per_worker, worker_RAM_GB // 10)
    strip_y          : Y-strip height in pixels. Each worker reads only strip_y
                       rows from each TIF via tifffile.memmap, so per-worker RAM is
                       chunk_z × strip_y × nx × itemsize.
                       Requires uncompressed TIF files.
                       Rule of thumb: strip_y = worker_RAM_GB * 1e9 // (chunk_z * nx * itemsize)
    dataset_name     : cosmetic name in OME metadata
    overwrite        : overwrite existing store
    DASK_client_retries : number of times a failed slab or pyramid-level task is
                       automatically re-submitted before raising an error (default 3).
                       Applies only when client is not None.  Each retry re-submits
                       the exact same work unit to a (potentially different) worker,
                       which recovers from transient worker crashes or I/O errors.
    neuroglancer_serve_base_url : base URL used when printing the Neuroglancer link
    neuroglancer_viewer_url     : Neuroglancer viewer URL for link generation
    neuroglancer_display_axes_order : list, axes order for Neuroglancer display e.g. ["x","y","z"].
                                  Default is None (Z-Y-X, matching the OME-ZARR storage order).

    Returns
    -------
    dict  {output_zarr_path, shape, dtype, chunks, n_levels, elapsed_s, neuroglancer_link}

    Examples
    --------
    # With a pre-existing Dask client (LSF, SSH, local, …)
    result = tif_stack_to_zarr(my_tif_list, "/data/out.zarr", client=client,
                               voxel_size_zyx=(8., 8., 8.))

    # Sequential (no client)
    result = tif_stack_to_zarr(my_tif_list, "/data/out.zarr", client=None)

    # Print Neuroglancer link
    print(result['neuroglancer_link'])
    """
    use_DASK, status_update_address = check_DASK(client)
    t0 = time.time()

    if not tif_files:
        raise ValueError("tif_files is empty")

    nz = len(tif_files)

    # Probe first file
    probe = _read_tif(tif_files[0])
    ny, nx = probe.shape
    dtype = probe.dtype
    del probe

    n_slabs = math.ceil(nz / chunk_z)
    sy = min(strip_y, ny)
    n_strips_per_slab = math.ceil(ny / sy)
    n_tasks = n_slabs * n_strips_per_slab
    strip_gb = chunk_z * sy * nx * dtype.itemsize / 1e9

    print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Starting tif_stack_to_zarr')
    print(f"Volume (Z, Y, X)  : ({nz}, {ny}, {nx})")
    print(f"dtype             : {dtype}")
    print(f"Chunks (Z, Y, X)  : ({chunk_z}, {chunk_y}, {chunk_x})")
    print(f"Raw size          : {nz*ny*nx*dtype.itemsize/1e12:.2f} TB")
    print(f"Voxel size (Z,Y,X): {voxel_size_zyx}  [{voxel_unit}]")
    print(f"Origin     (Z,Y,X): {origin_zyx}  [{voxel_unit}]")
    print(f"Mode              : memmap strip  (strip_y={sy})")
    print(f"Slabs             : {n_slabs}")
    print(f"Strips per slab   : {n_strips_per_slab}  ({n_tasks} tasks total)")
    print(f"RAM per task      : {strip_gb:.2f} GB")
    if client is not None:
        info = client.scheduler_info()
        n_workers = len(info.get("workers", {}))
        print(f"Dask workers      : {n_workers}")
        print(f"Dask retries      : {DASK_client_retries}")

    # Create zarr store
    print(f"\nCreating zarr store: {output_zarr_path}")
    create_zarr_store(
        output_zarr_path=output_zarr_path, nz=nz, ny=ny, nx=nx, dtype=dtype,
        chunk_z=chunk_z, chunk_y=chunk_y, chunk_x=chunk_x,
        n_pyramid_levels=n_pyramid_levels, downsample_factor=downsample_factor,
        zarr_compressor=zarr_compressor, zarr_compressor_level=zarr_compressor_level,
        voxel_size_zyx=voxel_size_zyx, origin_zyx=origin_zyx,
        voxel_unit=voxel_unit,
        dataset_name=dataset_name, overwrite=overwrite,
    )

    # Build slab list
    slabs = [
        (slab_idx * chunk_z, min((slab_idx + 1) * chunk_z, nz))
        for slab_idx in range(n_slabs)
    ]

    # Build task list (one task per (Z-slab, Y-strip) pair)
    tasks = [
        (z_start, z_end, y_start, min(y_start + sy, ny))
        for (z_start, z_end) in slabs
        for y_start in range(0, ny, sy)
    ]
    print()
    print(time.strftime('%Y/%m/%d  %H:%M:%S') +
          f'   Writing {n_tasks:,} strips to level s0 '
          f'({n_slabs} Z-slabs × {math.ceil(ny / sy)} Y-strips) …')

    speeds = []

    if client is not None:
        # client.submit() retries=N tells the Dask scheduler to automatically
        # re-run a task up to N times if it raises an exception, potentially
        # on a different worker each time.  Worker death is handled separately
        # and automatically by the scheduler regardless of this setting.
        futures = [
            client.submit(
                write_strip_to_zarr,
                output_zarr_path, tif_files, z_start, z_end, y_start, y_end,
                n_read_threads, 0,
                pure=False,
                retries=DASK_client_retries,
            )
            for z_start, z_end, y_start, y_end in tasks
        ]
        n_done = 0
        for fut in as_completed(futures):
            result = fut.result()   # raises only after all retries are exhausted
            speeds.append(result["mb_s"])
            n_done += 1
            elapsed = time.time() - t0
            eta_s = (elapsed / n_done) * (n_tasks - n_done) if n_done else 0
            print(time.strftime('%Y/%m/%d  %H:%M:%S') +
                  f'     [{n_done:d}/{n_tasks:d}]  '
                  f"z={result['z_start']}–{result['z_end']-1}  "
                  f"y={result['y_start']}–{result['y_end']-1}  "
                  f"{result['elapsed_s']:.1f}s  {result['mb_s']:.0f} MB/s  "
                  f"ETA {eta_s/60:.1f} min")
    else:
        # Sequential fallback
        for i, (z_start, z_end, y_start, y_end) in enumerate(tasks):
            result = write_strip_to_zarr(
                output_zarr_path, tif_files, z_start, z_end,
                y_start, y_end, n_read_threads, 0,
            )
            speeds.append(result["mb_s"])
            elapsed = time.time() - t0
            eta_s = (elapsed / (i + 1)) * (n_tasks - i - 1)
            print(f"  [{i+1}/{n_tasks}]  "
                  f"z={z_start}–{z_end-1}  y={y_start}–{y_end-1}  "
                  f"{result['elapsed_s']:.1f}s  {result['mb_s']:.0f} MB/s  "
                  f"ETA {eta_s/60:.1f} min")

    # Build pyramid from level s0
    if n_pyramid_levels > 1:
        print()
        print(time.strftime('%Y/%m/%d  %H:%M:%S') + '     Building downsampled pyramid levels …')
        finalize_pyramid(
            output_zarr_path=output_zarr_path, n_pyramid_levels=n_pyramid_levels,
            downsample_factor=downsample_factor,
            chunk_z=chunk_z, chunk_y=chunk_y, chunk_x=chunk_x,
            voxel_size_zyx=voxel_size_zyx, origin_zyx=origin_zyx,
            voxel_unit=voxel_unit,
            dataset_name=dataset_name, client=client,
            DASK_client_retries=DASK_client_retries,
        )

    elapsed = time.time() - t0
    avg_speed = np.mean(speeds) if speeds else 0.0
    print()
    print(time.strftime('%Y/%m/%d  %H:%M:%S') + f'     Done.  {elapsed/60:.1f} min total  avg {avg_speed:.0f} MB/s')

    ng_link = generate_neuroglancer_link(
        output_zarr_path,
        layer_name=dataset_name,
        viewer_url=neuroglancer_viewer_url,
        serve_base_url=neuroglancer_serve_base_url,
        display_axes_order = neuroglancer_display_axes_order,
    )
    _print_neuroglancer_info(
        output_zarr_path,
        serve_base_url=neuroglancer_serve_base_url,
        layer_name=dataset_name,
        viewer_url=neuroglancer_viewer_url,
        display_axes_order = neuroglancer_display_axes_order,
    )

    return {
        "output_zarr_path": output_zarr_path,
        "shape": (nz, ny, nx),
        "dtype": str(dtype),
        "chunks": (chunk_z, chunk_y, chunk_x),
        "n_levels": n_pyramid_levels,
        "elapsed_s": elapsed,
        "neuroglancer_link": ng_link,
    }
