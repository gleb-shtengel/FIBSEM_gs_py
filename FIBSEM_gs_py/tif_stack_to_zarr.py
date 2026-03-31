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
        output_zarr= "/data/out.zarr",
        client     = client,
        chunk_z=64, chunk_y=128, chunk_x=128,
        voxel_size_zyx=(8., 8., 8.),
        origin_zyx=(0., 0., 0.),       # physical origin in voxel_unit
    )

If client=None the work runs sequentially on the calling process (useful for
testing or when a single node is fast enough).

ARCHITECTURE
------------
Each Z-slab (chunk_z consecutive slices) is one independent unit of work:

    write_slab_to_zarr(output_zarr, tif_files, z_start, z_end)
        1. Reads chunk_z TIF files in parallel (ThreadPoolExecutor)
        2. Writes the slab to zarr['s0'][z_start:z_end, :, :]
           zarr handles XY chunking internally

Each TIF is read exactly ONCE.  No Dask rechunking is involved.
Multiple workers can write to the same zarr store simultaneously because
zarr chunks are independent files and each job writes a non-overlapping Z range.

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
  http://<host>:<port>/path/to/output.zarr/|zarr2:s0/

©G.Shtengel 2026  gleb.shtengel@gmail.com
"""

import json
import os
import math
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


def _read_slab_parallel(tif_files: list, n_threads: int = 8) -> np.ndarray:
    """Read a list of TIF files in parallel and stack into (nz, ny, nx)."""
    n = len(tif_files)
    results = [None] * n
    with ThreadPoolExecutor(max_workers=min(n_threads, n)) as pool:
        futures = {pool.submit(_read_tif, f): i for i, f in enumerate(tif_files)}
        for fut in thread_as_completed(futures):
            results[futures[fut]] = fut.result()
    return np.stack(results, axis=0)


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
            "type": "gaussian" if n_levels > 1 else "none",
        }]
    })


# ---------------------------------------------------------------------------
# zarr store creation
# ---------------------------------------------------------------------------

def create_zarr_store(
    output_zarr: str,
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
    root = zarr.open_group(output_zarr, mode="w" if overwrite else "w-")

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
# Slab writer  — the unit of work executed on each Dask worker
# ---------------------------------------------------------------------------

def write_slab_to_zarr(
    output_zarr: str,
    tif_files: list,
    z_start: int,
    z_end: int,
    n_read_threads: int = 8,
    level: int = 0,
) -> dict:
    """
    Read TIF files [z_start, z_end) and write them into zarr level `level`.

    Designed to run on a Dask worker (or directly).  Fully self-contained:
    opens the zarr store itself, reads TIFs in parallel via threads, writes
    the slab in one call (zarr handles XY chunking internally).

    Parameters
    ----------
    output_zarr    : path to existing .zarr store
    tif_files      : full ordered list of TIF paths for the entire volume
    z_start, z_end : half-open Z-slice range for this slab
    n_read_threads : threads for parallel TIF reading within this worker
    level          : pyramid level to write into (0 = full resolution)

    Returns
    -------
    dict  {z_start, z_end, elapsed_s, mb_s}
    """
    import os, time
    import numpy as np
    import zarr

    try:
        import tifffile as _tiff
    except ImportError:
        import skimage.external.tifffile as _tiff

    def _read(path):
        img = _tiff.imread(os.path.normpath(path))
        return img[..., 0] if img.ndim == 3 else img

    t0 = time.time()
    slab_files = tif_files[z_start:z_end]
    if not slab_files:
        return {"z_start": z_start, "z_end": z_end, "elapsed_s": 0.0, "mb_s": 0.0}

    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
    n = len(slab_files)
    results = [None] * n
    with ThreadPoolExecutor(max_workers=min(n_read_threads, n)) as pool:
        futs = {pool.submit(_read, f): i for i, f in enumerate(slab_files)}
        for fut in _as_completed(futs):
            results[futs[fut]] = fut.result()
    slab = np.stack(results, axis=0)

    zarr.open(output_zarr, mode="r+")[f"s{level}"][z_start:z_end, :, :] = slab

    elapsed = time.time() - t0
    mb = slab.nbytes / 1e6
    del slab
    return {"z_start": z_start, "z_end": z_end, "elapsed_s": elapsed, "mb_s": mb / elapsed}


# ---------------------------------------------------------------------------
# Pyramid chunk worker  — one unit of work per output zarr chunk
# ---------------------------------------------------------------------------

def _write_pyramid_chunk(
    output_zarr: str,
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

    root = zarr.open_group(output_zarr, mode="r+")
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
# Pyramid builder
# ---------------------------------------------------------------------------

def finalize_pyramid(
    output_zarr: str,
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
    root = zarr.open_group(output_zarr, mode="a")

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
                    output_zarr, level - 1, level,
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
                    output_zarr, level - 1, level,
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
    encoded = urllib.parse.quote(json.dumps(layer_config))
    return f"{viewer_url}#!{encoded}"


def _print_neuroglancer_info(
    zarr_path: str,
    serve_base_url: str = "https://s3.janelia.org/hess-lab/FIBSEM",
    layer_name: str = None,
    viewer_url: str = "https://neuroglancer-demo.appspot.com/",
):
    name   = os.path.basename(zarr_path.rstrip("/\\"))
    parent = os.path.dirname(os.path.abspath(zarr_path))
    link   = generate_neuroglancer_link(
        zarr_path, layer_name=layer_name,
        viewer_url=viewer_url, serve_base_url=serve_base_url,
    )
    print("\n" + "=" * 60)
    print("Neuroglancer — how to view")
    print("=" * 60)
    print(f"  Serve:  python -m http.server 9000 --directory {parent}")
    print(f"  Source: {serve_base_url.rstrip('/')}/{name}/|zarr2:")
    print(f"\n  Link:   {link}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def tif_stack_to_zarr(
    tif_files: list,
    output_zarr: str,
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
    dataset_name: str = "volume",
    overwrite: bool = True,
    DASK_client_retries: int = 3,
    neuroglancer_serve_base_url: str = "https://s3.janelia.org/hess-lab/FIBSEM",
    neuroglancer_viewer_url: str = "https://neuroglancer-demo.appspot.com/",
):
    """
    Convert a list of TIF files to OME-ZARR, parallelised via a Dask client.

    Writes OME-NGFF v0.4 multiscales metadata (paths 's0', 's1', …) with
    per-level scale + translation coordinate transformations, compatible with
    Neuroglancer (zarr2:// source).

    Parameters
    ----------
    tif_files        : ordered list of TIF file paths (index = Z position)
    output_zarr      : output .zarr path
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
    dataset_name     : cosmetic name in OME metadata
    overwrite        : overwrite existing store
    DASK_client_retries : number of times a failed slab or pyramid-level task is
                       automatically re-submitted before raising an error (default 3).
                       Applies only when client is not None.  Each retry re-submits
                       the exact same work unit to a (potentially different) worker,
                       which recovers from transient worker crashes or I/O errors.
    neuroglancer_serve_base_url : base URL used when printing the Neuroglancer link
    neuroglancer_viewer_url     : Neuroglancer viewer URL for link generation

    Returns
    -------
    dict  {output_zarr, shape, dtype, chunks, n_levels, elapsed_s, neuroglancer_link}

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
    slab_gb = chunk_z * ny * nx * dtype.itemsize / 1e9

    print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Starting tif_stack_to_zarr')
    print(f"Volume (Z, Y, X)  : ({nz}, {ny}, {nx})")
    print(f"dtype             : {dtype}")
    print(f"Chunks (Z, Y, X)  : ({chunk_z}, {chunk_y}, {chunk_x})")
    print(f"Raw size          : {nz*ny*nx*dtype.itemsize/1e12:.2f} TB")
    print(f"Voxel size (Z,Y,X): {voxel_size_zyx}  [{voxel_unit}]")
    print(f"Origin     (Z,Y,X): {origin_zyx}  [{voxel_unit}]")
    print(f"Slabs             : {n_slabs}  ({slab_gb:.1f} GB RAM each)")
    if client is not None:
        info = client.scheduler_info()
        n_workers = len(info.get("workers", {}))
        print(f"Dask workers      : {n_workers}")
        print(f"Dask retries      : {DASK_client_retries}")

    # Create zarr store
    print(f"\nCreating zarr store: {output_zarr}")
    create_zarr_store(
        output_zarr=output_zarr, nz=nz, ny=ny, nx=nx, dtype=dtype,
        chunk_z=chunk_z, chunk_y=chunk_y, chunk_x=chunk_x,
        n_pyramid_levels=n_pyramid_levels, downsample_factor=downsample_factor,
        zarr_compressor=zarr_compressor, zarr_compressor_level=zarr_compressor_level,
        voxel_size_zyx=voxel_size_zyx, origin_zyx=origin_zyx,
        voxel_unit=voxel_unit,
        dataset_name=dataset_name, overwrite=overwrite,
    )

    # Build slab parameter list
    slabs = [
        (slab_idx * chunk_z, min((slab_idx + 1) * chunk_z, nz))
        for slab_idx in range(n_slabs)
    ]

    print()
    print(time.strftime('%Y/%m/%d  %H:%M:%S') + '   Writing {:d} slabs to level s0 …'.format(n_slabs))
    speeds = []

    if client is not None:
        # client.submit() retries=N tells the Dask scheduler to automatically
        # re-run a task up to N times if it raises an exception, potentially
        # on a different worker each time.  Worker death is handled separately
        # and automatically by the scheduler regardless of this setting.
        futures = [
            client.submit(
                write_slab_to_zarr,
                output_zarr, tif_files, z_start, z_end,
                n_read_threads, 0,
                pure=False,
                retries=DASK_client_retries,
            )
            for z_start, z_end in slabs
        ]
        n_done = 0
        for fut in as_completed(futures):
            result = fut.result()   # raises only after all retries are exhausted
            speeds.append(result["mb_s"])
            n_done += 1
            elapsed = time.time() - t0
            eta_s = (elapsed / n_done) * (n_slabs - n_done) if n_done else 0
            print(time.strftime('%Y/%m/%d  %H:%M:%S') + '     [{:d}/{:d}]  '.format(n_done, n_slabs),
                  f"z={result['z_start']}–{result['z_end']-1}  "
                  f"{result['elapsed_s']:.1f}s  {result['mb_s']:.0f} MB/s  "
                  f"ETA {eta_s/60:.1f} min")
    else:
        # Sequential fallback
        for i, (z_start, z_end) in enumerate(slabs):
            result = write_slab_to_zarr(
                output_zarr, tif_files, z_start, z_end, n_read_threads, 0
            )
            speeds.append(result["mb_s"])
            elapsed = time.time() - t0
            eta_s = (elapsed / (i + 1)) * (n_slabs - i - 1)
            print(f"  [{i+1}/{n_slabs}]  z={z_start}–{z_end-1}  "
                  f"{result['elapsed_s']:.1f}s  {result['mb_s']:.0f} MB/s  "
                  f"ETA {eta_s/60:.1f} min")

    # Build pyramid from level s0
    if n_pyramid_levels > 1:
        print()
        print(time.strftime('%Y/%m/%d  %H:%M:%S') + '     Building downsampled pyramid levels …')
        finalize_pyramid(
            output_zarr=output_zarr, n_pyramid_levels=n_pyramid_levels,
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
        output_zarr,
        layer_name=dataset_name,
        viewer_url=neuroglancer_viewer_url,
        serve_base_url=neuroglancer_serve_base_url,
    )
    _print_neuroglancer_info(
        output_zarr,
        serve_base_url=neuroglancer_serve_base_url,
        layer_name=dataset_name,
        viewer_url=neuroglancer_viewer_url,
    )

    return {
        "output_zarr": output_zarr,
        "shape": (nz, ny, nx),
        "dtype": str(dtype),
        "chunks": (chunk_z, chunk_y, chunk_x),
        "n_levels": n_pyramid_levels,
        "elapsed_s": elapsed,
        "neuroglancer_link": ng_link,
    }
