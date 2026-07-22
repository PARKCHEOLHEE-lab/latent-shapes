"""Coarse-to-fine streaming reconstruction for the local FastAPI demo.

Ports docs/js/latent_backend.js (localReconstruct) to the Python backend so the
demo can stream a mesh coarse-to-fine (matching the static ONNX demo) instead of
returning one dense mesh in a single response. Reuses the trained decoder; the
dense model.py reconstruct() stays the equivalence oracle.
"""


import numpy as np
from scipy import ndimage
from skimage import measure

# Fixed sampling bounds + iso-level from latent_shapes/src/config.py (also mirrored in
# docs/js/latent_backend.js BOUNDS). Decoder query points come from these; the mesh is
# rescaled to the latent cage, not to these bounds (see _postprocess / model.py:226-238).
BOUNDS = (
    (-0.6149350, -0.767495, -0.666214),
    (0.644344, 0.792882, 0.581235),
)
MC_LEVEL = 0.0

# Points evaluated per decoder call: bounds peak input memory and lets the final level
# stream a partial preview after each chunk. (latent_backend.js CHUNK_ROWS)
CHUNK_ROWS = 65536

# A cell is "near the surface" if its corners change sign or the smallest |sdf| is
# under tau = ACTIVE_TAU_CELLS * cell_size. (latent_backend.js:130, calibrated on the
# real decoder: far-field |sdf| is 0.3-0.7, so 2.5x cell size is a safe margin.)
ACTIVE_TAU_CELLS = 2.5


def levels_for(resolution, max_levels=6, min_level_res=16):
    # coarse-to-fine ladder: prepend resolution/2 (finest stays last) while it is
    # still even and >= min_level_res, up to max_levels entries. (latent_backend.js:118)
    levels = [resolution]
    while len(levels) < max_levels and levels[0] % 2 == 0 and levels[0] // 2 >= min_level_res:
        levels.insert(0, levels[0] // 2)
    return levels


def build_next_mask(field, R, next_R, bounds):
    # Mark the next level's cells to refine: this level's near-surface cells, dilated
    # by one cell, mapped onto the finer cell grid. (latent_backend.js:134)
    mins, maxs = bounds
    C = R - 1
    next_c = next_R - 1
    f = np.asarray(field, dtype=np.float64).reshape(R, R, R)
    cell_size = max((maxs[a] - mins[a]) / C for a in range(3))
    tau = ACTIVE_TAU_CELLS * cell_size

    # the 8 corners of every cell -> (8, C, C, C)
    corners = np.stack([
        f[di:di + C, dj:dj + C, dk:dk + C]
        for di in (0, 1) for dj in (0, 1) for dk in (0, 1)
    ])
    mn = corners.min(axis=0)
    mx = corners.max(axis=0)
    mn_abs = np.abs(corners).min(axis=0)
    active = ((mn <= 0) & (mx >= 0)) | (mn_abs < tau)  # (C, C, C)

    # dilate by one cell (full 3x3x3 neighborhood)
    dilated = ndimage.binary_dilation(active, structure=np.ones((3, 3, 3), dtype=bool))

    # remap onto the finer grid's cells: next[i] samples dilated[min(C-1, floor(i*C/next_c))]
    idx = np.minimum(C - 1, (np.arange(next_c) * C) // next_c)
    return dilated[np.ix_(idx, idx, idx)]


def eval_level_field(R, decoder, bounds, cell_mask=None, prev_field=None, prev_R=0,
                     on_chunk=None, chunk_rows=CHUNK_ROWS):
    # Evaluate one level's SDF grid: every point at the coarsest level (cell_mask=None), or
    # only the corner points of masked cells with the rest inherited from the previous level
    # (nearest-sample fill keeps signs consistent, so marching cubes can't invent surfaces in
    # unrefined regions). Points are evaluated in chunks; on_chunk(rows_done, rows_total,
    # field) fires after each with the growing field, so the caller can stream partial
    # previews. decoder maps (N,3) xyz -> (N,) sdf. (latent_backend.js:201)
    mins, maxs = bounds
    xs = np.linspace(mins[0], maxs[0], R)
    ys = np.linspace(mins[1], maxs[1], R)
    zs = np.linspace(mins[2], maxs[2], R)
    n = R * R * R
    field = np.empty(n, dtype=np.float64)

    if cell_mask is None:
        point_indices = np.arange(n)
    else:
        # inherit the previous level via nearest-sample fill (round-half-up to match the JS ref)
        pf = np.asarray(prev_field, dtype=np.float64).reshape(prev_R, prev_R, prev_R)
        mi = np.floor(np.arange(R) * (prev_R - 1) / (R - 1) + 0.5).astype(int)
        field[:] = pf[np.ix_(mi, mi, mi)].reshape(-1)
        # corner points of masked cells (8 corners per active cell, deduped by the grid)
        C = R - 1
        mark = np.zeros((R, R, R), dtype=bool)
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    mark[di:di + C, dj:dj + C, dk:dk + C] |= cell_mask
        point_indices = np.flatnonzero(mark.reshape(-1))

    total = len(point_indices)
    for start in range(0, total, chunk_rows):
        idx = point_indices[start:start + chunk_rows]
        points = np.stack([xs[idx // (R * R)], ys[(idx // R) % R], zs[idx % R]], axis=1)
        field[idx] = decoder(points)
        if on_chunk is not None:
            on_chunk(min(start + chunk_rows, total), total, field)
    return field


def _postprocess(vertices, cage, rescale):
    # Rescale the marching-cubes mesh to the latent cage's size, then centralize onto the
    # cage (bounds-midpoint translation). Reproduces model.py:226-238 / postProcessMesh.
    cage = np.asarray(cage, dtype=np.float64)
    lat_min, lat_max = cage.min(axis=0), cage.max(axis=0)
    v = np.asarray(vertices, dtype=np.float64)
    if rescale:
        m_min, m_max = v.min(axis=0), v.max(axis=0)
        v = v * ((lat_max - lat_min) / (m_max - m_min))
    m_min, m_max = v.min(axis=0), v.max(axis=0)  # recompute after the optional scale
    return v + (0.5 * (lat_min + lat_max) - 0.5 * (m_min + m_max))


def reconstruct_adaptive(cage, resolution, decoder, bounds=BOUNDS, rescale=True,
                         map_z_to_y=True, adaptive=True, max_levels=6, min_level_res=16,
                         chunk_rows=CHUNK_ROWS):
    # Coarse-to-fine with a highlight sweep: the coarsest level samples the whole grid; each
    # finer level re-evaluates only near-surface cells and inherits the rest. Every preview
    # splits faces into base (inherited) + changed (freshly refined vs the previous preview);
    # the final level streams one preview per decoder chunk, so the highlight sweeps across the
    # surface. Yields {vertices, faces_base, faces_changed} per event, the last flagged final.
    # (localReconstruct + worker postPreview/onChunk.)
    levels = levels_for(resolution, max_levels, min_level_res) if adaptive else [resolution]
    level_count = len(levels)
    field = None
    prev_R = 0
    cell_mask = None
    prev_preview = {"field": None, "R": 0}  # last posted preview, for the change detection

    def make_preview(f, R):
        changed = (np.ones((R, R, R), dtype=bool) if prev_preview["field"] is None
                   else _changed_mask(f, R, prev_preview["field"], prev_preview["R"]))
        prev_preview["field"] = np.asarray(f, dtype=np.float64).copy()
        prev_preview["R"] = R
        grid = np.asarray(f, dtype=np.float64).reshape(R, R, R)
        if not (grid.min() <= MC_LEVEL <= grid.max()):
            return None
        verts, faces, _, _ = measure.marching_cubes(grid, level=MC_LEVEL)
        faces_base, faces_changed = _split_faces(verts, faces, R, changed)
        verts = _postprocess(verts, cage, rescale)
        if map_z_to_y:
            verts = verts[:, [0, 2, 1]]
        return {"vertices": verts.tolist(), "faces_base": faces_base, "faces_changed": faces_changed}

    for li, R in enumerate(levels):
        is_final = li == level_count - 1
        if is_final and cell_mask is not None:
            # final level of a multi-level run: stream a preview per decoder chunk (the sweep)
            previews = []

            def on_chunk(rows_done, rows_total, f):
                previews.append((rows_done == rows_total, make_preview(f, R)))

            field = eval_level_field(R, decoder, bounds, cell_mask=cell_mask, prev_field=field,
                                     prev_R=prev_R, on_chunk=on_chunk, chunk_rows=chunk_rows)
            if not previews:  # no near-surface cells to chunk -> one preview of the inherited field
                previews = [(True, make_preview(field, R))]
            for is_last, mesh in previews:
                yield {"level": li + 1, "level_count": level_count, "resolution": R,
                       "final": is_last, "mesh": mesh}
        else:
            # coarse or single level: evaluate fully, then post one preview
            field = eval_level_field(R, decoder, bounds, cell_mask=cell_mask,
                                     prev_field=field, prev_R=prev_R, chunk_rows=chunk_rows)
            yield {"level": li + 1, "level_count": level_count, "resolution": R,
                   "final": is_final, "mesh": make_preview(field, R)}
            if not is_final:
                cell_mask = build_next_mask(field, R, levels[li + 1], bounds)
        prev_R = R


def _split_faces(vertices, faces, R, changed_mask):
    # Split marching-cubes faces into base (unchanged) and changed (freshly refined): a face
    # is "changed" if any corner of its generating cell changed. The cell is recovered from
    # the face's vertex centroid, clamped so all 8 corners exist. (reconstruct_worker.js:158-172)
    verts = np.asarray(vertices)
    changed_mask = np.asarray(changed_mask)
    faces_base = []
    faces_changed = []
    for face in np.asarray(faces):
        tri = verts[face]
        i = min(R - 2, int(np.floor(tri[:, 0].sum() / 3)))
        j = min(R - 2, int(np.floor(tri[:, 1].sum() / 3)))
        k = min(R - 2, int(np.floor(tri[:, 2].sum() / 3)))
        cell_changed = changed_mask[i:i + 2, j:j + 2, k:k + 2].any()
        (faces_changed if cell_changed else faces_base).append([int(v) for v in face])
    return faces_base, faces_changed


def _changed_mask(field, R, prev_field, prev_R):
    # Per-point change vs the previous preview: no previous -> everything changed; same
    # resolution -> a plain value diff; a level transition -> diff against the nearest-
    # upsampled coarser field (same round-half-up mapping the inherit fill uses, so an
    # inherited point is bit-exact and reads as unchanged). (reconstruct_worker.js:125-151)
    f = np.asarray(field, dtype=np.float64).reshape(R, R, R)
    if prev_field is None:
        return np.ones((R, R, R), dtype=bool)
    pf = np.asarray(prev_field, dtype=np.float64).reshape(prev_R, prev_R, prev_R)
    if prev_R == R:
        return f != pf
    mi = np.floor(np.arange(R) * (prev_R - 1) / (R - 1) + 0.5).astype(int)
    return f != pf[np.ix_(mi, mi, mi)]
