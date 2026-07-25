import json
import os
import sys

import numpy as np

HERE = os.path.dirname(__file__)
DEMO_DIR = os.path.abspath(os.path.join(HERE, ".."))
if DEMO_DIR not in sys.path:
    sys.path.insert(0, DEMO_DIR)

from reconstruct_stream import (  # noqa: E402
    _changed_mask,
    _split_faces,
    build_next_mask,
    eval_level_field,
    levels_for,
    reconstruct_adaptive,
)


def test_levels_for_builds_coarse_to_fine_ladder():
    # halve the resolution down to >= min_level_res, capped at max_levels, finest last
    assert levels_for(128, max_levels=6, min_level_res=16) == [16, 32, 64, 128]
    # capped at max_levels — keep the finest levels
    assert levels_for(128, max_levels=2, min_level_res=16) == [64, 128]
    # never go below min_level_res
    assert levels_for(64, max_levels=6, min_level_res=16) == [16, 32, 64]
    # an odd resolution can't be halved -> single level
    assert levels_for(97, max_levels=6, min_level_res=16) == [97]


def test_build_next_mask_marks_near_surface_cells_and_dilates():
    # A field depending only on x: a plane crossing zero at x=3, steep enough that
    # far-from-surface cells exceed tau (= 2.5 * cell_size = 2.5 here). With next_R == R
    # the coarse->fine remap is the identity, so the returned mask == the dilated mask.
    R = 7
    bounds = ((0.0, 0.0, 0.0), (6.0, 6.0, 6.0))  # cell size 1.0 -> tau = 2.5
    xs = np.linspace(0.0, 6.0, R)
    field = np.empty((R, R, R), dtype=np.float64)
    for i in range(R):
        field[i, :, :] = (xs[i] - 3.0) * 5.0  # -15..0..+15 across x
    mask = build_next_mask(field.reshape(-1), R, R, bounds)

    assert mask.shape == (R - 1, R - 1, R - 1)      # one entry per next-grid cell
    assert mask[2].all() and mask[3].all()          # cells straddling x=3 -> active (sign change)
    assert mask[1].all() and mask[4].all()          # 1-cell dilation of {2,3}
    assert not mask[0].any()                        # x-slab 0: |sdf|>=10 > tau, no crossing
    assert not mask[R - 2].any()                    # x-slab 5: far side, likewise inactive


def test_eval_level_field_coarsest_all_then_masked_inherits():
    bounds = ((0.0, 0.0, 0.0), (4.0, 4.0, 4.0))

    # --- coarsest (no mask): every grid point is evaluated, in flat i*R*R+j*R+k order ---
    R0 = 3
    seen = {"n": 0}

    def dec_x(xyz):
        seen["n"] += xyz.shape[0]
        return xyz[:, 0].copy()  # sdf = x-coordinate

    field0 = eval_level_field(R0, dec_x, bounds)
    assert seen["n"] == R0 ** 3                       # all points evaluated
    xs0 = np.linspace(0.0, 4.0, R0)
    assert np.allclose(field0.reshape(R0, R0, R0)[:, 0, 0], xs0)  # x varies along axis 0

    # --- finer level with a 1-cell mask: inherit prev everywhere, evaluate only masked corners ---
    R, prev_R = 5, 3
    prev_field = np.full(prev_R ** 3, 100.0)          # distinctive inherited value
    calls = {"n": 0}

    def dec_const(xyz):
        calls["n"] += xyz.shape[0]
        return np.full(xyz.shape[0], -999.0)          # distinctive evaluated value

    cell_mask = np.zeros((R - 1, R - 1, R - 1), dtype=bool)
    cell_mask[0, 0, 0] = True                          # refine only cell (0,0,0)
    field = eval_level_field(R, dec_const, bounds, cell_mask=cell_mask, prev_field=prev_field, prev_R=prev_R)
    f = field.reshape(R, R, R)

    assert calls["n"] == 8                             # only the masked cell's 8 corner points
    assert f[0, 0, 0] == -999.0 and f[1, 1, 1] == -999.0   # those corners hold the evaluated value
    assert f[R - 1, R - 1, R - 1] == 100.0            # a far point is inherited from prev


def _sphere_sdf(center, radius):
    center = np.asarray(center, dtype=np.float64)

    def sdf(xyz):
        return np.linalg.norm(np.asarray(xyz, dtype=np.float64) - center, axis=1) - radius

    return sdf


def _final_event(cage, R, decoder, bounds, adaptive):
    last = None
    for event in reconstruct_adaptive(cage, R, decoder, bounds=bounds, adaptive=adaptive,
                                      max_levels=6, min_level_res=16):
        last = event
    return last


def test_reconstruct_adaptive_streams_previews_and_merged_final_matches_dense():
    # Mock sphere SDF. Each preview splits faces into base (inherited) + changed (freshly
    # refined); the coarsest preview is all-changed; the final level streams a preview per
    # chunk (the sweep); and the merged final mesh (base+changed) still matches the dense one.
    sphere = _sphere_sdf([0.0, 0.0, 0.0], 0.4)
    bounds = ((-0.6, -0.6, -0.6), (0.6, 0.6, 0.6))
    cage = [[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]]
    R = 64

    events = list(reconstruct_adaptive(cage, R, sphere, bounds=bounds, adaptive=True,
                                       max_levels=6, min_level_res=16, chunk_rows=3000))

    # streamed multiple previews: 3 coarse levels + several final-level chunks
    assert len(events) >= 4
    for ev in events:                                    # every preview is base + changed split
        m = ev["mesh"]
        assert m is not None and "faces_base" in m and "faces_changed" in m
    assert events[0]["mesh"]["faces_base"] == []         # coarsest: everything freshly refined
    final_level = events[-1]["level"]
    final_events = [ev for ev in events if ev["level"] == final_level]
    assert len(final_events) >= 2                         # final level streamed in chunks -> the sweep
    assert [i for i, ev in enumerate(events) if ev["final"]] == [len(events) - 1]  # one final, last

    # the merged final mesh (base + changed) matches the dense single-grid mesh
    final_mesh = events[-1]["mesh"]
    merged = final_mesh["faces_base"] + final_mesh["faces_changed"]
    dense = _final_event(cage, R, sphere, bounds, adaptive=False)["mesh"]
    dense_merged = dense["faces_base"] + dense["faces_changed"]
    fv = np.array(final_mesh["vertices"])
    dv = np.array(dense["vertices"])
    assert np.allclose(fv.min(axis=0), dv.min(axis=0), atol=1e-3)
    assert np.allclose(fv.max(axis=0), dv.max(axis=0), atol=1e-3)
    assert abs(len(merged) - len(dense_merged)) <= 0.02 * len(dense_merged)


def _parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        block = block.strip()
        if block.startswith("data:"):
            events.append(json.loads(block[len("data:"):].strip()))
    return events


def test_reconstruct_stream_endpoint_streams_sweep():
    # Integration: the SSE endpoint drives the real decoder and streams the highlight sweep —
    # every event carries the base/changed face split, and the final level streams several
    # per-chunk previews (so the sweep reaches the client), ending with the merged final mesh.
    from fastapi.testclient import TestClient
    import app  # loads the trained model (slow) — imported inside the test so KR1-4 stay fast

    client = TestClient(app.app)
    ls = client.get("/api/latent_shapes").json()
    resp = client.post("/api/reconstruct/stream", json={
        "latent_shapes": ls["latent_shape"],
        "resolution": 64,
        "rescale": True,
        "map_z_to_y": True,
        "ensure_watertight": False,
    })
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")

    events = _parse_sse(resp.text)
    for ev in events:                                  # every mesh is a base + changed split
        assert "faces_base" in ev["mesh"] and "faces_changed" in ev["mesh"]
    res = [e["resolution"] for e in events]
    assert res == sorted(res) and res[-1] == 64        # increasing, ends at the target resolution
    final_events = [ev for ev in events if ev["level"] == events[-1]["level"]]
    assert len(final_events) >= 2                       # final level streamed in chunks -> the sweep
    assert events[-1]["final"] is True
    fm = events[-1]["mesh"]
    assert len(fm["faces_base"]) + len(fm["faces_changed"]) > 0  # merged final mesh has faces


def test_split_faces_partitions_by_changed_cell():
    R = 4
    # triangle A sits in cell (0,0,0); triangle B in cell (2,2,2)
    vertices = np.array([
        [0.2, 0.2, 0.2], [0.8, 0.2, 0.2], [0.2, 0.8, 0.2],   # A
        [2.2, 2.2, 2.2], [2.8, 2.2, 2.2], [2.2, 2.8, 2.2],   # B
    ])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    changed = np.zeros((R, R, R), dtype=bool)
    changed[0:2, 0:2, 0:2] = True  # only cell (0,0,0)'s corners are marked changed

    base, changed_faces = _split_faces(vertices, faces, R, changed)
    assert changed_faces == [[0, 1, 2]]     # triangle A (in the changed cell) -> changed
    assert base == [[3, 4, 5]]              # triangle B (unchanged cell) -> base
    assert sorted(base + changed_faces) == sorted(faces.tolist())  # partition: nothing lost/duplicated

    # an all-changed mask puts every face in 'changed', with base empty
    base2, changed2 = _split_faces(vertices, faces, R, np.ones((R, R, R), dtype=bool))
    assert base2 == [] and len(changed2) == 2


def test_changed_mask_flags_points_that_differ_from_previous_preview():
    R = 4
    field = np.arange(R ** 3, dtype=float)  # distinct value per point

    # no previous preview -> everything is 'changed'
    m0 = _changed_mask(field, R, None, 0)
    assert m0.shape == (R, R, R) and m0.all()

    # same resolution -> only the point whose value differs is 'changed'
    prev = field.copy()
    prev[5] = field[5] + 1.0
    m1 = _changed_mask(field, R, prev, R)
    expected = np.zeros(R ** 3, dtype=bool)
    expected[5] = True
    assert np.array_equal(m1.reshape(-1), expected)

    # cross-resolution -> compare against the nearest-upsampled coarser prev
    R2, prev_R = 3, 2
    prev2 = np.zeros(prev_R ** 3)
    assert not _changed_mask(np.zeros(R2 ** 3), R2, prev2, prev_R).any()  # identical -> none
    cur = np.zeros(R2 ** 3)
    cur[0] = 9.0  # point (0,0,0) now differs from the upsampled prev value (0)
    assert _changed_mask(cur, R2, prev2, prev_R).reshape(-1)[0]


def test_eval_level_field_on_chunk_fires_per_chunk_with_partial_field():
    R = 4  # 64 grid points
    bounds = ((0.0, 0.0, 0.0), (3.0, 3.0, 3.0))

    def dec(xyz):
        return xyz[:, 0].copy()  # sdf = x

    seen = []
    last = {}

    def on_chunk(rows_done, rows_total, field):
        seen.append((rows_done, rows_total))
        last["field"] = field.copy()

    field = eval_level_field(R, dec, bounds, on_chunk=on_chunk, chunk_rows=20)
    assert [d for d, _ in seen] == [20, 40, 60, 64]   # 64 points, chunks of 20 -> 4 callbacks
    assert all(t == 64 for _, t in seen)              # rows_total constant
    non_chunked = eval_level_field(R, dec, bounds)
    assert np.allclose(field, non_chunked)            # chunking must not change the result
    assert np.allclose(last["field"], non_chunked)    # the callback receives the working field


def test_torch_decoder_batches_are_smaller_than_preview_chunks(monkeypatch):
    import app

    inference_batches = []

    def fake_forward(cxyz):
        inference_batches.append(len(cxyz))
        return cxyz[:, :1]

    monkeypatch.setattr(app.sdf_decoder, "forward", fake_forward)
    decoder = app._make_stream_decoder([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    preview_updates = []
    R = 41  # 68,921 points: one 65,536-row preview chunk plus a tail
    field = eval_level_field(
        R,
        decoder,
        ((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)),
        on_chunk=lambda done, total, _field: preview_updates.append((done, total)),
    )

    assert inference_batches == [32768, 32768, 3385]
    assert preview_updates == [(65536, R ** 3), (R ** 3, R ** 3)]
    assert field.shape == (R ** 3,)
    expected = np.broadcast_to(
        np.linspace(-1.0, 1.0, R)[:, None, None],
        (R, R, R),
    )
    assert np.allclose(field.reshape(R, R, R), expected)
