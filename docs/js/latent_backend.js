// Browser-side reproduction of the FastAPI demo's numeric pipeline
// (latent_shapes/demo/app.py + latent_shapes/src/model.py reconstruct()).

import { marchingCubes } from "./marching_cubes.js?v=3";

// Fixed sampling bounds and iso-level from latent_shapes/src/config.py.
export const BOUNDS = {
  min: [-0.6149350, -0.767495, -0.666214],
  max: [0.644344, 0.792882, 0.581235],
};
export const MC_LEVEL = 0.0;
export const NUM_LATENT_VERTS = 98; // subdivided box, config.NUM_LATENT_SHAPE_VERTICES
export const DECODER_INPUT_DIM = (NUM_LATENT_VERTS + 1) * 3; // 297

// torch.linspace(a, b, n): n evenly spaced points including both endpoints.
function linspace(a, b, n) {
  if (n === 1) return [a];
  const out = new Array(n);
  const step = (b - a) / (n - 1);
  for (let i = 0; i < n; i++) out[i] = a + step * i;
  return out;
}

// Reproduces model.py:174-190 exactly:
//   x/y/z = linspace(MIN_*, MAX_*, R)
//   meshgrid(x, y, z, indexing="ij") -> stack -> reshape(-1, 3)
// With indexing="ij" the flattened order is x slowest, z fastest:
//   point[i*R*R + j*R + k] = (x[i], y[j], z[k]).
export function buildGrid(resolution, bounds) {
  const R = resolution;
  const xs = linspace(bounds.min[0], bounds.max[0], R);
  const ys = linspace(bounds.min[1], bounds.max[1], R);
  const zs = linspace(bounds.min[2], bounds.max[2], R);

  const points = new Array(R * R * R);
  let p = 0;
  for (let i = 0; i < R; i++) {
    for (let j = 0; j < R; j++) {
      for (let k = 0; k < R; k++) {
        points[p++] = [xs[i], ys[j], zs[k]];
      }
    }
  }
  return points;
}

// Swap components 1 and 2 (y<->z). This single operation appears server-side at
// app.py:64 (latent output), app.py:77 (decoder input), and model.py:249 (mesh
// output when map_z_to_y). The Y-up three.js scene uses the same reorder on the
// client (interpolator.html:711/778/817), so with map_z_to_y=True the server
// swap and the display swap cancel — reproduce both, never drop one.
export function swapYZ(v) {
  return [v[0], v[2], v[1]];
}

// Reproduces model.py:248-249: swap the reconstructed mesh's y/z iff map_z_to_y.
export function mapMeshToClient(vertices, mapZToY) {
  return mapZToY
    ? vertices.map(swapYZ)
    : vertices.map((v) => [v[0], v[1], v[2]]);
}

// Per-axis [min, max] bounding box over a list of [x, y, z] points
// (equivalent to trimesh's mesh.bounds).
function bboxMinMax(vertices) {
  const mn = [Infinity, Infinity, Infinity];
  const mx = [-Infinity, -Infinity, -Infinity];
  for (const v of vertices) {
    for (let c = 0; c < 3; c++) {
      if (v[c] < mn[c]) mn[c] = v[c];
      if (v[c] > mx[c]) mx[c] = v[c];
    }
  }
  return [mn, mx];
}

// Reproduces model.py:226-238. Optional per-axis rescale so the marching-cubes
// mesh matches the latent cage's size, then a bounds-midpoint centralize onto
// the cage:  translation = midpoint(cage_bbox) - midpoint(mesh_bbox).
// This is deliberately NOT data.py._centralize_mesh's mean-subtraction.
export function postProcessMesh(vertices, latentCage, { rescale } = {}) {
  const [latMin, latMax] = bboxMinMax(latentCage);
  let v = vertices.map((p) => [p[0], p[1], p[2]]);

  if (rescale) {
    const [meshMin, meshMax] = bboxMinMax(v);
    const scale = [
      (latMax[0] - latMin[0]) / (meshMax[0] - meshMin[0]),
      (latMax[1] - latMin[1]) / (meshMax[1] - meshMin[1]),
      (latMax[2] - latMin[2]) / (meshMax[2] - meshMin[2]),
    ];
    v = v.map((p) => [p[0] * scale[0], p[1] * scale[1], p[2] * scale[2]]);
  }

  const [meshMin, meshMax] = bboxMinMax(v); // recompute after the optional scale
  const translation = [
    0.5 * (latMin[0] + latMax[0]) - 0.5 * (meshMin[0] + meshMax[0]),
    0.5 * (latMin[1] + latMax[1]) - 0.5 * (meshMin[1] + meshMax[1]),
    0.5 * (latMin[2] + latMax[2]) - 0.5 * (meshMin[2] + meshMax[2]),
  ];
  return v.map((p) => [
    p[0] + translation[0],
    p[1] + translation[1],
    p[2] + translation[2],
  ]);
}

// Rows streamed to the decoder per chunk. One full R=128 grid as a single tensor
// would be 128^3 * 297 * 4B ≈ 2.5 GB ("Array buffer allocation failed"); chunking
// caps peak input memory at 65536 * 297 * 4B ≈ 78 MB regardless of resolution.
const CHUNK_ROWS = 65536;

// Coarse-to-fine refinement ladder: halve the resolution down to >=minLevelRes, up
// to maxLevels levels. Only cells near the surface at one level are re-evaluated at
// the next, which is where the speedup comes from (the decoder is ~99% of the cost).
// maxLevels/minLevelRes default to the shipped values (3 / 16); callers can widen the
// ladder (more, coarser levels) to trade total time for a sooner, smoother reveal.
function levelsFor(resolution, maxLevels = 3, minLevelRes = 16) {
  const levels = [resolution];
  while (levels.length < maxLevels && levels[0] % 2 === 0 && levels[0] / 2 >= minLevelRes) {
    levels.unshift(levels[0] / 2);
  }
  return levels;
}

// Activity threshold: a cell is "near the surface" if its corners change sign or
// the smallest |sdf| is under tau = 2.5x the cell size. Calibrated on the real
// decoder (3 cages): far-field |sdf| is 0.3-0.7, and even tau=1.5x covered 100%
// of true fine-level surface cells after 1-cell dilation; 2.5x is the margin.
const ACTIVE_TAU_CELLS = 2.5;

// Mark the next level's cells to refine: this level's near-surface cells,
// dilated by one cell, mapped onto the finer cell grid.
function buildNextMask(field, R, nextR, bounds) {
  const C = R - 1;
  const nextC = nextR - 1;
  const tau = ACTIVE_TAU_CELLS *
    Math.max(...[0, 1, 2].map((a) => (bounds.max[a] - bounds.min[a]) / C));

  const active = new Uint8Array(C * C * C);
  for (let i = 0; i < C; i++) {
    for (let j = 0; j < C; j++) {
      for (let k = 0; k < C; k++) {
        let mn = Infinity, mx = -Infinity, mnAbs = Infinity;
        for (let d = 0; d < 8; d++) {
          const v = field[(i + (d >> 2)) * R * R + (j + ((d >> 1) & 1)) * R + (k + (d & 1))];
          if (v < mn) mn = v;
          if (v > mx) mx = v;
          const a = Math.abs(v);
          if (a < mnAbs) mnAbs = a;
        }
        if ((mn <= 0 && 0 <= mx) || mnAbs < tau) active[(i * C + j) * C + k] = 1;
      }
    }
  }

  // dilate by one cell (stamp neighbors of active cells)
  const dilated = new Uint8Array(C * C * C);
  for (let i = 0; i < C; i++) {
    for (let j = 0; j < C; j++) {
      for (let k = 0; k < C; k++) {
        if (!active[(i * C + j) * C + k]) continue;
        for (let di = -1; di <= 1; di++) {
          for (let dj = -1; dj <= 1; dj++) {
            for (let dk = -1; dk <= 1; dk++) {
              const a = i + di, b = j + dj, c = k + dk;
              if (a >= 0 && a < C && b >= 0 && b < C && c >= 0 && c < C) {
                dilated[(a * C + b) * C + c] = 1;
              }
            }
          }
        }
      }
    }
  }

  const next = new Uint8Array(nextC * nextC * nextC);
  for (let i = 0; i < nextC; i++) {
    const ci = Math.min(C - 1, Math.floor((i * C) / nextC));
    for (let j = 0; j < nextC; j++) {
      const cj = Math.min(C - 1, Math.floor((j * C) / nextC));
      for (let k = 0; k < nextC; k++) {
        const ck = Math.min(C - 1, Math.floor((k * C) / nextC));
        next[(i * nextC + j) * nextC + k] = dilated[(ci * C + cj) * C + ck];
      }
    }
  }
  return next;
}

// Full browser reproduction of app.py /api/reconstruct + model.py reconstruct().
//   latentShapes: N x 3 cage in the POST-body (xzy) convention the viewer sends.
//   runDecoder:   (Float32Array[N*297], N) -> Promise<Float32Array[N]> of SDF values;
//                 the caller injects onnxruntime-web (browser) or onnxruntime-node (test).
// Returns { vertices, faces } in the server-output convention (the viewer applies
// its own display swap), or null if the field never crosses the iso-level.
// Evaluate one level's grid: all points (mask null, coarsest level) or only the
// corner points of masked cells, with the rest of the field inherited from the
// previous level (nearest-sample fill keeps signs consistent, so marching cubes
// cannot invent surfaces in unrefined regions).
async function evalLevelField(
  R, cageFlat, runDecoder, bounds, cellMask, prevField, prevR, onChunkRows, chunkRows = CHUNK_ROWS,
) {
  const xs = linspace(bounds.min[0], bounds.max[0], R);
  const ys = linspace(bounds.min[1], bounds.max[1], R);
  const zs = linspace(bounds.min[2], bounds.max[2], R);
  const n = R * R * R;
  const field = new Float32Array(n);

  let points = null; // Uint32Array of flat point indices to evaluate; null = all
  if (cellMask) {
    for (let i = 0; i < R; i++) {
      const pi = Math.round((i * (prevR - 1)) / (R - 1));
      for (let j = 0; j < R; j++) {
        const pj = Math.round((j * (prevR - 1)) / (R - 1));
        for (let k = 0; k < R; k++) {
          const pk = Math.round((k * (prevR - 1)) / (R - 1));
          field[i * R * R + j * R + k] = prevField[pi * prevR * prevR + pj * prevR + pk];
        }
      }
    }
    const C = R - 1;
    const mark = new Uint8Array(n);
    for (let i = 0; i < C; i++) {
      for (let j = 0; j < C; j++) {
        for (let k = 0; k < C; k++) {
          if (!cellMask[(i * C + j) * C + k]) continue;
          for (let d = 0; d < 8; d++) {
            mark[(i + (d >> 2)) * R * R + (j + ((d >> 1) & 1)) * R + (k + (d & 1))] = 1;
          }
        }
      }
    }
    let count = 0;
    for (let p = 0; p < n; p++) count += mark[p];
    points = new Uint32Array(count);
    let w = 0;
    for (let p = 0; p < n; p++) if (mark[p]) points[w++] = p;
  }

  const total = points ? points.length : n;
  for (let s = 0; s < total; s += chunkRows) {
    const rows = Math.min(chunkRows, total - s);
    const chunk = new Float32Array(rows * DECODER_INPUT_DIM);
    for (let r = 0; r < rows; r++) {
      const p = points ? points[s + r] : s + r;
      const base = r * DECODER_INPUT_DIM;
      chunk[base] = xs[Math.floor(p / (R * R))];
      chunk[base + 1] = ys[Math.floor(p / R) % R];
      chunk[base + 2] = zs[p % R];
      for (let m = 0; m < cageFlat.length; m++) chunk[base + 3 + m] = cageFlat[m];
    }
    const out = await runDecoder(chunk, rows);
    if (points) {
      for (let r = 0; r < rows; r++) field[points[s + r]] = out[r];
    } else {
      field.set(out, s);
    }
    if (onChunkRows) onChunkRows(s + rows, total, field);
  }
  return field;
}

export async function localReconstruct(
  {
    latentShapes, resolution, rescale = true, mapZToY = true,
    adaptive = true, onChunk = null, onLevel = null,
    // Responsiveness tuning — defaults preserve the shipped behavior.
    chunkRows = CHUNK_ROWS, maxLevels = 3, minLevelRes = 16,
  },
  runDecoder,
  bounds = BOUNDS,
) {
  const R = resolution;

  // app.py:77 — swap the body cage (xzy) back to decoder xyz space.
  const cage = latentShapes.map(swapYZ);
  const cageFlat = cage.flat(); // 98 * 3 = 294, row-major (model.py reshape(1, -1))

  // model.py:174-202, evaluated coarse-to-fine: the coarsest level samples the
  // whole grid; each finer level re-evaluates only near-surface cells and
  // inherits the rest. Point order within a level matches buildGrid.
  const levels = adaptive ? levelsFor(R, maxLevels, minLevelRes) : [R];
  let field = null;
  let prevR = 0;
  let cellMask = null;
  for (let li = 0; li < levels.length; li++) {
    const levelR = levels[li];
    field = await evalLevelField(
      levelR, cageFlat, runDecoder, bounds, cellMask, field, prevR,
      onChunk
        ? (rowsDone, rowsTotal, liveField) => onChunk({
            level: li + 1, levelCount: levels.length, rowsDone, rowsTotal,
            field: liveField, resolution: levelR,
          })
        : null,
      chunkRows,
    );
    if (li < levels.length - 1) {
      cellMask = buildNextMask(field, levelR, levels[li + 1], bounds);
      if (onLevel) onLevel({ field, resolution: levelR, levelIndex: li, levelCount: levels.length });
    }
    prevR = levelR;
  }
  const sdf = field;
  const n = R * R * R;

  // model.py:209 — bail out if the field never crosses the iso-level.
  let lo = Infinity;
  let hi = -Infinity;
  for (let p = 0; p < n; p++) {
    const s = sdf[p];
    if (s < lo) lo = s;
    if (s > hi) hi = s;
  }
  if (!(lo <= MC_LEVEL && MC_LEVEL <= hi)) return null;

  const { vertices, faces } = marchingCubes([R, R, R], sdf, MC_LEVEL);
  const centered = postProcessMesh(vertices, cage, { rescale }); // rescale + centralize
  return { vertices: mapMeshToClient(centered, mapZToY), faces }; // model.py:249
}
