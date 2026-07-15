import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { marchingCubes } from "../../../docs/js/marching_cubes.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const g = JSON.parse(
  readFileSync(join(HERE, "fixtures", "sphere_mc_golden.json"), "utf8"),
);

// field[i*R*R + j*R + k] = distance((i,j,k), center) - radius  (same layout as buildGrid)
function sphereField(R, center, radius) {
  const field = new Float32Array(R * R * R);
  for (let i = 0; i < R; i++) {
    for (let j = 0; j < R; j++) {
      for (let k = 0; k < R; k++) {
        const dx = i - center[0], dy = j - center[1], dz = k - center[2];
        field[i * R * R + j * R + k] = Math.sqrt(dx * dx + dy * dy + dz * dz) - radius;
      }
    }
  }
  return field;
}

test("layer-range marching cubes concatenates to the exact full-pass surface", () => {
  const R = g.resolution;
  const field = sphereField(R, g.center, g.radius);
  const full = marchingCubes([R, R, R], field, 0);

  // split the first axis into 4 uneven cell ranges, as progressive slabs would
  const cuts = [0, 4, Math.floor((R - 1) / 2), Math.floor(((R - 1) * 3) / 4), R - 1];
  let faceCount = 0;
  let vertCount = 0;
  const areaOf = (m) => {
    let area = 0;
    for (const f of m.faces) {
      const [a, b, c] = f.map((i) => m.vertices[i]);
      const ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      const ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
      const cx = ab[1] * ac[2] - ab[2] * ac[1];
      const cy = ab[2] * ac[0] - ab[0] * ac[2];
      const cz = ab[0] * ac[1] - ab[1] * ac[0];
      area += 0.5 * Math.hypot(cx, cy, cz);
    }
    return area;
  };
  let slabArea = 0;
  for (let s = 0; s < cuts.length - 1; s++) {
    const slab = marchingCubes([R, R, R], field, 0, [cuts[s], cuts[s + 1]]);
    faceCount += slab.faces.length;
    vertCount += slab.vertices.length;
    slabArea += areaOf(slab);
  }
  assert.equal(faceCount, full.faces.length, "slab faces must sum to the full pass");
  assert.equal(vertCount, full.vertices.length, "slab vertices must sum to the full pass");
  const fullArea = areaOf(full);
  assert.ok(
    Math.abs(slabArea - fullArea) < 1e-9 * Math.max(1, fullArea),
    `total surface area differs: slabs ${slabArea} vs full ${fullArea}`,
  );
});

test("marching cubes extracts a correct level-0 isosurface of a sphere", () => {
  const R = g.resolution, c = g.center, r = g.radius;
  const { vertices, faces } = marchingCubes([R, R, R], sphereField(R, c, r), 0);

  assert.ok(vertices.length > 0, "non-empty vertices");
  assert.ok(faces.length > 0, "non-empty faces");
  for (const f of faces) {
    for (const idx of f) {
      assert.ok(idx >= 0 && idx < vertices.length, "face references a valid vertex");
    }
  }

  // a real surface has thousands of edge-crossing vertices, not a stray handful
  assert.ok(vertices.length > 500, `only ${vertices.length} vertices — surface too sparse`);

  // analytic roundness: interpolated marching cubes hugs the true sphere tightly.
  // These thresholds pass for real MC (max ~0.014, mean ~0.006) but fail if edge
  // interpolation is dropped (corner-snapping pushes errors toward ~0.9).
  let maxRadErr = 0;
  let sumRadErr = 0;
  const bbMin = [Infinity, Infinity, Infinity];
  const bbMax = [-Infinity, -Infinity, -Infinity];
  for (const v of vertices) {
    const d = Math.hypot(v[0] - c[0], v[1] - c[1], v[2] - c[2]);
    maxRadErr = Math.max(maxRadErr, Math.abs(d - r));
    sumRadErr += Math.abs(d - r);
    for (let ax = 0; ax < 3; ax++) {
      if (v[ax] < bbMin[ax]) bbMin[ax] = v[ax];
      if (v[ax] > bbMax[ax]) bbMax[ax] = v[ax];
    }
  }
  const meanRadErr = sumRadErr / vertices.length;
  assert.ok(maxRadErr < 0.1, `max radius error ${maxRadErr.toFixed(3)} exceeds 0.1`);
  assert.ok(meanRadErr < 0.02, `mean radius error ${meanRadErr.toFixed(4)} exceeds 0.02`);

  // cross-check bbox against the skimage.measure.marching_cubes reference
  for (let ax = 0; ax < 3; ax++) {
    assert.ok(
      Math.abs(bbMin[ax] - g.bbox.min[ax]) < 0.75,
      `bbox min axis ${ax}: ${bbMin[ax].toFixed(3)} vs skimage ${g.bbox.min[ax].toFixed(3)}`,
    );
    assert.ok(
      Math.abs(bbMax[ax] - g.bbox.max[ax]) < 0.75,
      `bbox max axis ${ax}: ${bbMax[ax].toFixed(3)} vs skimage ${g.bbox.max[ax].toFixed(3)}`,
    );
  }
});
