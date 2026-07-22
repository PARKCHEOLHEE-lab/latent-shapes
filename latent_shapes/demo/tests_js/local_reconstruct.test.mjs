import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { createRequire } from "node:module";

import {
  localReconstruct,
  swapYZ,
  DECODER_INPUT_DIM,
} from "../../../docs/js/latent_backend.js";

const require = createRequire(import.meta.url);
const ort = require("onnxruntime-node");

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO = join(HERE, "..", "..", "..");
const ref = JSON.parse(
  readFileSync(join(HERE, "fixtures", "reconstruct_ref.json"), "utf8"),
);
const ONNX_PATH = join(REPO, "docs", "models", "decoder.onnx");
const INPUT_DIM = 297;

// runDecoder: (Float32Array length N*297, N) -> Promise<Float32Array length N>,
// batched to bound memory (mirrors what the browser must do).
async function makeRunDecoder() {
  const session = await ort.InferenceSession.create(ONNX_PATH);
  const inName = session.inputNames[0];
  const outName = session.outputNames[0];
  return async (input, N) => {
    const out = new Float32Array(N);
    const BATCH = 8192;
    for (let start = 0; start < N; start += BATCH) {
      const n = Math.min(BATCH, N - start);
      const slice = input.subarray(start * INPUT_DIM, (start + n) * INPUT_DIM);
      const tensor = new ort.Tensor("float32", slice, [n, INPUT_DIM]);
      const result = await session.run({ [inName]: tensor });
      out.set(result[outName].data, start);
    }
    return out;
  };
}

const bboxOf = (verts) => {
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const v of verts) {
    for (let a = 0; a < 3; a++) {
      if (v[a] < min[a]) min[a] = v[a];
      if (v[a] > max[a]) max[a] = v[a];
    }
  }
  return { min, max };
};

// bbox in y/z is near-symmetric for this cage, so the checks above can't see the
// final map_z_to_y swap. This deterministic check pins that wiring directly:
// with the same inputs only the final swap differs, so map=true output must equal
// swapYZ(map=false) output element-for-element.
test("localReconstruct wires the map_z_to_y output swap", async () => {
  const runDecoder = await makeRunDecoder();
  const common = {
    latentShapes: ref.cage.map(swapYZ),
    resolution: ref.resolution,
    rescale: true,
  };
  const withSwap = await localReconstruct({ ...common, mapZToY: true }, runDecoder);
  const noSwap = await localReconstruct({ ...common, mapZToY: false }, runDecoder);

  assert.equal(withSwap.vertices.length, noSwap.vertices.length);
  let maxDiff = 0;
  for (let i = 0; i < withSwap.vertices.length; i++) {
    const expected = swapYZ(noSwap.vertices[i]);
    for (let c = 0; c < 3; c++) {
      maxDiff = Math.max(maxDiff, Math.abs(withSwap.vertices[i][c] - expected[c]));
    }
  }
  assert.ok(maxDiff < 1e-9, `map=true output must equal swapYZ(map=false), maxDiff=${maxDiff}`);
});

// Adaptive coarse-to-fine evaluation must reproduce the dense-eval surface: the
// active-cell shell (tau + dilation) has to cover every surface-crossing cell.
// Small far-field floaters that dense picks up may be legitimately dropped, so
// face count gets a small tolerance while the bbox must match tightly.
test("adaptive refinement matches dense evaluation", async () => {
  const runDecoder = await makeRunDecoder();
  const common = { latentShapes: ref.cage.map(swapYZ), resolution: ref.resolution, rescale: true, mapZToY: true };
  const adaptive = await localReconstruct({ ...common, adaptive: true }, runDecoder);
  const dense = await localReconstruct({ ...common, adaptive: false }, runDecoder);

  const a = bboxOf(adaptive.vertices);
  const d = bboxOf(dense.vertices);
  let maxDelta = 0;
  for (let ax = 0; ax < 3; ax++) {
    maxDelta = Math.max(maxDelta, Math.abs(a.min[ax] - d.min[ax]), Math.abs(a.max[ax] - d.max[ax]));
  }
  assert.ok(maxDelta < 1e-3, `adaptive vs dense bbox delta ${maxDelta} exceeds 1e-3`);

  const faceDiff = Math.abs(adaptive.faces.length - dense.faces.length) / dense.faces.length;
  assert.ok(
    faceDiff < 0.02,
    `face count differs ${(faceDiff * 100).toFixed(2)}% (adaptive ${adaptive.faces.length} vs dense ${dense.faces.length})`,
  );
});

// The bbox checks also can't see the app.py:77 input swap (with rescale the mesh
// tracks the cage bbox, so a y/z-permuted cage lands in nearly the same box). Pin
// it with a spy: the cage fed to the decoder must be swapYZ(body). No ONNX needed.
test("localReconstruct feeds the app.py:77-swapped cage to the decoder", async () => {
  let captured = null;
  const spy = async (input, n) => {
    captured = input.slice(0, DECODER_INPUT_DIM); // first row = [x, y, z, cageFlat]
    return new Float32Array(n);
  };
  const body = ref.cage.map(swapYZ);
  await localReconstruct(
    { latentShapes: body, resolution: 4, rescale: true, mapZToY: true },
    spy,
  );

  const expectedCage = body.map(swapYZ).flat(); // swapYZ(body) = the raw xyz cage
  let maxDiff = 0;
  for (let m = 0; m < expectedCage.length; m++) {
    maxDiff = Math.max(maxDiff, Math.abs(captured[3 + m] - expectedCage[m]));
  }
  assert.ok(maxDiff < 1e-6, `decoder cage must be swapYZ(body), maxDiff=${maxDiff}`);
});

for (const c of ref.configs) {
  const label = `rescale=${c.rescale} map_z_to_y=${c.map_z_to_y}`;
  test(`localReconstruct bbox matches Python reconstruct() (${label})`, async () => {
    const runDecoder = await makeRunDecoder();
    // The reused viewer sends the cage in POST-body (xzy) form; localReconstruct
    // applies the app.py:77 swap internally. ref.cage is raw xyz, so pre-swap it.
    const body = ref.cage.map(swapYZ);

    const res = await localReconstruct(
      {
        latentShapes: body,
        resolution: ref.resolution,
        rescale: c.rescale,
        mapZToY: c.map_z_to_y,
      },
      runDecoder,
    );

    assert.ok(res && res.vertices.length > 0, "produced a mesh");
    const bb = bboxOf(res.vertices);

    // Same isosurface as skimage but a different MC algorithm, so allow a small
    // absolute gap; scaled by whether the mesh lives in cage-space or index-space.
    const tol = c.rescale ? 0.1 : 1.5;
    let maxDelta = 0;
    for (let a = 0; a < 3; a++) {
      maxDelta = Math.max(
        maxDelta,
        Math.abs(bb.min[a] - c.bbox_min[a]),
        Math.abs(bb.max[a] - c.bbox_max[a]),
      );
    }
    assert.ok(
      maxDelta < tol,
      `${label}: bbox delta ${maxDelta.toFixed(4)} exceeds tol ${tol}\n` +
        `  js  min ${bb.min.map((x) => x.toFixed(3))} max ${bb.max.map((x) => x.toFixed(3))}\n` +
        `  py  min ${c.bbox_min.map((x) => x.toFixed(3))} max ${c.bbox_max.map((x) => x.toFixed(3))}`,
    );
  });
}
