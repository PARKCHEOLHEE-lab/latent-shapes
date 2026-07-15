import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { postProcessMesh } from "../../../docs/js/latent_backend.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const loadGolden = (name) =>
  JSON.parse(readFileSync(join(HERE, "fixtures", name), "utf8"));

for (const name of [
  "postprocess_rescale_true.json",
  "postprocess_rescale_false.json",
]) {
  test(`postProcessMesh reproduces model.py bounds-midpoint centralize (${name})`, () => {
    const g = loadGolden(name);
    const out = postProcessMesh(g.vertices_in, g.cage, { rescale: g.rescale });

    assert.equal(out.length, g.vertices_out.length, "vertex count");
    let maxDiff = 0;
    for (let i = 0; i < g.vertices_out.length; i++) {
      for (let c = 0; c < 3; c++) {
        maxDiff = Math.max(maxDiff, Math.abs(out[i][c] - g.vertices_out[i][c]));
      }
    }
    assert.ok(
      maxDiff < 1e-6,
      `postProcessMesh vs model.py max|diff|=${maxDiff.toExponential(2)} exceeds 1e-6`,
    );
  });
}
