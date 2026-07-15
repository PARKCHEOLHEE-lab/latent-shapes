import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { buildGrid } from "../../../docs/js/latent_backend.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const golden = JSON.parse(
  readFileSync(join(HERE, "fixtures", "grid_golden.json"), "utf8"),
);

test("buildGrid matches torch linspace x meshgrid(ij) reshape(-1,3) ordering", () => {
  const points = buildGrid(golden.resolution, golden.bounds);

  assert.equal(points.length, golden.points.length, "grid point count");

  let maxDiff = 0;
  for (let i = 0; i < golden.points.length; i++) {
    for (let c = 0; c < 3; c++) {
      maxDiff = Math.max(maxDiff, Math.abs(points[i][c] - golden.points[i][c]));
    }
  }
  assert.ok(
    maxDiff < 1e-6,
    `buildGrid vs torch meshgrid max|diff|=${maxDiff.toExponential(2)} exceeds 1e-6`,
  );
});
