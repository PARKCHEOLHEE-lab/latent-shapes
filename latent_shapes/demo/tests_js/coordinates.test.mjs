import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { swapYZ, mapMeshToClient } from "../../../docs/js/latent_backend.js";

const HERE = dirname(fileURLToPath(import.meta.url));
const g = JSON.parse(
  readFileSync(join(HERE, "fixtures", "coord_golden.json"), "utf8"),
);

const maxDiff = (a, b) => {
  let m = 0;
  for (let i = 0; i < a.length; i++)
    for (let c = 0; c < 3; c++) m = Math.max(m, Math.abs(a[i][c] - b[i][c]));
  return m;
};

test("swapYZ swaps components 1 and 2 (y<->z), not 0 and 1", () => {
  assert.deepEqual(swapYZ(g.swap_in), g.swap_out); // [1,2,3] -> [1,3,2]
});

test("mapMeshToClient applies model.py:249 swap iff map_z_to_y", () => {
  assert.ok(maxDiff(mapMeshToClient(g.mesh, true), g.mesh_client_map_true) < 1e-9);
  assert.ok(maxDiff(mapMeshToClient(g.mesh, false), g.mesh_client_map_false) < 1e-9);
});

test("net display double-swap: map_z_to_y=true cancels, false stays single", () => {
  // html:817 display reorder is the same swapYZ; net = display(server_output)
  const displayTrue = mapMeshToClient(g.mesh, true).map(swapYZ);
  const displayFalse = mapMeshToClient(g.mesh, false).map(swapYZ);
  assert.ok(maxDiff(displayTrue, g.net_display_map_true) < 1e-9, "map_true must round-trip to mesh");
  assert.ok(maxDiff(displayFalse, g.net_display_map_false) < 1e-9, "map_false must be single swap");
});
