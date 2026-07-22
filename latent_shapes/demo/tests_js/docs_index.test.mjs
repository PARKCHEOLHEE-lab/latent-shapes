import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const HERE = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(
  join(HERE, "..", "..", "..", "docs", "index.html"),
  "utf8",
);

test("docs/index.html makes no server /api/ calls (fully static)", () => {
  assert.ok(!html.includes("/api/"), "must not reference any /api/ endpoint");
});

test("docs/index.html preserves the interactive controls", () => {
  for (const id of [
    "resolution_slider",
    "show_latent_shape",
    "show_reconstructed_mesh",
    "show_wireframe",
    "load_random_shape_button",
    "reconstruct_button",
  ]) {
    assert.ok(html.includes(`id="${id}"`), `control #${id} must be present`);
  }
});

test("docs/index.html removes the ensure_watertight control (no browser pcu)", () => {
  assert.ok(!html.includes("ensure_watertight"), "ensure_watertight must be removed");
});

test("docs/index.html gives the faces-only mode a shaded white surface (wireframe off)", () => {
  assert.ok(html.includes("MeshMatcapMaterial"), "solid uses a shaded matcap (white clay) material when the wireframe is hidden");
});

test("docs/index.html offloads reconstruction to a worker + loads latent data", () => {
  assert.ok(html.includes("reconstruct_worker.js"), "spawns the reconstruction worker");
  assert.ok(html.includes("latent_shapes.json"), "loads the latent shapes data");
});

test("reconstruct_worker.js wires the local ONNX backend", () => {
  const worker = readFileSync(
    join(HERE, "..", "..", "..", "docs", "js", "reconstruct_worker.js"),
    "utf8",
  );
  assert.ok(worker.includes("latent_backend.js"), "imports the local backend module");
  assert.ok(worker.includes("decoder.onnx"), "loads the decoder model");
});
