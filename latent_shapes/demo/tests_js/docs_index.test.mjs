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
const localHtml = readFileSync(
  join(HERE, "..", "templates", "interpolator.html"),
  "utf8",
);
const pages = [
  ["static demo", html],
  ["local demo", localHtml],
];

test("both demos activate the stacked layout on real mobile devices", () => {
  for (const [name, pageHtml] of pages) {
    assert.match(
      pageHtml,
      /<meta name="viewport" content="width=device-width, initial-scale=1"\s*\/?>/,
      `${name} must use the real mobile device width`,
    );
    assert.match(
      pageHtml,
      /@media \(max-width:\s*760px\)\s*\{\s*\.stage\s*\{\s*grid-template-columns:\s*1fr;/,
      `${name} must stack the viewport and controls on mobile`,
    );
    assert.match(
      pageHtml,
      /\.viewport\s*\{[^}]*touch-action:\s*none;/s,
      `${name} must reserve viewport gestures for the 3D interaction`,
    );
  }
});

test("both demos keep one bounded control surface on tablet and desktop", () => {
  for (const [name, pageHtml] of pages) {
    assert.match(
      pageHtml,
      /\*\s*\{\s*box-sizing:\s*border-box;\s*\}/,
      `${name} must include borders and padding in responsive widths`,
    );
    assert.match(
      pageHtml,
      /\.stage\s*\{[^}]*grid-template-columns:\s*minmax\(0,\s*1fr\)\s+268px;/s,
      `${name} must retain the tablet and desktop two-column layout`,
    );
    for (const id of [
      "resolution_slider",
      "show_latent_shape",
      "show_reconstructed_mesh",
      "show_wireframe",
      "load_random_shape_button",
      "reconstruct_button",
    ]) {
      assert.ok(pageHtml.includes(`id="${id}"`), `${name} must preserve #${id}`);
    }
    for (const id of [
      "show_latent_shape",
      "show_reconstructed_mesh",
      "show_wireframe",
    ]) {
      assert.match(
        pageHtml,
        new RegExp(`<input[^>]+id="${id}"[^>]+checked`),
        `${name} must preserve the checked default for #${id}`,
      );
    }
  }
});

test("both demos start one-shot box selection on the second press, before release", () => {
  for (const [name, pageHtml] of pages) {
    assert.match(
      pageHtml,
      /function arm_one_shot_selection\(\)\s*\{[^}]*set_selection_mode\(true,\s*true\);/s,
      `${name} must arm the existing selection mode as one-shot`,
    );
    const completeTap = pageHtml.slice(
      pageHtml.indexOf("function complete_tap(event)"),
      pageHtml.indexOf("function on_pointerdown(event)"),
    );
    const pointerDown = pageHtml.slice(
      pageHtml.indexOf("function on_pointerdown(event)"),
      pageHtml.indexOf("function on_pointermove(event)"),
    );
    assert.match(
      pointerDown,
      /const started_on_gizmo = transform_controls\.axis !== null;[\s\S]*const hit_vertex = pick_latent_vertex\(event\);[\s\S]*!started_on_gizmo[\s\S]*hit_vertex === null[\s\S]*is_double_tap\(event\.pointerType,\s*event\)[\s\S]*arm_one_shot_selection\(\);[\s\S]*selection_helper\.isDown = true;[\s\S]*selection_helper\.onSelectStart\(event\);[\s\S]*selection_pointer_id = event\.pointerId;[\s\S]*selection_box\.startPoint\.set/,
      `${name} must start the existing box-selection drag during the second press`,
    );
    assert.doesNotMatch(
      completeTap,
      /arm_one_shot_selection\(\)/,
      `${name} must not wait for the second release to arm selection`,
    );
  }
});

test("both demos consume one box drag and cancel armed mode safely", () => {
  for (const [name, pageHtml] of pages) {
    assert.match(
      pageHtml,
      /selection_helper\.onSelectStart = function \(event\)\s*\{\s*if \(!is_selection_mode \|\| !event\.isPrimary \|\| event\.button !== 0\)/,
      `${name} must keep SelectionHelper out of camera and multi-touch gestures`,
    );
    assert.match(
      pageHtml,
      /if \(active_pointer_ids\.size > 1\)\s*\{[\s\S]*if \(is_one_shot_selection\)\s*\{\s*set_selection_mode\(false\);/,
      `${name} must cancel armed selection when a second pointer starts`,
    );
    assert.match(
      pageHtml,
      /event\.target !== renderer\.domElement[\s\S]*!event\.isPrimary[\s\S]*event\.button !== 0[\s\S]*selection_pointer_id = event\.pointerId/,
      `${name} must begin box selection only from one primary canvas pointer`,
    );
    assert.match(
      pageHtml,
      /const should_exit_one_shot = is_one_shot_selection;[\s\S]*const selections = selection_box\.select\(\);[\s\S]*if \(should_exit_one_shot\)\s*\{\s*set_selection_mode\(false\);/,
      `${name} must exit one-shot mode after its box selection`,
    );
    assert.match(
      pageHtml,
      /else if \(event\.key === "Escape"\)\s*\{\s*if \(is_one_shot_selection\)\s*\{\s*set_selection_mode\(false\);/,
      `${name} must let Escape cancel one-shot mode`,
    );
  }
});

test("both demos route box-drag and vertex-tap through one selection state", () => {
  for (const [name, pageHtml] of pages) {
    assert.match(
      pageHtml,
      /function sync_selected_shape\(\)\s*\{[\s\S]*selected_indices[\s\S]*LATENT_SHAPE_SPHERE_SELECTED_COLOR[\s\S]*transform_controls\.attach\(selected_shape\[0\]\)[\s\S]*transform_controls\.detach\(\);[\s\S]*\}/,
      `${name} must have one color/index/gizmo synchronizer`,
    );
    assert.match(
      pageHtml,
      /function replace_selected_shape\(selections\)\s*\{[\s\S]*selection\.name === LATENT_SHAPE[\s\S]*sync_selected_shape\(\);[\s\S]*\}/,
      `${name} must feed box results into the shared synchronizer`,
    );
    assert.match(
      pageHtml,
      /function toggle_selected_vertex\(vertex\)\s*\{[\s\S]*selected_shape\.indexOf\(vertex\)[\s\S]*selected_shape\.splice[\s\S]*selected_shape\.push\(vertex\)[\s\S]*sync_selected_shape\(\);[\s\S]*\}/,
      `${name} must toggle one tapped vertex through the shared synchronizer`,
    );
    assert.match(
      pageHtml,
      /const selections = selection_box\.select\(\);\s*replace_selected_shape\(selections\);/,
      `${name} must replace selection from the existing SelectionBox`,
    );
    assert.match(
      pageHtml,
      /candidate\.started_on_gizmo \|\| hit_vertex !== null[\s\S]*if \(hit_vertex !== null\)\s*\{\s*toggle_selected_vertex\(hit_vertex\);/,
      `${name} must let a no-move tap toggle the vertex beneath its gizmo`,
    );
  }
});

test("both demos preserve camera controls and explain the mobile gestures", () => {
  for (const [name, pageHtml] of pages) {
    assert.match(
      pageHtml,
      /orbit_controls\.touches\.ONE = THREE\.TOUCH\.ROTATE;\s*orbit_controls\.touches\.TWO = THREE\.TOUCH\.DOLLY_PAN;/,
      `${name} must keep one-finger rotation separate from two-finger pan/zoom`,
    );
    assert.match(
      pageHtml,
      /@media \(hover:\s*none\) and \(pointer:\s*coarse\)[\s\S]*\.desktop-shortcut\s*\{\s*display:\s*none;[\s\S]*\.touch-shortcut\s*\{\s*display:\s*flex;/,
      `${name} must show touch instructions on touch-first devices`,
    );
    for (const gesture of [
      "double tap, then drag",
      "tap vertex",
      "one-finger drag",
      "two-finger drag",
      "pinch",
    ]) {
      assert.ok(pageHtml.includes(gesture), `${name} must explain ${gesture}`);
    }
    assert.match(
      pageHtml,
      /async function load_random_shape\(\)\s*\{\s*if \(is_selection_mode\)\s*\{\s*set_selection_mode\(false\);/,
      `${name} must cancel selection mode before loading another shape`,
    );
    assert.match(
      pageHtml,
      /show_latent_shape_checkbox\.addEventListener\("change", function\(\)\s*\{\s*if \(!this\.checked && is_selection_mode\)\s*\{\s*set_selection_mode\(false\);/,
      `${name} must cancel selection mode before hiding selectable vertices`,
    );
    assert.match(
      pageHtml,
      /async function reconstruct\(\)\s*\{[\s\S]*if \(is_selection_mode\)\s*\{\s*set_selection_mode\(false\);[\s\S]*is_reconstructing = true;/,
      `${name} must cancel selection mode before reconstruction locks the cage`,
    );
    assert.match(
      pageHtml,
      /function on_dragging_changed\(event\)\s*\{\s*orbit_controls\.enabled = !event\.value;/,
      `${name} must keep camera rotation disabled while the transform gizmo is dragged`,
    );
  }
});

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

test("docs/index.html stops reconstruction immediately", () => {
  const cancelReconstruct = html.slice(
    html.indexOf("function cancelReconstruct()"),
    html.indexOf("const LATENT_SHAPE"),
  );
  assert.match(
    cancelReconstruct,
    /_worker\.terminate\(\);[\s\S]*_pending\.clear\(\);[\s\S]*resolve\(\{ cancelled: true \}\);/,
    "Stop must terminate the busy worker and settle its pending request as cancelled",
  );
  assert.doesNotMatch(
    cancelReconstruct,
    /postMessage\(\{ cancel: true \}\)/,
    "Stop must not wait for a busy worker to receive a cooperative cancel message",
  );
});

test("docs/index.html replaces a stopped worker without accepting stale events", () => {
  const workerLifecycle = html.slice(
    html.indexOf("const _pending"),
    html.indexOf("const LATENT_SHAPE"),
  );
  assert.match(
    workerLifecycle,
    /function createReconstructWorker\(\)[\s\S]*const worker = new Worker\([\s\S]*return worker;[\s\S]*let _worker = createReconstructWorker\(\);/,
    "the initial and replacement workers must share one configured factory",
  );
  assert.equal(
    (workerLifecycle.match(/if \(worker !== _worker\) return;/g) || []).length,
    2,
    "both messages and errors from a replaced worker must be ignored",
  );
  assert.match(
    workerLifecycle,
    /function cancelReconstruct\(\)[\s\S]*_worker\.terminate\(\);[\s\S]*_worker = createReconstructWorker\(\);/,
    "Stop must replace the terminated worker for the next reconstruction",
  );
  assert.match(
    workerLifecycle,
    /function reconstructViaWorker\([\s\S]*_worker\.postMessage\(\{ id, params \}\);/,
    "later reconstruction requests must use the current worker",
  );
});

test("reconstruct_worker.js wires the local ONNX backend", () => {
  const worker = readFileSync(
    join(HERE, "..", "..", "..", "docs", "js", "reconstruct_worker.js"),
    "utf8",
  );
  assert.ok(worker.includes("latent_backend.js"), "imports the local backend module");
  assert.ok(worker.includes("decoder.onnx"), "loads the decoder model");
});
