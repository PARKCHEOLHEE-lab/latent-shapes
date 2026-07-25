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

test("both demos leave box-selection activation to pointer gestures, not Shift", () => {
  for (const [name, pageHtml] of pages) {
    const onKeyDown = pageHtml.slice(
      pageHtml.indexOf("function on_keydown(event)"),
      pageHtml.indexOf("// mouse position ->"),
    );
    const runKeyDown = new Function(
      "event",
      `
        let is_one_shot_selection = false;
        const calls = { selectionModes: [], replacements: [], resets: 0 };
        function set_selection_mode(...args) {
          calls.selectionModes.push(args);
        }
        function replace_selected_shape(selections) {
          calls.replacements.push(selections);
        }
        function reset_camera() {
          calls.resets += 1;
        }
        ${onKeyDown}
        on_keydown(event);
        return calls;
      `,
    );

    assert.deepEqual(
      runKeyDown({ key: "Shift" }).selectionModes,
      [],
      `${name} must not enter selection mode from Shift`,
    );
    assert.deepEqual(
      runKeyDown({ key: "Escape" }).replacements,
      [[]],
      `${name} must keep Escape deselection`,
    );
    assert.equal(
      runKeyDown({ key: "r" }).resets,
      1,
      `${name} must keep the camera-reset shortcut`,
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
      /function on_keydown\(event\)\s*\{\s*if \(event\.key === "Escape"\)\s*\{\s*if \(is_one_shot_selection\)\s*\{\s*set_selection_mode\(false\);/,
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

test("both demos clear selection only after a stationary empty-canvas release", () => {
  for (const [name, pageHtml] of pages) {
    const completeTap = pageHtml.slice(
      pageHtml.indexOf("function complete_tap(event)"),
      pageHtml.indexOf("function on_pointerdown(event)"),
    );
    const runCompleteTap = new Function(
      "candidate",
      "event",
      "pointerCount",
      "reconstructing",
      "hitVertex",
      `
        let tap_candidate = candidate;
        let last_empty_tap = null;
        const active_pointer_ids = { size: pointerCount };
        const is_reconstructing = reconstructing;
        const TAP_MOVE_THRESHOLD = 8;
        const calls = { replacements: [], toggled: [] };
        function pick_latent_vertex() { return hitVertex; }
        function replace_selected_shape(selections) {
          calls.replacements.push(selections);
        }
        function toggle_selected_vertex(vertex) {
          calls.toggled.push(vertex);
        }
        ${completeTap}
        complete_tap(event);
        return { calls, last_empty_tap };
      `,
    );
    const candidate = {
      pointer_id: 1,
      pointer_type: "touch",
      x: 100,
      y: 100,
      moved: false,
      started_on_gizmo: false,
    };
    const event = {
      pointerId: 1,
      pointerType: "touch",
      clientX: 100,
      clientY: 100,
      timeStamp: 250,
    };

    const emptyTap = runCompleteTap(candidate, event, 1, false, null);
    assert.deepEqual(
      emptyTap.calls.replacements,
      [[]],
      `${name} must clear through the shared selection path`,
    );
    assert.deepEqual(
      emptyTap.last_empty_tap,
      { time: 250, x: 100, y: 100, pointer_type: "touch" },
      `${name} must retain the empty tap for double-tap selection`,
    );

    const ignoredCases = [
      ["observed camera drag", { ...candidate, moved: true }, event, 1, false, null],
      ["release 9px from press", candidate, { ...event, clientX: 109 }, 1, false, null],
      ["two-pointer gesture", candidate, event, 2, false, null],
      ["reconstruction", candidate, event, 1, true, null],
      ["transform gizmo", { ...candidate, started_on_gizmo: true }, event, 1, false, null],
    ];
    for (const [gesture, gestureCandidate, gestureEvent, pointerCount, reconstructing, hitVertex] of ignoredCases) {
      const result = runCompleteTap(
        gestureCandidate,
        gestureEvent,
        pointerCount,
        reconstructing,
        hitVertex,
      );
      assert.deepEqual(
        result.calls.replacements,
        [],
        `${name} must preserve selection during ${gesture}`,
      );
      assert.equal(
        result.last_empty_tap,
        null,
        `${name} must not seed double-tap selection during ${gesture}`,
      );
    }

    const vertex = { name: "latent-shape" };
    const vertexTap = runCompleteTap(candidate, event, 1, false, vertex);
    assert.deepEqual(
      vertexTap.calls.replacements,
      [],
      `${name} must not clear before toggling a vertex`,
    );
    assert.deepEqual(
      vertexTap.calls.toggled,
      [vertex],
      `${name} must preserve vertex toggling`,
    );
    assert.ok(
      completeTap.indexOf("replace_selected_shape([]);")
        < completeTap.indexOf("last_empty_tap = {"),
      `${name} must clear before storing the next empty tap`,
    );
  }
});

test("both demos show concise desktop and touch shortcut maps", () => {
  for (const [name, pageHtml] of pages) {
    const shortcuts = [
      ...pageHtml.matchAll(
        /<div class="shortcut (desktop|touch)-shortcut"><span class="key">([^<]+)<\/span><span class="act">([^<]+)<\/span><\/div>/g,
      ),
    ].map((match) => [match[1], match[2], match[3]]);
    assert.deepEqual(
      shortcuts.filter(([device]) => device === "desktop"),
      [
        ["desktop", "double-click + drag", "select"],
        ["desktop", "click vertex", "add / remove"],
        ["desktop", "click empty", "deselect"],
        ["desktop", "ctrl + drag", "pan"],
        ["desktop", "drag", "rotate"],
        ["desktop", "esc", "deselect"],
        ["desktop", "r", "reset camera"],
      ],
      `${name} must show the concise desktop map without Shift selection`,
    );
    assert.deepEqual(
      shortcuts.filter(([device]) => device === "touch"),
      [
        ["touch", "double tap, then drag", "select"],
        ["touch", "tap vertex", "add / remove"],
        ["touch", "tap empty", "deselect"],
        ["touch", "one-finger drag", "rotate"],
        ["touch", "two-finger drag", "pan"],
        ["touch", "pinch", "zoom"],
      ],
      `${name} must show the concise touch map`,
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
const worker = readFileSync(
  join(HERE, "..", "..", "..", "docs", "js", "reconstruct_worker.js"),
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
  assert.ok(
    html.includes("reconstruct_worker.js?v=9"),
    "spawns the current reconstruction worker",
  );
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
  assert.ok(worker.includes("latent_backend.js"), "imports the local backend module");
  assert.ok(worker.includes("decoder.onnx?v=2"), "loads the current decoder model");
});

test("reconstruct_worker.js uses the measured 32K inference batch", () => {
  assert.match(worker, /const FIXED_BATCH = 32768;/);
  assert.ok(
    worker.includes("freeDimensionOverrides: { n: FIXED_BATCH }"),
    "the static WebGPU shape must match the measured batch",
  );
  assert.ok(
    worker.includes("chunkRows: FIXED_BATCH"),
    "preview progress must follow each completed inference batch",
  );
});
