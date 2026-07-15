// Runs the whole reconstruction (grid, ONNX inference, marching cubes, post-process)
// off the main thread so the UI never freezes. The numeric pipeline is the already
// tested localReconstruct(); this worker only adds the ONNX session + message transport.
//
// Note: import maps are document-scoped and do NOT apply in workers, so onnxruntime-web
// is imported by its full CDN URL here (not the bare "onnxruntime-web" specifier).
// 1.21.0+ is required: 1.20.x ships a naive scalar WebGPU Gemm kernel that runs
// nn.Linear layers at ~1% GPU utilization (fixed upstream in onnxruntime#22706).
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.27.0/dist/ort.webgpu.bundle.min.mjs";
import {
  localReconstruct, swapYZ, postProcessMesh, mapMeshToClient,
  DECODER_INPUT_DIM, MC_LEVEL,
} from "./latent_backend.js?v=5";
import { marchingCubes } from "./marching_cubes.js?v=3";

// GitHub Pages can't send COOP/COEP, so SharedArrayBuffer threads are unavailable.
ort.env.wasm.numThreads = 1;

// Feeding a CONSTANT input shape is the key perf fix: it lets ORT compile the
// WebGPU pipelines ONCE (cached + graph-captured) instead of recompiling every
// run, which is the likely cause of the ~45s. Every batch is padded to this size.
const FIXED_BATCH = 65536;

let sessionPromise = null;
let usedProvider = null;

function getSession() {
  if (sessionPromise === null) {
    const modelUrl = new URL("../models/decoder.onnx", import.meta.url).href;

    // A warmup run compiles the pipelines now (at load) rather than on the first
    // reconstruct, and — with graph capture — records the command sequence to replay.
    const create = async (opts, label) => {
      const session = await ort.InferenceSession.create(modelUrl, opts);
      const warm = new ort.Tensor(
        "float32", new Float32Array(FIXED_BATCH * DECODER_INPUT_DIM), [FIXED_BATCH, DECODER_INPUT_DIM],
      );
      await session.run({ [session.inputNames[0]]: warm });
      usedProvider = label;
      return session;
    };

    // freeDimensionOverrides fixes the exported "n" batch axis -> static shape, which
    // both caches pipelines and permits graph capture. Graph capture "does not always
    // work" per the docs, so degrade gracefully: graphCapture -> webgpu -> wasm.
    const webgpu = { executionProviders: ["webgpu"], freeDimensionOverrides: { n: FIXED_BATCH } };
    sessionPromise = create({ ...webgpu, enableGraphCapture: true }, "webgpu+graphCapture")
      .catch(() => create(webgpu, "webgpu"))
      .catch(() => create({ executionProviders: ["wasm"] }, "wasm"));
  }
  return sessionPromise;
}

// runDecoder(input: Float32Array[N*297], N) -> Promise<Float32Array[N]>.
// Each batch is padded to FIXED_BATCH rows so the input shape never varies and the
// cached (graph-captured) pipelines are reused instead of recompiled.
async function runDecoder(input, N) {
  const session = await getSession();
  const inName = session.inputNames[0];
  const outName = session.outputNames[0];
  const out = new Float32Array(N);
  const rowFloats = DECODER_INPUT_DIM;
  for (let start = 0; start < N; start += FIXED_BATCH) {
    const n = Math.min(FIXED_BATCH, N - start);
    const buf = new Float32Array(FIXED_BATCH * rowFloats); // zero-padded tail rows
    buf.set(input.subarray(start * rowFloats, (start + n) * rowFloats));
    const tensor = new ort.Tensor("float32", buf, [FIXED_BATCH, rowFloats]);
    const result = await session.run({ [inName]: tensor });
    out.set(result[outName].data.subarray(0, n), start);
  }
  return out;
}

self.onmessage = async (event) => {
  const { id, params } = event.data;
  try {
    const t0 = performance.now();
    await getSession();
    const t1 = performance.now();

    // Progressive coarse-to-fine previews: after each non-final refinement level,
    // polygonize that level's full field and run it through the SAME transform the
    // final mesh gets (rescale + centralize onto the cage), so the preview already
    // sits where the final mesh will land instead of floating at the grid origin.
    const mapZToY = params.mapZToY !== false;
    const cageXyz = params.latentShapes.map(swapYZ);

    // Polygonize a (possibly partially refined) field and post it as a preview.
    // Above 128 the field is strided down first so preview cost stays ~100ms
    // flat instead of seconds — previews are transient, coarser is fine.
    // Faces are split into "base" (unchanged since the last preview — drawn in
    // the final mesh's style) and "changed" (freshly refined — drawn highlighted),
    // so the highlight itself sweeps across the surface as refinement progresses.
    let prevPreview = null; // { f, mcR } of the last posted preview
    const postPreview = (field, R) => {
      let f;
      let mcR = R;
      if (R > 128) {
        const stride = Math.ceil(R / 128);
        mcR = Math.floor((R - 1) / stride) + 1;
        f = new Float32Array(mcR * mcR * mcR);
        for (let i = 0; i < mcR; i++) {
          for (let j = 0; j < mcR; j++) {
            for (let k = 0; k < mcR; k++) {
              f[(i * mcR + j) * mcR + k] = field[((i * stride) * R + j * stride) * R + k * stride];
            }
          }
        }
      } else {
        f = Float32Array.from(field);
      }

      // per-point change mask vs the last preview. Inherited (unrefined) values
      // are bit-exact copies, so float equality is a reliable "unchanged" signal.
      const n = mcR * mcR * mcR;
      const changedPoint = new Uint8Array(n);
      if (prevPreview === null) {
        changedPoint.fill(1);
      } else {
        const prev = prevPreview.f;
        const prevR = prevPreview.mcR;
        if (prevR === mcR) {
          for (let p = 0; p < n; p++) changedPoint[p] = f[p] !== prev[p] ? 1 : 0;
        } else {
          // level transition: compare against the upsampled previous preview,
          // using the same nearest mapping the refinement fill uses
          for (let i = 0; i < mcR; i++) {
            const pi = Math.round((i * (prevR - 1)) / (mcR - 1));
            for (let j = 0; j < mcR; j++) {
              const pj = Math.round((j * (prevR - 1)) / (mcR - 1));
              for (let k = 0; k < mcR; k++) {
                const pk = Math.round((k * (prevR - 1)) / (mcR - 1));
                const p = (i * mcR + j) * mcR + k;
                changedPoint[p] = f[p] !== prev[(pi * prevR + pj) * prevR + pk] ? 1 : 0;
              }
            }
          }
        }
      }
      prevPreview = { f, mcR };

      const mc = marchingCubes([mcR, mcR, mcR], f, MC_LEVEL);
      if (mc.vertices.length === 0) return;

      // a triangle is "changed" if any corner point of its generating cell changed
      const cellChanged = (i, j, k) => {
        for (let d = 0; d < 8; d++) {
          if (changedPoint[((i + (d >> 2)) * mcR + (j + ((d >> 1) & 1))) * mcR + (k + (d & 1))]) return true;
        }
        return false;
      };
      const facesBase = [];
      const facesChanged = [];
      for (const face of mc.faces) {
        const [a, b, c] = face.map((v) => mc.vertices[v]);
        const i = Math.min(mcR - 2, Math.floor((a[0] + b[0] + c[0]) / 3));
        const j = Math.min(mcR - 2, Math.floor((a[1] + b[1] + c[1]) / 3));
        const k = Math.min(mcR - 2, Math.floor((a[2] + b[2] + c[2]) / 3));
        (cellChanged(i, j, k) ? facesChanged : facesBase).push(face);
      }

      const centered = postProcessMesh(mc.vertices, cageXyz, { rescale: params.rescale !== false });
      self.postMessage({
        id,
        progress: {
          preview: {
            vertices: mapMeshToClient(centered, mapZToY),
            faces_base: facesBase,
            faces_changed: facesChanged,
          },
        },
      });
    };

    const onLevel = ({ field, resolution }) => postPreview(field, resolution);
    // per decoder chunk: counter tick, plus a live sharpening preview during the
    // final level (the longest phase used to freeze between level 2 and done)
    const onChunk = ({ level, levelCount, rowsDone, rowsTotal, field, resolution }) => {
      self.postMessage({ id, progress: { level, levelCount, rowsDone, rowsTotal } });
      if (level === levelCount && rowsDone < rowsTotal) {
        postPreview(field, resolution);
      }
    };

    const result = await localReconstruct({ ...params, onLevel, onChunk }, runDecoder);
    const t2 = performance.now();
    self.postMessage({
      id, result,
      diag: {
        provider: usedProvider,
        sessionMs: Math.round(t1 - t0),
        reconstructMs: Math.round(t2 - t1),
      },
    });
  } catch (err) {
    self.postMessage({ id, error: String(err && err.message ? err.message : err) });
  }
};
