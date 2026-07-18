// Client for the /api/reconstruct/stream Server-Sent Events feed. Pure parsing helpers
// (no DOM), so interpolator.html can import them and node --test can cover them.

// Pull every complete "data: <json>\n\n" frame out of an accumulating buffer, returning
// the parsed events plus the leftover partial frame (a frame can split across network
// chunks, so the remainder must be carried into the next read).
// The merged final mesh is base + freshly-refined faces (no highlight once done).
export function mergedFaces(mesh) {
  return (mesh.faces_base || []).concat(mesh.faces_changed || []);
}

export function readSseFrames(buffer) {
  const events = [];
  let idx;
  while ((idx = buffer.indexOf("\n\n")) !== -1) {
    const frame = buffer.slice(0, idx);
    buffer = buffer.slice(idx + 2);
    const dataLine = frame.split("\n").find((line) => line.startsWith("data:"));
    if (dataLine) events.push(JSON.parse(dataLine.slice(5).trim()));
  }
  return { events, rest: buffer };
}
