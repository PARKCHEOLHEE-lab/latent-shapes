import { test } from "node:test";
import assert from "node:assert/strict";

import { mergedFaces, readSseFrames } from "../js/stream_client.mjs";

test("readSseFrames parses complete SSE data frames and keeps the partial remainder", () => {
  const buffer =
    'data: {"level":1,"final":false}\n\n' +
    'data: {"level":2,"final":false}\n\n' +
    'data: {"lev'; // a third frame has only partially arrived

  const { events, rest } = readSseFrames(buffer);

  assert.equal(events.length, 2); // both complete frames parsed
  assert.equal(events[0].level, 1);
  assert.equal(events[1].level, 2);
  assert.equal(rest, 'data: {"lev'); // the incomplete frame is retained for the next chunk
});

test("readSseFrames returns no events until a frame terminator arrives", () => {
  const { events, rest } = readSseFrames('data: {"level":1,"final":true}');
  assert.equal(events.length, 0); // no "\n\n" yet -> nothing complete
  assert.equal(rest, 'data: {"level":1,"final":true}');
});

test("mergedFaces concatenates base + changed faces for the final mesh", () => {
  assert.deepEqual(
    mergedFaces({ faces_base: [[0, 1, 2]], faces_changed: [[3, 4, 5]] }),
    [[0, 1, 2], [3, 4, 5]],
  );
  assert.deepEqual(mergedFaces({ faces_base: [], faces_changed: [[1, 2, 3]] }), [[1, 2, 3]]);
  assert.deepEqual(mergedFaces({ faces_base: [[7, 8, 9]], faces_changed: [] }), [[7, 8, 9]]);
});
