import assert from "node:assert/strict";
import test from "node:test";

import {
  normalizePlannerExecution,
  normalizeSearchResponse,
  normalizeVideoList,
  readJsonResponse,
} from "../src/runtimeContracts.ts";

test("200 HTML response is rejected instead of becoming an empty object", async () => {
  const response = new Response("<html>proxy error</html>", { status: 200 });
  await assert.rejects(() => readJsonResponse(response), /无法解析的响应/);
});

test("top-level video contract errors are rejected", () => {
  assert.throws(() => normalizeVideoList({}), /videos 必须是数组/);
});

test("optional video fields are normalized without hiding top-level errors", () => {
  const [video] = normalizeVideoList([{ id: "video-1", name: "demo" }]);
  assert.deepEqual(video.indexed_modalities, []);
  assert.deepEqual(video.folder_ids, []);
  assert.deepEqual(video.folders, []);
  assert.equal(video.fps, 0);
  assert.equal(video.duration, 0);
});

test("search results with missing nested arrays remain renderable", () => {
  const response = normalizeSearchResponse({
    results: [{ video_id: "video-1", video_name: "demo", start_time: 2, end_time: 4 }],
  });
  assert.equal(response.results.length, 1);
  assert.deepEqual(response.results[0].modalities, []);
  assert.deepEqual(response.results[0].evidence, []);
  assert.equal(response.results[0].start_time, 2);
  assert.equal(response.results[0].score, 0);
});

test("missing or degenerate result time ranges fail closed", () => {
  assert.throws(
    () => normalizeSearchResponse({ results: [{ video_id: "video-1" }] }),
    /start_time 必须是有限数值/,
  );
  assert.throws(
    () => normalizeSearchResponse({
      results: [{ video_id: "video-1", start_time: 0, end_time: 0 }],
    }),
    /时间范围无效/,
  );
});

test("planner execution receives stable arrays and finite defaults", () => {
  const execution = normalizePlannerExecution({
    execution_id: "execution-1",
    plan: { plan_id: "balanced", steps: [] },
    results: [],
  });
  assert.deepEqual(execution.trace, []);
  assert.deepEqual(execution.results, []);
  assert.equal(execution.elapsed_seconds, 0);
  assert.equal(execution.executed_steps, 0);
});
