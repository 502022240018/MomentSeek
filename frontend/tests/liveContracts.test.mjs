import assert from "node:assert/strict";
import test from "node:test";

import {
  normalizeColorGradingCapability,
  normalizeColorGradingTaskList,
  normalizeEntityList,
  normalizeFolderList,
  normalizeJobList,
  normalizeOrchestrationProfiles,
  normalizePlanSetResponse,
  normalizePlannerCapabilities,
  normalizeSearchResponse,
  normalizeVideoList,
} from "../src/runtimeContracts.ts";

const baseUrl = process.env.MOMENTSEEK_LIVE_BASE_URL;
const exerciseWorkflow = process.env.MOMENTSEEK_LIVE_WORKFLOW === "1";
const cases = [
  ["/api/videos", normalizeVideoList],
  ["/api/folders", normalizeFolderList],
  ["/api/jobs", normalizeJobList],
  ["/api/entities", normalizeEntityList],
  ["/api/color-grading/status", normalizeColorGradingCapability],
  ["/api/color-grading/tasks", normalizeColorGradingTaskList],
  ["/api/orchestration/profiles", normalizeOrchestrationProfiles],
  ["/api/planner-lab/capabilities", normalizePlannerCapabilities],
];

test("deployed read-only APIs satisfy frontend runtime contracts", { skip: !baseUrl }, async () => {
  for (const [path, normalize] of cases) {
    const response = await fetch(`${baseUrl}${path}`);
    assert.equal(response.ok, true, `${path} returned ${response.status}`);
    const body = await response.json();
    assert.doesNotThrow(() => normalize(body), path);
  }
});

test("deployed planner and scoped visual search satisfy runtime contracts", {
  skip: !baseUrl || !exerciseWorkflow,
}, async () => {
  const videosResponse = await fetch(`${baseUrl}/api/videos`);
  assert.equal(videosResponse.ok, true);
  const videos = normalizeVideoList(await videosResponse.json());
  const video = videos.find(item => item.indexed_modalities.includes("visual"));
  assert.ok(video, "a visual-indexed video is required for the live workflow smoke test");

  const planForm = new FormData();
  planForm.append("query_text", "人物近景");
  planForm.append("video_ids", JSON.stringify([video.id]));
  planForm.append("mode", "assist");
  const planResponse = await fetch(`${baseUrl}/api/planner-lab/plans`, { method: "POST", body: planForm });
  assert.equal(planResponse.ok, true, `planner returned ${planResponse.status}`);
  const planSet = normalizePlanSetResponse(await planResponse.json());
  assert.ok(planSet.plans.length > 0);

  const searchForm = new FormData();
  searchForm.append("query_text", "人物近景");
  searchForm.append("modalities", "visual");
  searchForm.append("video_ids", JSON.stringify([video.id]));
  searchForm.append("alpha", "0.5");
  searchForm.append("limit", "3");
  searchForm.append("planner_mode", "off");
  searchForm.append("reranker_mode", "off");
  const searchResponse = await fetch(`${baseUrl}/api/search`, { method: "POST", body: searchForm });
  assert.equal(searchResponse.ok, true, `search returned ${searchResponse.status}`);
  normalizeSearchResponse(await searchResponse.json());
});
