from __future__ import annotations

import ast
import base64
import concurrent.futures
import json
import math
import mimetypes
import os
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.db import Catalog
from app.media import extract_video_frame
from app.search import SearchEngine
from app.settings import Settings


ALLOWED_MODALITIES = {"visual", "face", "asr", "ocr"}


class OrchestrationError(RuntimeError):
    pass


class ProviderSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    provider_type: Literal["openai_compatible"] = Field(alias="type")
    base_url: str
    base_url_env: str | None = None
    model: str
    model_env: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = Field(default=180.0, gt=0, le=900)
    extra_body: dict[str, Any] = Field(default_factory=dict)


class PlannerSpec(BaseModel):
    provider: str
    prompt_path: str
    prompt_version: str
    max_tokens: int = Field(default=256, ge=16, le=2048)
    temperature: float = Field(default=0.0, ge=0, le=2)


class RerankerSpec(BaseModel):
    provider: str
    prompt_path: str
    prompt_version: str
    concurrency: int = Field(default=4, ge=1, le=16)
    default_top_n: int = Field(default=20, ge=1, le=100)
    default_frame_count: int = Field(default=4, ge=0, le=16)
    max_visual_pixels: int = Field(default=262144, ge=16384)
    positive_label: str = "Yes"
    negative_label: str = "No"
    positive_token_id: int | None = None
    negative_token_id: int | None = None


class ProfileSpec(BaseModel):
    description: str = ""
    planner: PlannerSpec | None = None
    reranker: RerankerSpec | None = None


class OrchestrationRegistry(BaseModel):
    schema_version: int = 1
    providers: dict[str, ProviderSpec]
    profiles: dict[str, ProfileSpec]


class RerankPlan(BaseModel):
    enabled: bool = True
    strategy: Literal["multimodal", "text"] = "multimodal"
    top_n: int = Field(default=20, ge=1, le=100)
    frame_count: int = Field(default=4, ge=0, le=16)
    score_weight: float = Field(default=0.7, ge=0, le=1)


class RetrievalPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query_intent: str = "general"
    modalities: list[str]
    alpha: float = Field(default=0.5, ge=0, le=1)
    visual_profile: Literal["recall", "balanced", "precision"] = "balanced"
    candidate_limit: int = Field(default=24, ge=1, le=100)
    result_limit: int = Field(default=24, ge=1, le=100)
    merge_gap: float = Field(default=2.0, ge=0, le=15)
    max_result_seconds: float = Field(default=15.0, ge=1, le=120)
    channel_limits: dict[str, int] = Field(default_factory=dict)
    rerank: RerankPlan = Field(default_factory=RerankPlan)
    rationale: list[str] = Field(default_factory=list)

    @field_validator("modalities")
    @classmethod
    def validate_modalities(cls, value: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(str(item).strip().lower() for item in value))
        if not normalized or any(item not in ALLOWED_MODALITIES for item in normalized):
            raise ValueError("planner modalities contain an unsupported retrieval channel")
        return normalized

    @model_validator(mode="after")
    def validate_limits(self):
        invalid_channels = set(self.channel_limits) - ALLOWED_MODALITIES
        if invalid_channels:
            raise ValueError(
                f"unsupported channel_limits: {sorted(invalid_channels)}"
            )
        self.channel_limits = {
            name: max(1, min(300, int(value)))
            for name, value in self.channel_limits.items()
        }
        self.result_limit = min(self.result_limit, self.candidate_limit)
        self.rerank.top_n = min(self.rerank.top_n, self.candidate_limit)
        if self.rerank.strategy == "text":
            self.rerank.frame_count = 0
        return self


def _extract_json_object(value: str) -> dict[str, Any]:
    text = value.strip()
    if text.startswith("```"):
        text = text.removeprefix("```json").removeprefix("```")
        text = text.rsplit("```", 1)[0].strip()
    candidate = text
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as json_error:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise OrchestrationError("model response does not contain a JSON object")
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            # Some OpenAI-compatible servers occasionally return a Python
            # literal despite JSON mode. literal_eval accepts only literals,
            # not executable expressions, and keeps fail-open robust.
            try:
                parsed = ast.literal_eval(candidate)
            except (SyntaxError, ValueError) as exc:
                preview = " ".join(candidate[:200].splitlines())
                raise OrchestrationError(
                    f"planner response is not valid JSON: {preview}"
                ) from json_error
    if not isinstance(parsed, dict):
        raise OrchestrationError("model response must be a JSON object")
    return parsed


def _render_prompt(template: str, values: dict[str, Any]) -> str:
    rendered = template
    for key, value in values.items():
        if not isinstance(value, str):
            value = json.dumps(value, ensure_ascii=False)
        rendered = rendered.replace("{{" + key + "}}", value)
    return rendered


class OpenAICompatibleProvider:
    def __init__(self, name: str, spec: ProviderSpec):
        self.name = name
        self.spec = spec

    @property
    def base_url(self) -> str:
        return (
            os.getenv(self.spec.base_url_env, self.spec.base_url)
            if self.spec.base_url_env
            else self.spec.base_url
        )

    @property
    def model(self) -> str:
        return (
            os.getenv(self.spec.model_env, self.spec.model)
            if self.spec.model_env
            else self.spec.model
        )

    @property
    def descriptor(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "type": self.spec.provider_type,
            "model": self.model,
        }

    def chat(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        body = {
            "model": self.model,
            **self.spec.extra_body,
            **payload,
        }
        headers = {"Content-Type": "application/json"}
        if self.spec.api_key_env:
            api_key = os.getenv(self.spec.api_key_env)
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(
            self.base_url.rstrip("/") + "/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers=headers,
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(
                request, timeout=self.spec.timeout_seconds
            ) as response:
                raw = response.read()
        except urllib.error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            raise OrchestrationError(
                f"{self.name} returned HTTP {exc.code}: {details[-1200:]}"
            ) from exc
        except (OSError, TimeoutError) as exc:
            raise OrchestrationError(f"{self.name} request failed: {exc}") from exc
        elapsed = time.perf_counter() - started
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise OrchestrationError(f"{self.name} returned invalid JSON") from exc
        return value, elapsed


class SearchOrchestrator:
    def __init__(self, settings: Settings, catalog: Catalog, search_engine: SearchEngine):
        self.settings = settings
        self.catalog = catalog
        self.search_engine = search_engine
        self._registry: OrchestrationRegistry | None = None
        self._config_path: Path | None = None
        self._providers: dict[str, OpenAICompatibleProvider] = {}
        self._trace_lock = threading.Lock()

    def _load_registry(self) -> OrchestrationRegistry:
        path = self.settings.resolve_path(self.settings.orchestration_config_path)
        if self._registry is not None and path == self._config_path:
            return self._registry
        if not path.is_file():
            raise OrchestrationError(f"orchestration config does not exist: {path}")
        registry = OrchestrationRegistry.model_validate_json(
            path.read_text(encoding="utf-8")
        )
        self._registry = registry
        self._config_path = path
        self._providers = {
            name: OpenAICompatibleProvider(name, spec)
            for name, spec in registry.providers.items()
        }
        return registry

    def _profile(self, name: str | None) -> tuple[str, ProfileSpec]:
        registry = self._load_registry()
        profile_name = name or self.settings.orchestration_profile
        try:
            return profile_name, registry.profiles[profile_name]
        except KeyError as exc:
            raise OrchestrationError(
                f"unknown orchestration profile: {profile_name}"
            ) from exc

    def _provider(self, name: str) -> OpenAICompatibleProvider:
        try:
            return self._providers[name]
        except KeyError as exc:
            raise OrchestrationError(f"unknown model provider: {name}") from exc

    def _prompt(self, relative_path: str) -> str:
        if self._config_path is None:
            raise OrchestrationError("orchestration registry has not been loaded")
        path = Path(relative_path)
        if not path.is_absolute():
            path = self._config_path.parent / path
        if not path.is_file():
            raise OrchestrationError(f"orchestration prompt does not exist: {path}")
        return path.read_text(encoding="utf-8")

    def profiles(self) -> dict[str, Any]:
        if not self.settings.orchestration_enabled:
            return {
                "enabled": False,
                "default_profile": self.settings.orchestration_profile,
                "profiles": [],
            }
        registry = self._load_registry()
        values = []
        for name, profile in registry.profiles.items():
            values.append(
                {
                    "name": name,
                    "description": profile.description,
                    "planner": self._component_descriptor(profile.planner),
                    "reranker": self._component_descriptor(profile.reranker),
                }
            )
        return {
            "enabled": True,
            "default_profile": self.settings.orchestration_profile,
            "profiles": values,
        }

    def _component_descriptor(
        self, component: PlannerSpec | RerankerSpec | None
    ) -> dict[str, Any] | None:
        if component is None:
            return None
        provider = self._provider(component.provider)
        return {
            **provider.descriptor,
            "prompt_version": component.prompt_version,
        }

    @staticmethod
    def fallback_plan(
        modalities: list[str],
        alpha: float,
        limit: int,
        *,
        rerank_enabled: bool = False,
    ) -> RetrievalPlan:
        candidate_limit = max(1, min(100, limit))
        return RetrievalPlan(
            query_intent="legacy_explicit",
            modalities=modalities,
            alpha=max(0, min(1, alpha)),
            candidate_limit=candidate_limit,
            result_limit=candidate_limit,
            rerank=RerankPlan(
                enabled=rerank_enabled,
                top_n=min(20, candidate_limit),
                frame_count=4,
            ),
            rationale=["Use explicit request parameters as a deterministic fallback."],
        )

    def _available_modalities(self, video_ids: list[str] | None) -> list[str]:
        allowed_ids = set(video_ids or [])
        modalities: set[str] = set()
        for video in self.catalog.list_videos():
            if allowed_ids and video["id"] not in allowed_ids:
                continue
            modalities.update(video.get("indexed_modalities") or [])
        return sorted(modalities & ALLOWED_MODALITIES)

    def _run_planner(
        self,
        profile: ProfileSpec,
        query: str,
        requested_modalities: list[str],
        available_modalities: list[str],
        alpha: float,
        limit: int,
        has_query_image: bool,
    ) -> tuple[RetrievalPlan, dict[str, Any]]:
        if profile.planner is None:
            raise OrchestrationError("selected profile has no planner")
        spec = profile.planner
        provider = self._provider(spec.provider)
        prompt = self._prompt(spec.prompt_path)
        context = {
            "query": query,
            "requested_modalities": requested_modalities,
            "available_modalities": available_modalities,
            "has_query_image": has_query_image,
            "request_alpha": alpha,
            "request_limit": limit,
        }
        response, elapsed = provider.chat(
            {
                "messages": [
                    {"role": "system", "content": prompt},
                    {
                        "role": "user",
                        "content": json.dumps(context, ensure_ascii=False),
                    },
                ],
                "temperature": spec.temperature,
                "max_tokens": spec.max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "retrieval_plan",
                        "schema": RetrievalPlan.model_json_schema(),
                        "strict": True,
                    },
                },
                "chat_template_kwargs": {"enable_thinking": False},
            }
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise OrchestrationError("planner response has no message content") from exc
        plan = RetrievalPlan.model_validate(_extract_json_object(content))
        permitted = set(requested_modalities) & set(available_modalities)
        plan.modalities = [item for item in plan.modalities if item in permitted]
        if not plan.modalities:
            plan.modalities = [
                item for item in requested_modalities if item in available_modalities
            ]
        if not plan.modalities:
            raise OrchestrationError("none of the requested modalities are indexed")
        # The request limit is a hard latency/cost ceiling.  The prompt asks the
        # model to respect it, and the server enforces it independently.
        plan.candidate_limit = min(plan.candidate_limit, limit)
        per_channel_ceiling = min(300, limit * 3)
        plan.channel_limits = {
            name: min(value, per_channel_ceiling)
            for name, value in plan.channel_limits.items()
            if name in plan.modalities
        }
        plan.result_limit = min(plan.result_limit, plan.candidate_limit)
        plan.rerank.top_n = min(plan.rerank.top_n, plan.candidate_limit)
        trace = {
            "status": "ok",
            **provider.descriptor,
            "prompt_version": spec.prompt_version,
            "elapsed_seconds": round(elapsed, 6),
            "raw_output": content,
            "plan": plan.model_dump(),
        }
        return plan, trace

    @staticmethod
    def _data_url(path: Path) -> str:
        media_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{media_type};base64,{encoded}"

    def _candidate_frames(
        self, result: dict[str, Any], frame_count: int, max_width: int
    ) -> list[Path]:
        if frame_count <= 0:
            return []
        video = self.catalog.get_video(result["video_id"])
        if not video:
            raise OrchestrationError(f"video does not exist: {result['video_id']}")
        video_path = self.settings.resolve_path(video["file_path"])
        start = max(0.0, float(result["start_time"]))
        end = max(start + 0.05, float(result["end_time"]))
        duration = max(0.0, float(video.get("duration") or end))
        end = min(end, duration) if duration else end
        span = max(0.05, end - start)
        paths = []
        for index in range(frame_count):
            timestamp = start + span * (index + 0.5) / frame_count
            timestamp = min(max(0.0, timestamp), max(0.0, duration - 0.01))
            timestamp_ms = round(timestamp * 1000)
            path = (
                self.settings.frame_cache_dir
                / result["video_id"]
                / f"{timestamp_ms:012d}.jpg"
            )
            if not path.is_file() or path.stat().st_size == 0:
                extract_video_frame(video_path, path, timestamp, max_width=max_width)
            paths.append(path)
        return paths

    @staticmethod
    def _evidence_text(result: dict[str, Any]) -> str:
        details = []
        for item in result.get("evidence") or []:
            detail = item.get("detail") or item.get("text")
            if detail and detail not in details:
                details.append(str(detail))
        return "\n".join(details)[:2000]

    @staticmethod
    def _binary_score(
        response: dict[str, Any], positive_label: str, negative_label: str
    ) -> float:
        choice = response["choices"][0]
        content = str(choice.get("message", {}).get("content") or "").strip()
        entries = (
            choice.get("logprobs", {}).get("content", [{}])[0].get("top_logprobs", [])
        )
        positive = negative = None
        for entry in entries:
            token = str(entry.get("token") or "").strip().casefold()
            logprob = float(entry["logprob"])
            if token == positive_label.strip().casefold():
                positive = logprob if positive is None else max(positive, logprob)
            elif token == negative_label.strip().casefold():
                negative = logprob if negative is None else max(negative, logprob)
        if positive is not None and negative is not None:
            maximum = max(positive, negative)
            yes = math.exp(positive - maximum)
            no = math.exp(negative - maximum)
            return yes / (yes + no)
        if content.casefold().startswith(positive_label.casefold()):
            return 1.0
        if content.casefold().startswith(negative_label.casefold()):
            return 0.0
        raise OrchestrationError(f"reranker returned neither label: {content[:80]}")

    def _score_candidate(
        self,
        result: dict[str, Any],
        query: str,
        plan: RetrievalPlan,
        spec: RerankerSpec,
        provider: OpenAICompatibleProvider,
        prompt: str,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        try:
            frame_paths = self._candidate_frames(
                result,
                plan.rerank.frame_count,
                max_width=max(224, round(math.sqrt(spec.max_visual_pixels * 16 / 9))),
            )
            candidate_context = {
                "query": query,
                "query_intent": plan.query_intent,
                "video_name": result.get("video_name"),
                "start_time": result.get("start_time"),
                "end_time": result.get("end_time"),
                "modalities": result.get("modalities"),
                "retrieval_score": result.get("score"),
                "evidence": self._evidence_text(result),
            }
            content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": _render_prompt(
                        prompt,
                        {
                            "query": query,
                            "candidate": candidate_context,
                        },
                    ),
                }
            ]
            content.extend(
                {
                    "type": "image_url",
                    "image_url": {"url": self._data_url(path)},
                    "max_pixels": spec.max_visual_pixels,
                }
                for path in frame_paths
            )
            payload: dict[str, Any] = {
                "messages": [{"role": "user", "content": content}],
                "temperature": 0,
                "max_tokens": 1,
                "logprobs": True,
                "top_logprobs": 20,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            if (
                spec.positive_token_id is not None
                and spec.negative_token_id is not None
            ):
                payload["allowed_token_ids"] = [
                    spec.positive_token_id,
                    spec.negative_token_id,
                ]
            response, model_elapsed = provider.chat(payload)
            score = self._binary_score(
                response, spec.positive_label, spec.negative_label
            )
            return {
                "status": "ok",
                "rerank_score": round(score, 8),
                "model_elapsed_seconds": round(model_elapsed, 6),
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "frame_count": len(frame_paths),
            }
        except Exception as exc:
            return {
                "status": "error",
                "error": str(exc),
                "elapsed_seconds": round(time.perf_counter() - started, 6),
            }

    def _run_reranker(
        self,
        profile: ProfileSpec,
        query: str,
        plan: RetrievalPlan,
        results: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if profile.reranker is None:
            raise OrchestrationError("selected profile has no reranker")
        spec = profile.reranker
        provider = self._provider(spec.provider)
        prompt = self._prompt(spec.prompt_path)
        top_n = min(plan.rerank.top_n, len(results))
        selected = results[:top_n]
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(spec.concurrency, max(1, top_n))
        ) as executor:
            scores = list(
                executor.map(
                    lambda result: self._score_candidate(
                        result, query, plan, spec, provider, prompt
                    ),
                    selected,
                )
            )
        reranked = []
        trace_candidates = []
        for original_rank, (result, scored) in enumerate(
            zip(selected, scores), start=1
        ):
            item = dict(result)
            original_score = float(item["score"])
            rerank_score = scored.get("rerank_score")
            if rerank_score is None:
                final_score = original_score
            else:
                final_score = (
                    plan.rerank.score_weight * float(rerank_score)
                    + (1 - plan.rerank.score_weight) * original_score
                )
            item["retrieval_score"] = round(original_score, 6)
            item["rerank_score"] = rerank_score
            item["score"] = round(final_score, 6)
            item["original_rank"] = original_rank
            reranked.append(item)
            trace_candidates.append(
                {
                    "video_id": item["video_id"],
                    "start_time": item["start_time"],
                    "end_time": item["end_time"],
                    "original_rank": original_rank,
                    "retrieval_score": round(original_score, 6),
                    **scored,
                }
            )
        reranked.sort(
            key=lambda item: (item["score"], -item["original_rank"]),
            reverse=True,
        )
        final = reranked + results[top_n:]
        trace = {
            "status": "ok",
            **provider.descriptor,
            "prompt_version": spec.prompt_version,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "top_n": top_n,
            "concurrency": spec.concurrency,
            "frame_count": plan.rerank.frame_count,
            "score_weight": plan.rerank.score_weight,
            "candidates": trace_candidates,
        }
        return final, trace

    def _write_trace(self, trace: dict[str, Any]) -> None:
        if not self.settings.orchestration_trace_enabled:
            return
        path = self.settings.resolve_path(self.settings.orchestration_trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(trace, ensure_ascii=False, separators=(",", ":"))
        with self._trace_lock:
            with path.open("a", encoding="utf-8") as target:
                target.write(encoded + "\n")

    def search(
        self,
        text: str | None,
        image_path: str | None,
        modalities: list[str],
        video_ids: list[str] | None,
        alpha: float,
        limit: int,
        *,
        profile_name: str | None = None,
        planner_mode: Literal["auto", "off", "force"] = "auto",
        reranker_mode: Literal["auto", "off", "force"] = "auto",
    ) -> dict[str, Any]:
        request_id = uuid.uuid4().hex
        started = time.perf_counter()
        trace: dict[str, Any] = {
            "schema_version": 1,
            "request_id": request_id,
            "created_at": datetime.now(UTC).isoformat(),
            "query": text,
            "requested_modalities": modalities,
            "planner": {"status": "skipped"},
            "reranker": {"status": "skipped"},
        }
        plan = self.fallback_plan(modalities, alpha, limit)
        profile = None
        orchestration_active = self.settings.orchestration_enabled and (
            planner_mode != "off" or reranker_mode != "off"
        )
        if orchestration_active:
            try:
                resolved_name, profile = self._profile(profile_name)
                trace["profile"] = resolved_name
            except Exception as exc:
                if not self.settings.orchestration_fail_open:
                    raise
                trace["profile_error"] = str(exc)

        if profile is not None and planner_mode != "off" and text:
            try:
                plan, trace["planner"] = self._run_planner(
                    profile,
                    text,
                    modalities,
                    self._available_modalities(video_ids),
                    alpha,
                    limit,
                    bool(image_path),
                )
            except Exception as exc:
                trace["planner"] = {"status": "error", "error": str(exc)}
                if planner_mode == "force" or not self.settings.orchestration_fail_open:
                    raise

        retrieval_started = time.perf_counter()
        results = self.search_engine.search(
            text,
            image_path,
            plan.modalities,
            video_ids,
            plan.alpha,
            plan.candidate_limit,
            plan.merge_gap,
            plan.max_result_seconds,
            plan.visual_profile,
            plan.channel_limits,
        )
        trace["retrieval"] = {
            "status": "ok",
            "elapsed_seconds": round(time.perf_counter() - retrieval_started, 6),
            "result_count": len(results),
            "parameters": {
                "modalities": plan.modalities,
                "alpha": plan.alpha,
                "candidate_limit": plan.candidate_limit,
                "result_limit": plan.result_limit,
                "merge_gap": plan.merge_gap,
                "max_result_seconds": plan.max_result_seconds,
                "visual_profile": plan.visual_profile,
                "channel_limits": plan.channel_limits,
            },
        }

        wants_rerank = (
            reranker_mode == "force"
            or (reranker_mode == "auto" and plan.rerank.enabled)
        )
        if profile is not None and profile.reranker and wants_rerank and text and results:
            if reranker_mode == "force" and not plan.rerank.enabled:
                plan.rerank.enabled = True
                plan.rerank.top_n = min(
                    profile.reranker.default_top_n, plan.candidate_limit
                )
                plan.rerank.frame_count = profile.reranker.default_frame_count
            try:
                results, trace["reranker"] = self._run_reranker(
                    profile, text, plan, results
                )
            except Exception as exc:
                trace["reranker"] = {"status": "error", "error": str(exc)}
                if reranker_mode == "force" or not self.settings.orchestration_fail_open:
                    raise

        results = results[: plan.result_limit]
        trace["plan"] = plan.model_dump()
        trace["final_result_count"] = len(results)
        trace["elapsed_seconds"] = round(time.perf_counter() - started, 6)
        try:
            self._write_trace(trace)
        except OSError as exc:
            # Trace collection is observability, not part of retrieval
            # correctness.  A read-only/full runtime must not lose search.
            trace["trace_write_error"] = str(exc)
        return {"results": results, "execution": trace}
