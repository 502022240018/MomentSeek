from __future__ import annotations

import json
import math
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.orchestration.retrieval_orchestration import (
    ALLOWED_MODALITIES,
    OrchestrationError,
    RerankPlan,
    RetrievalPlan,
    SearchOrchestrator,
    _extract_json_object,
)


PlanMode = Literal["guide", "assist", "auto"]
FusionMethod = Literal["rrf", "combsum", "combmnz"]
Operation = Literal["search", "rerank", "filter"]
EvidenceRole = Literal["primary", "support", "constraint", "verifier", "fallback"]
FailurePolicy = Literal["skip", "rollback", "fallback", "abort"]


class StepQualityGate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    min_results: int = Field(default=1, ge=0, le=300)
    min_score_spread: float = Field(default=0.01, ge=0, le=1)
    min_match_rate: float = Field(default=0.0, ge=0, le=1)
    min_survivors: int = Field(default=1, ge=0, le=100)
    max_top_k_disruption: float = Field(default=1.0, ge=0, le=1)


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="ignore")

    step_id: str
    tool_id: str
    operation: Operation
    role: EvidenceRole | None = None
    target_id: str = "main"
    depends_on: list[str] = Field(default_factory=list)
    query: str = ""
    weight: float = Field(default=1.0, ge=0, le=3)
    top_k: int = Field(default=50, ge=1, le=300)
    support_bonus_cap: float = Field(default=0.4, ge=0, le=1)
    fallback_for: str | None = None
    failure_policy: FailurePolicy = "skip"
    quality_gate: StepQualityGate = Field(default_factory=StepQualityGate)
    enabled: bool = True
    rationale: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def infer_role(self):
        if self.role is None:
            if self.tool_id == "vlm.rerank":
                self.role = "verifier"
            elif self.operation == "filter":
                self.role = "constraint"
            else:
                self.role = "primary"
        self.target_id = self.target_id.strip() or "main"
        self.depends_on = list(dict.fromkeys(item.strip() for item in self.depends_on if item.strip()))
        return self


class CandidatePlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    plan_id: str
    label: str
    description: str
    estimated_cost: Literal["low", "medium", "high"]
    fusion: FusionMethod = "rrf"
    result_limit: int = Field(default=24, ge=1, le=100)
    early_stop_threshold: float = Field(default=0.9, ge=0, le=1)
    steps: list[PlanStep] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def validate_steps(self):
        ids = [step.step_id for step in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError("plan step_id values must be unique")
        known = set(ids)
        for step in self.steps:
            if any(item not in known for item in step.depends_on):
                raise ValueError(f"step {step.step_id} depends on an unknown step")
            if step.fallback_for is not None and step.fallback_for not in known:
                raise ValueError(f"step {step.step_id} fallback_for is unknown")
        return self


class PlanSet(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query_intent: str
    constraints: list[str] = Field(default_factory=list)
    negative_constraints: list[str] = Field(default_factory=list)
    plans: list[CandidatePlan] = Field(min_length=3, max_length=3)

    @model_validator(mode="after")
    def validate_plan_shapes(self):
        if {plan.plan_id for plan in self.plans} != {"fast", "balanced", "deep"}:
            raise ValueError("plans must contain fast, balanced, and deep")
        return self


class PlannerLabScope(BaseModel):
    video_ids: list[str] | None = None
    folder_ids: list[str] | None = None

    @field_validator("video_ids", "folder_ids")
    @classmethod
    def normalize_ids(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        normalized = list(dict.fromkeys(item.strip() for item in value if item.strip()))
        return normalized or None


@dataclass(frozen=True)
class Capability:
    tool_id: str
    label: str
    modality: str
    operations: tuple[Operation, ...]
    score_range: tuple[float, float]
    calibration: str
    default_top_k: int
    default_weight: float
    cost: Literal["low", "medium", "high"]
    latency: Literal["low", "medium", "high"]
    temporal_granularity: str
    description: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool_id": self.tool_id,
            "label": self.label,
            "modality": self.modality,
            "operations": list(self.operations),
            "score_range": list(self.score_range),
            "calibration": self.calibration,
            "default_top_k": self.default_top_k,
            "default_weight": self.default_weight,
            "cost": self.cost,
            "latency": self.latency,
            "temporal_granularity": self.temporal_granularity,
            "description": self.description,
        }


CAPABILITIES: tuple[Capability, ...] = (
    Capability(
        "visual.search", "视觉语义检索", "visual", ("search", "rerank"),
        (-1.0, 1.0), "per_step_minmax", 100, 1.0, "low", "medium",
        "frame_or_segment", "搜索场景、动作、物体、外观和空间关系。",
    ),
    Capability(
        "face.search", "人物检索", "face", ("search",),
        (0.0, 1.0), "per_step_minmax", 80, 1.0, "low", "medium",
        "face_track", "使用参考图或人物库中的身份搜索人物。",
    ),
    Capability(
        "asr.search", "ASR 词法与语义检索", "asr", ("search", "rerank"),
        (0.0, 1.0), "per_step_minmax", 100, 0.9, "low", "low",
        "utterance", "搜索台词、讲话内容、人名、数字和主题。",
    ),
    Capability(
        "ocr.search", "OCR 词法与语义检索", "ocr", ("search", "rerank"),
        (0.0, 1.0), "per_step_minmax", 80, 0.75, "low", "low",
        "screen_text_segment", "搜索字幕、招牌、Logo、幻灯片和画面文字。",
    ),
    Capability(
        "confidence.filter", "置信度与共识过滤", "aggregate", ("filter",),
        (0.0, 1.0), "none", 50, 1.0, "low", "low",
        "moment", "按融合分数和独立证据来源数量过滤候选。",
    ),
    Capability(
        "vlm.rerank", "Qwen3.5 多模态重排", "aggregate", ("rerank",),
        (0.0, 1.0), "binary_probability", 20, 1.0, "high", "high",
        "candidate_window", "读取候选片段的多帧与检索证据，判断其是否真正匹配查询。",
    ),
)
CAPABILITY_BY_ID = {item.tool_id: item for item in CAPABILITIES}


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    folded = text.casefold()
    return any(term.casefold() in folded for term in terms)


def _make_step(
    step_id: str,
    tool_id: str,
    query: str,
    weight: float,
    top_k: int,
    rationale: str,
    operation: Operation = "search",
    role: EvidenceRole | None = None,
    depends_on: list[str] | None = None,
    fallback_for: str | None = None,
    **parameters: Any,
) -> PlanStep:
    return PlanStep(
        step_id=step_id,
        tool_id=tool_id,
        operation=operation,
        role=role,
        depends_on=depends_on or [],
        fallback_for=fallback_for,
        query=query,
        weight=weight,
        top_k=top_k,
        rationale=rationale,
        parameters=parameters,
    )


class HeuristicPlanGenerator:
    """Deterministic fail-open plan generator for an unavailable or invalid LLM."""

    OCR_TERMS = ("画面文字", "屏幕", "字幕", "招牌", "海报", "logo", "标题", "写着", "ocr")
    ASR_TERMS = ("说", "提到", "谈到", "讲话", "台词", "语音", "听到", "讨论", "asr")
    TEMPORAL_TERMS = ("先", "随后", "然后", "之后", "之前", "同时", "直到", "before", "after", "then")
    FACE_TERMS = ("这个人", "同一个人", "人物", "人脸", "face")

    def generate(
        self,
        query: str,
        available_modalities: list[str],
        has_query_image: bool,
    ) -> PlanSet:
        available = set(available_modalities)
        wants_ocr = _contains_any(query, self.OCR_TERMS)
        wants_asr = _contains_any(query, self.ASR_TERMS)
        wants_face = has_query_image and _contains_any(query, self.FACE_TERMS)
        temporal = _contains_any(query, self.TEMPORAL_TERMS)
        preferred = "face" if wants_face else "ocr" if wants_ocr else "asr" if wants_asr else "visual"
        primary_modality = preferred if preferred in available else next(iter(sorted(available)), "visual")
        primary = f"{primary_modality}.search"
        searchable = [name for name in (primary_modality, "visual", "asr", "ocr") if name in available]
        searchable = list(dict.fromkeys(searchable))

        fast = CandidatePlan(
            plan_id="fast", label="Fast", estimated_cost="low", fusion="rrf",
            description=f"只执行{CAPABILITY_BY_ID[primary].label}，快速获得可浏览结果。",
            result_limit=24, early_stop_threshold=0.96,
            steps=[_make_step("s1", primary, query, 1.0, 80, "选择最直接的召回通道。")],
        )
        balanced = CandidatePlan(
            plan_id="balanced", label="Balanced", estimated_cost="medium", fusion="rrf",
            description="使用互补模态逐步召回并以 RRF 融合。",
            result_limit=24, early_stop_threshold=0.9,
            steps=[
                _make_step(
                    f"s{index + 1}", f"{modality}.search", query,
                    {"visual": 1.0, "face": 1.0, "asr": 0.85, "ocr": 0.7}[modality],
                    {"visual": 100, "face": 80, "asr": 90, "ocr": 70}[modality],
                    "补充独立模态证据并更新融合排名。",
                    role="primary" if index == 0 else "support",
                    depends_on=[] if index == 0 else ["s1"],
                )
                for index, modality in enumerate(searchable[:3])
            ],
        )
        deep_steps = [
            _make_step(
                f"s{index + 1}", f"{modality}.search", query,
                {"visual": 1.1, "face": 1.1, "asr": 0.9, "ocr": 0.75}[modality],
                {"visual": 140, "face": 100, "asr": 120, "ocr": 100}[modality],
                "扩大召回并积累可审计的跨模态证据。",
                role="primary" if index == 0 else "support",
                depends_on=[] if index == 0 else ["s1"],
            )
            for index, modality in enumerate(searchable)
        ]
        if temporal and "visual" in available:
            deep_steps.append(
                _make_step(
                    f"s{len(deep_steps) + 1}", "visual.search", query, 1.2, 60,
                    "时序查询在当前候选范围内再次强化视觉排序。", "rerank",
                    role="support", depends_on=["s1"],
                    restrict_to_current=True,
                )
            )
        else:
            deep_steps.append(
                _make_step(
                    f"s{len(deep_steps) + 1}", "confidence.filter", query, 1.0, 50,
                    "过滤极低置信度候选，同时保留强单路命中。", "filter",
                    role="constraint", depends_on=["s1"],
                    min_score=0.01, min_sources=1,
                )
            )
        deep = CandidatePlan(
            plan_id="deep", label="Deep", estimated_cost="high", fusion="combmnz",
            description="扩大召回并增加重排或过滤，强调多路共同命中。",
            result_limit=30, early_stop_threshold=0.94, steps=deep_steps[:6],
        )
        return PlanSet(
            query_intent="temporal" if temporal else f"{primary_modality}_retrieval",
            constraints=["保持视频时间证据", "所有工具必须来自能力注册表"],
            negative_constraints=[],
            plans=[fast, balanced, deep],
        )


@dataclass
class MomentNode:
    video_id: str
    video_name: str
    start_time: float
    end_time: float
    representative: dict[str, Any]
    primary_contributions: dict[str, float] = field(default_factory=dict)
    support_contributions: dict[str, float] = field(default_factory=dict)
    verifier_contributions: dict[str, float] = field(default_factory=dict)
    support_caps: dict[str, float] = field(default_factory=dict)
    constraint_results: dict[str, bool] = field(default_factory=dict)
    raw_scores: dict[str, float] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    primary_score: float = 0.0
    support_bonus: float = 0.0
    verifier_score: float | None = None
    verifier_blend_weight: float = 0.0
    aggregate_score: float = 0.0

    @property
    def key(self) -> str:
        return f"{self.video_id}:{self.start_time:.3f}-{self.end_time:.3f}"

    @property
    def source_count(self) -> int:
        return len(set(self.primary_contributions) | set(self.support_contributions))

    @property
    def primary_source_count(self) -> int:
        return len(self.primary_contributions)

    @property
    def support_source_count(self) -> int:
        return len(self.support_contributions)

    @property
    def contributions(self) -> dict[str, float]:
        return {
            **self.primary_contributions,
            **self.support_contributions,
            **self.verifier_contributions,
        }


def _minmax(scores: list[float]) -> list[float]:
    if not scores:
        return []
    low, high = min(scores), max(scores)
    if math.isclose(low, high):
        # One isolated hit may be useful; a flat multi-hit distribution carries
        # no ranking information and must not become a row of perfect scores.
        return [1.0] if len(scores) == 1 else [0.0 for _ in scores]
    return [(score - low) / (high - low) for score in scores]


def _top_keys(nodes: list[MomentNode], limit: int = 10) -> list[str]:
    return [node.key for node in sorted(nodes, key=lambda item: item.aggregate_score, reverse=True)[:limit]]


def _jaccard(left: list[str], right: list[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def _rank_stability(left: list[str], right: list[str]) -> float:
    shared = set(left) & set(right)
    if not shared:
        return 0.0
    max_displacement = max(1, max(len(left), len(right)) - 1)
    displacement = sum(abs(left.index(key) - right.index(key)) for key in shared) / len(shared)
    return max(0.0, 1.0 - displacement / max_displacement)


class SnapMindPlannerLab:
    def __init__(self, orchestrator: SearchOrchestrator):
        self.orchestrator = orchestrator
        self.settings = orchestrator.settings
        self.catalog = orchestrator.catalog
        self.search_engine = orchestrator.search_engine
        self.fallback_generator = HeuristicPlanGenerator()
        self.merge_gap_seconds = 3.0

    def capabilities(self) -> dict[str, Any]:
        return {
            "enabled": getattr(self.settings, "planner_lab_enabled", True),
            "llm_enabled": self.settings.orchestration_enabled,
            "planner": "qwen3.5-vllm" if self.settings.orchestration_enabled else "heuristic-v1",
            "capabilities": [item.as_dict() for item in CAPABILITIES],
            "fusion_methods": ["rrf", "combsum", "combmnz"],
            "modes": ["guide", "assist", "auto"],
            "evidence_roles": ["primary", "support", "constraint", "verifier", "fallback"],
            "step_decisions": ["accepted", "skipped", "rolled_back", "downweighted"],
            "role_policy": {
                "primary": "create_candidates",
                "support": "enrich_existing_only",
                "constraint": "filter_with_rollback",
                "verifier": "rerank_with_rollback",
                "fallback": "create_when_primary_fails",
            },
        }

    def _available_modalities(self, video_ids: list[str] | None, has_query_image: bool) -> list[str]:
        values = self.orchestrator._available_modalities(video_ids)
        if not has_query_image:
            # Text can still resolve a registered face entity, so face remains
            # available when the catalog knows the name. The prompt is told not
            # to invent identity evidence.
            return values
        return values

    def _sanitize_plan_set(
        self,
        plan_set: PlanSet,
        available_modalities: list[str],
        has_query_image: bool,
        query: str,
        matched_entity: dict[str, Any] | None = None,
    ) -> PlanSet:
        available = set(available_modalities)
        for plan in plan_set.plans:
            accepted = []
            expensive_steps = 0
            for step in plan.steps:
                capability = CAPABILITY_BY_ID.get(step.tool_id)
                if capability is None or step.operation not in capability.operations:
                    continue
                if step.tool_id == "vlm.rerank" and not self.settings.orchestration_enabled:
                    continue
                if capability.modality in ALLOWED_MODALITIES and capability.modality not in available:
                    continue
                if capability.modality == "face" and not has_query_image:
                    if not matched_entity:
                        continue
                if capability.cost == "high":
                    expensive_steps += 1
                    if expensive_steps > 1:
                        continue
                if step.operation in {"rerank", "filter"} and not any(
                    accepted_step.operation == "search" for accepted_step in accepted
                ):
                    continue
                if step.role == "verifier" and step.tool_id != "vlm.rerank":
                    continue
                if step.role == "constraint" and step.operation != "filter":
                    continue
                if step.role == "fallback" and not step.fallback_for:
                    continue
                step.query = step.query.strip() or query
                step.depends_on = [item for item in step.depends_on if any(
                    accepted_step.step_id == item for accepted_step in accepted
                )]
                accepted.append(step)
            if not accepted:
                fallback = self.fallback_generator.generate(query, available_modalities, has_query_image)
                replacement = next(item for item in fallback.plans if item.plan_id == plan.plan_id)
                accepted = replacement.steps
            if matched_entity and "face" in available and not any(
                step.tool_id == "face.search" for step in accepted
            ):
                entity_name = str(matched_entity.get("name") or query)
                accepted.insert(
                    0,
                    _make_step(
                        "identity-face",
                        "face.search",
                        entity_name,
                        1.0,
                        80,
                        f"实体库已匹配 {entity_name}，确定性加入人物身份证据。",
                        role="primary",
                    ),
                )
            if len(accepted) > 6:
                verifier = next(
                    (step for step in reversed(accepted) if step.role == "verifier"),
                    None,
                )
                accepted = accepted[:6]
                if verifier is not None and verifier not in accepted:
                    accepted[-1] = verifier
            kept_ids = {step.step_id for step in accepted}
            for step in accepted:
                step.depends_on = [item for item in step.depends_on if item in kept_ids]
                if step.fallback_for not in kept_ids:
                    step.fallback_for = None
                    if step.role == "fallback":
                        step.role = "support"
            plan.steps = accepted
        return plan_set

    @staticmethod
    def _public_entity(entity: dict[str, Any] | None) -> dict[str, Any] | None:
        if not entity:
            return None
        return {
            key: entity[key]
            for key in ("id", "name")
            if entity.get(key) is not None
        } or None

    @staticmethod
    def _normalize_llm_payload(payload: dict[str, Any]) -> dict[str, Any]:
        """Repair model-owned identifiers before strict semantic validation.

        step_id is an executor bookkeeping key, not a planning decision. Small
        models often repeat it across a plan even when every tool/parameter is
        otherwise valid, so the deterministic boundary owns canonical IDs.
        """
        for plan in payload.get("plans", []):
            if not isinstance(plan, dict):
                continue
            steps = plan.get("steps")
            if not isinstance(steps, list):
                continue
            id_map: dict[str, str] = {}
            for index, step in enumerate(steps, start=1):
                if isinstance(step, dict):
                    old_id = step.get("step_id")
                    if isinstance(old_id, str) and old_id not in id_map:
                        id_map[old_id] = f"s{index}"
            for index, step in enumerate(steps, start=1):
                if isinstance(step, dict):
                    step["step_id"] = f"s{index}"
                    dependencies = step.get("depends_on")
                    if isinstance(dependencies, list):
                        step["depends_on"] = [
                            id_map.get(item, item) for item in dependencies if isinstance(item, str)
                        ]
                    fallback_for = step.get("fallback_for")
                    if isinstance(fallback_for, str):
                        step["fallback_for"] = id_map.get(fallback_for, fallback_for)
                    parameters = step.get("parameters")
                    if isinstance(parameters, dict):
                        for field_name in ("query", "weight", "top_k"):
                            if field_name in parameters:
                                step[field_name] = parameters.pop(field_name)
        return payload

    def propose(
        self,
        query: str,
        mode: PlanMode,
        video_ids: list[str] | None,
        has_query_image: bool,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        available = self._available_modalities(video_ids, has_query_image)
        if not available:
            raise OrchestrationError("所选范围没有可用的检索索引")
        matched_entity = self.catalog.find_entity_in_text(query) if not has_query_image else None
        public_entity = self._public_entity(matched_entity)
        fallback = self.fallback_generator.generate(query, available, has_query_image)
        if self.settings.orchestration_enabled:
            deep = next(item for item in fallback.plans if item.plan_id == "deep")
            deep.steps = [step for step in deep.steps if step.tool_id != "confidence.filter"]
            deep.steps.append(
                _make_step(
                    f"s{len(deep.steps) + 1}", "vlm.rerank", query, 1.0, 20,
                    "用 Qwen3.5 多帧判断对高分候选做一次昂贵但精确的重排。", "rerank",
                    role="verifier", depends_on=["s1"],
                    frame_count=4, window_seconds=2.0, score_weight=0.8,
                )
            )
        trace: dict[str, Any] = {"status": "fallback", "planner": "heuristic-v1"}
        plan_set = fallback
        if self.settings.orchestration_enabled:
            try:
                _resolved_name, profile = self.orchestrator._profile(profile_name)
                if profile.planner is None:
                    raise OrchestrationError("selected profile has no planner")
                provider = self.orchestrator._provider(profile.planner.provider)
                prompt_path = self.settings.resolve_path(
                    getattr(
                        self.settings,
                        "planner_lab_prompt_path",
                        "deploy/orchestration/prompts/snapmind-planner-v2-role-aware.txt",
                    )
                )
                prompt = prompt_path.read_text(encoding="utf-8")
                context = {
                    "query": query,
                    "mode": mode,
                    "available_modalities": available,
                    "has_query_image": has_query_image,
                    "matched_entity": public_entity,
                    "capability_registry": [
                        item.as_dict() for item in CAPABILITIES
                        if item.tool_id != "vlm.rerank" or self.settings.orchestration_enabled
                    ],
                }
                response, elapsed = provider.chat(
                    {
                        "messages": [
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
                        ],
                        "temperature": 0,
                        "max_tokens": 2200,
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "snapmind_plan_set",
                                "schema": PlanSet.model_json_schema(),
                                "strict": True,
                            },
                        },
                        "chat_template_kwargs": {"enable_thinking": False},
                    }
                )
                content = response["choices"][0]["message"]["content"]
                raw_plan_set = self._normalize_llm_payload(_extract_json_object(content))
                plan_set = PlanSet.model_validate(raw_plan_set)
                plan_set = self._sanitize_plan_set(
                    plan_set,
                    available,
                    has_query_image,
                    query,
                    matched_entity,
                )
                trace = {
                    "status": "ok",
                    **provider.descriptor,
                    "prompt_version": "snapmind-planner-v2-role-aware",
                    "elapsed_seconds": round(elapsed, 6),
                    "raw_output": content,
                }
            except Exception as exc:
                trace = {
                    "status": "fallback",
                    "planner": "heuristic-v1",
                    "error": str(exc),
                }
                if not self.settings.orchestration_fail_open:
                    raise
        plan_set = self._sanitize_plan_set(
            plan_set,
            available,
            has_query_image,
            query,
            matched_entity,
        )
        return {
            "mode": mode,
            "available_modalities": available,
            "matched_entity": public_entity,
            "planner_trace": trace,
            **plan_set.model_dump(),
        }

    @staticmethod
    def _validate_plan(plan: CandidatePlan) -> None:
        reranker_count = 0
        has_retrieval = False
        primary_steps: set[str] = set()
        for step in plan.steps:
            capability = CAPABILITY_BY_ID.get(step.tool_id)
            if capability is None:
                raise OrchestrationError(f"计划引用了未注册工具: {step.tool_id}")
            if step.operation not in capability.operations:
                raise OrchestrationError(f"{step.tool_id} 不支持 {step.operation} 操作")
            if step.operation in {"rerank", "filter"} and not has_retrieval:
                raise OrchestrationError("rerank/filter 必须放在至少一个检索步骤之后")
            if step.tool_id == "vlm.rerank":
                if step.role != "verifier":
                    raise OrchestrationError("vlm.rerank 必须使用 verifier 证据角色")
                reranker_count += 1
                if reranker_count > 1:
                    raise OrchestrationError("每个计划最多执行一次高成本 Qwen3.5 reranker")
            if step.operation == "filter" and step.role != "constraint":
                raise OrchestrationError("filter 步骤必须使用 constraint 证据角色")
            if step.role == "constraint" and step.operation != "filter":
                raise OrchestrationError("constraint 证据角色必须执行 filter")
            if step.role == "fallback":
                if not step.fallback_for or step.fallback_for not in primary_steps:
                    raise OrchestrationError("fallback 必须引用此前的 primary 步骤")
            if step.role in {"support", "constraint", "verifier"} and not has_retrieval:
                raise OrchestrationError(f"{step.role} 必须放在至少一个检索步骤之后")
            if step.role == "primary" and step.operation == "search":
                primary_steps.add(step.step_id)
            has_retrieval = has_retrieval or step.operation == "search"

    def _find_node(self, nodes: list[MomentNode], result: dict[str, Any]) -> MomentNode | None:
        planner_evidence = result.get("planner_evidence")
        moment_id = (
            planner_evidence.get("moment_id")
            if isinstance(planner_evidence, dict)
            else None
        )
        if moment_id:
            exact = next((node for node in nodes if node.key == moment_id), None)
            if exact is not None:
                return exact
        start = float(result.get("original_start_time", result.get("start_time", 0.0)))
        end = max(
            start,
            float(result.get("original_end_time", result.get("end_time", start))),
        )
        center = (start + end) / 2
        best: tuple[float, MomentNode] | None = None
        for node in nodes:
            if node.video_id != result.get("video_id"):
                continue
            node_center = (node.start_time + node.end_time) / 2
            overlaps = start <= node.end_time and end >= node.start_time
            distance = abs(center - node_center)
            if overlaps or distance <= self.merge_gap_seconds:
                if best is None or distance < best[0]:
                    best = (distance, node)
        return best[1] if best else None

    @staticmethod
    def _recompute(nodes: list[MomentNode], fusion: FusionMethod) -> None:
        for node in nodes:
            primary = sum(node.primary_contributions.values())
            if fusion == "combmnz":
                primary *= max(1, node.primary_source_count)
            node.primary_score = primary
            # Candidate admission is represented by the presence of a primary
            # contribution, not by its normalized magnitude. Per-step MinMax
            # legitimately assigns 0 to the weakest admitted candidate; a
            # later verifier must still be able to rescue it.
            if not node.primary_contributions:
                node.support_bonus = 0.0
                node.verifier_score = None
                node.aggregate_score = 0.0
                continue
            cap_ratio = max(node.support_caps.values(), default=0.4)
            support_bonus = min(sum(node.support_contributions.values()), primary * cap_ratio)
            node.support_bonus = support_bonus
            score = primary + support_bonus
            if node.verifier_contributions:
                verifier = max(node.verifier_contributions.values())
                node.verifier_score = verifier
                blend = node.verifier_blend_weight or 0.8
                score = (1.0 - blend) * score + blend * verifier
            else:
                node.verifier_score = None
            node.aggregate_score = score

    def _merge_results(
        self,
        nodes: list[MomentNode],
        results: list[dict[str, Any]],
        step: PlanStep,
        fusion: FusionMethod,
        effective_role: EvidenceRole | None = None,
    ) -> dict[str, Any]:
        role = effective_role or step.role or "primary"
        def score_for_role(item: dict[str, Any]) -> float:
            if role == "verifier" and item.get("rerank_score") is not None:
                try:
                    value = float(item["rerank_score"])
                    if math.isfinite(value):
                        return value
                except (TypeError, ValueError):
                    pass
            return float(item.get("score", 0.0))

        scores = [score_for_role(item) for item in results]
        normalized = _minmax(scores)
        matched_count = 0
        new_count = 0
        unmatched_count = 0
        for rank, (result, norm) in enumerate(zip(results, normalized), start=1):
            node = self._find_node(nodes, result)
            if node is None:
                if role not in {"primary", "fallback"}:
                    unmatched_count += 1
                    continue
                node = MomentNode(
                    video_id=str(result.get("video_id", "")),
                    video_name=str(result.get("video_name", "")),
                    start_time=float(result.get("start_time", 0.0)),
                    end_time=float(result.get("end_time", result.get("start_time", 0.0))),
                    representative=dict(result),
                )
                nodes.append(node)
                new_count += 1
            else:
                matched_count += 1
                node.start_time = min(node.start_time, float(result.get("start_time", node.start_time)))
                node.end_time = max(node.end_time, float(result.get("end_time", node.end_time)))
                if (
                    role != "verifier"
                    and float(result.get("score", 0.0))
                    > float(node.representative.get("score", 0.0))
                ):
                    node.representative = dict(result)
            contribution = (
                step.weight * norm
                if role == "verifier"
                else step.weight / (60.0 + rank)
                if fusion == "rrf"
                else step.weight * norm
            )
            # One modality/tool is one independent source even when a plan
            # invokes it more than once. This prevents CombMNZ rewarding a
            # repeated visual call as if it were cross-modal consensus.
            key = step.tool_id
            if role in {"primary", "fallback"}:
                bucket = node.primary_contributions
            elif role == "support":
                bucket = node.support_contributions
                node.support_caps[key] = step.support_bonus_cap
            elif role == "verifier":
                bucket = node.verifier_contributions
                node.verifier_blend_weight = max(
                    node.verifier_blend_weight,
                    min(1.0, max(0.0, float(step.parameters.get("score_weight", 0.8)))),
                )
            else:
                bucket = node.support_contributions
            bucket[key] = max(bucket.get(key, 0.0), contribution)
            node.raw_scores[key] = max(
                node.raw_scores.get(key, float("-inf")), score_for_role(result)
            )
            existing = {
                (item.get("modality"), item.get("detail"), item.get("best_time"))
                for item in node.evidence
            }
            for evidence in result.get("evidence", []):
                marker = (evidence.get("modality"), evidence.get("detail"), evidence.get("best_time"))
                if marker not in existing:
                    node.evidence.append(dict(evidence))
                    existing.add(marker)
        self._recompute(nodes, fusion)
        spread = max(scores) - min(scores) if len(scores) > 1 else (1.0 if scores else 0.0)
        above_count = sum(1 for item in results if item.get("above_threshold", True))
        return {
            "matched_existing_count": matched_count,
            "new_candidate_count": new_count,
            "unmatched_result_count": unmatched_count,
            "score_spread": spread,
            "above_threshold_ratio": above_count / len(results) if results else 0.0,
        }

    def _search_step(
        self,
        query: str,
        image_path: str | None,
        step: PlanStep,
        video_ids: list[str] | None,
    ) -> list[dict[str, Any]]:
        capability = CAPABILITY_BY_ID[step.tool_id]
        results = self.search_engine.search(
            step.query or query,
            image_path,
            [capability.modality],
            video_ids,
            0.5,
            step.top_k,
            float(step.parameters.get("merge_gap", 2.0)),
            float(step.parameters.get("max_result_seconds", 15.0)),
            str(step.parameters.get("visual_profile", "balanced")),
            {capability.modality: step.top_k},
            list(step.parameters.get("visual_subqueries", [])),
        )
        return [dict(item) for item in results]

    def _restrict_to_current(
        self, nodes: list[MomentNode], results: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [item for item in results if self._find_node(nodes, item) is not None]

    def _rerank_step(
        self, query: str, nodes: list[MomentNode], step: PlanStep
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if not self.settings.orchestration_enabled:
            raise OrchestrationError("Qwen3.5 reranker 未启用")
        _profile_name, profile = self.orchestrator._profile(None)
        if profile.reranker is None:
            raise OrchestrationError("当前编排 Profile 没有 reranker")
        candidates = self._serialize_results(nodes, min(step.top_k, len(nodes)))
        if not candidates:
            return [], {"status": "skipped", "reason": "no_candidates"}
        retrieval_plan = RetrievalPlan(
            query_intent="planner_lab_rerank",
            modalities=["visual"],
            candidate_limit=max(1, len(candidates)),
            result_limit=max(1, len(candidates)),
            rerank=RerankPlan(
                enabled=True,
                top_n=min(step.top_k, len(candidates)),
                frame_count=int(step.parameters.get("frame_count", 4)),
                window_seconds=float(step.parameters.get("window_seconds", 2.0)),
                score_weight=float(step.parameters.get("score_weight", 0.8)),
            ),
        )
        return self.orchestrator._run_reranker(
            profile, query, retrieval_plan, candidates
        )

    @staticmethod
    def _filter(nodes: list[MomentNode], step: PlanStep) -> list[MomentNode]:
        min_score = float(step.parameters.get("min_score", 0.0))
        min_sources = int(step.parameters.get("min_sources", 1))
        kept = []
        for node in nodes:
            passed = node.aggregate_score >= min_score and node.source_count >= min_sources
            node.constraint_results[step.step_id] = passed
            if passed:
                kept.append(node)
        return sorted(kept, key=lambda item: item.aggregate_score, reverse=True)[:step.top_k]

    @staticmethod
    def _serialize_results(nodes: list[MomentNode], limit: int) -> list[dict[str, Any]]:
        ordered = sorted(nodes, key=lambda item: item.aggregate_score, reverse=True)[:limit]
        maximum = max((item.aggregate_score for item in ordered), default=0.0)
        payload = []
        for node in ordered:
            result = dict(node.representative)
            result.update(
                {
                    "start_time": round(node.start_time, 3),
                    "end_time": round(node.end_time, 3),
                    "score": round(node.aggregate_score / maximum, 6) if maximum > 0 else 0.0,
                    "modalities": sorted({
                        item.get("modality", "") for item in node.evidence if item.get("modality")
                    }),
                    "evidence": node.evidence,
                    "above_threshold": bool(node.representative.get("above_threshold", True)),
                    "planner_evidence": {
                        "moment_id": node.key,
                        "raw_scores": {key: round(value, 6) for key, value in node.raw_scores.items()},
                        "source_contrib": {
                            key: round(value, 6) for key, value in node.contributions.items()
                        },
                        "source_count": node.source_count,
                        "primary_source_count": node.primary_source_count,
                        "support_source_count": node.support_source_count,
                        "primary_contrib": {
                            key: round(value, 6) for key, value in node.primary_contributions.items()
                        },
                        "support_contrib": {
                            key: round(value, 6) for key, value in node.support_contributions.items()
                        },
                        "verifier_contrib": {
                            key: round(value, 6) for key, value in node.verifier_contributions.items()
                        },
                        "constraint_results": dict(node.constraint_results),
                        "primary_score": round(node.primary_score, 6),
                        "support_bonus": round(node.support_bonus, 6),
                        "verifier_score": (
                            round(node.verifier_score, 6) if node.verifier_score is not None else None
                        ),
                        "verifier_blend_weight": round(node.verifier_blend_weight, 6),
                    },
                }
            )
            payload.append(result)
        return payload

    @staticmethod
    def _quality_decision(
        step: PlanStep,
        role: EvidenceRole,
        metrics: dict[str, Any],
    ) -> tuple[str, str]:
        gate = step.quality_gate
        raw_count = int(metrics.get("raw_result_count", 0))
        if role == "constraint" and int(metrics.get("output_candidate_count", 0)) < gate.min_survivors:
            return "rolled_back", "constraint_too_restrictive"
        if raw_count < gate.min_results:
            return "skipped", "insufficient_results"
        if (
            role != "constraint"
            and raw_count > 1
            and float(metrics.get("score_spread", 0.0)) < gate.min_score_spread
        ):
            return "rolled_back", "flat_score_distribution"
        if role == "support":
            matched = int(metrics.get("matched_existing_count", 0))
            match_rate = matched / raw_count if raw_count else 0.0
            if matched == 0 or match_rate < gate.min_match_rate:
                return "skipped", "support_did_not_match_primary"
        if (
            int(metrics.get("input_candidate_count", 0)) > 0
            and float(metrics.get("top_k_disruption", 0.0)) > gate.max_top_k_disruption
        ):
            return "rolled_back", "top_k_disruption_exceeded"
        return "accepted", "quality_gate_passed"

    def execute(
        self,
        query: str,
        image_path: str | None,
        plan: CandidatePlan,
        video_ids: list[str] | None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        self._validate_plan(plan)
        execution_id = uuid.uuid4().hex
        started = time.perf_counter()
        nodes: list[MomentNode] = []
        trace: list[dict[str, Any]] = []
        failed_primary_steps: set[str] = set()
        step_statuses: dict[str, str] = {}
        stop_reason = "plan_completed"
        planned_steps = plan.steps[:max_steps or len(plan.steps)]

        for index, step in enumerate(planned_steps):
            step_started = time.perf_counter()
            before = _top_keys(nodes)
            input_count = len(nodes)
            tool_trace: dict[str, Any] | None = None
            effective_role: EvidenceRole = step.role or "primary"
            decision = "accepted"
            decision_reason = "quality_gate_passed"
            merge_metrics: dict[str, Any] = {}
            checkpoint = deepcopy(nodes)

            dependency_failed = any(
                step_statuses.get(item) not in {"accepted", "downweighted"}
                for item in step.depends_on
            )
            fallback_not_needed = (
                effective_role == "fallback"
                and step.fallback_for not in failed_primary_steps
            )
            if not step.enabled:
                decision, decision_reason = "skipped", "disabled_by_user"
                raw_count = 0
            elif (
                dependency_failed
                and effective_role != "fallback"
                and not (effective_role == "support" and not nodes)
            ):
                decision, decision_reason = "skipped", "dependency_not_accepted"
                raw_count = 0
            elif fallback_not_needed:
                decision, decision_reason = "skipped", "fallback_not_needed"
                raw_count = 0
            else:
                if effective_role == "support" and not nodes:
                    effective_role = "fallback"
                    decision_reason = "support_promoted_for_empty_primary_pool"
                try:
                    if step.operation == "filter":
                        nodes = self._filter(nodes, step)
                        raw_count = len(nodes)
                    elif step.tool_id == "vlm.rerank":
                        raw_results, tool_trace = self._rerank_step(query, nodes, step)
                        raw_count = len(raw_results)
                        merge_metrics = self._merge_results(
                            nodes, raw_results, step, plan.fusion, "verifier"
                        )
                        nodes = sorted(nodes, key=lambda item: item.aggregate_score, reverse=True)
                    else:
                        raw_results = self._search_step(query, image_path, step, video_ids)
                        if step.operation == "rerank" and step.parameters.get("restrict_to_current") and nodes:
                            raw_results = self._restrict_to_current(nodes, raw_results)
                        raw_count = len(raw_results)
                        merge_metrics = self._merge_results(
                            nodes, raw_results, step, plan.fusion, effective_role
                        )
                        nodes = sorted(nodes, key=lambda item: item.aggregate_score, reverse=True)
                        candidate_limit = max(step.top_k, plan.result_limit)
                        if effective_role == "support":
                            candidate_limit = max(candidate_limit, input_count)
                        nodes = nodes[:candidate_limit]
                except Exception as exc:
                    if step.failure_policy == "abort":
                        raise
                    nodes = checkpoint
                    raw_count = 0
                    decision = "rolled_back" if step.failure_policy == "rollback" else "skipped"
                    decision_reason = f"tool_error:{type(exc).__name__}"
                    tool_trace = {"status": "error", "error": str(exc)}

            preview_after = _top_keys(nodes)
            preview_jaccard = _jaccard(before, preview_after) if before else 0.0
            preview_stability = _rank_stability(before, preview_after) if before else 0.0
            metrics = {
                "input_candidate_count": input_count,
                "raw_result_count": raw_count,
                "output_candidate_count": len(nodes),
                "top_k_jaccard": preview_jaccard,
                "rank_stability": preview_stability,
                "top_k_disruption": 1.0 - preview_jaccard if before else 0.0,
                **merge_metrics,
            }
            if decision == "accepted":
                decision, quality_reason = self._quality_decision(step, effective_role, metrics)
                if decision_reason == "quality_gate_passed":
                    decision_reason = quality_reason
                elif decision != "accepted":
                    decision_reason = f"{decision_reason};{quality_reason}"
            if decision in {"skipped", "rolled_back"}:
                nodes = checkpoint
                if step.role == "primary":
                    failed_primary_steps.add(step.step_id)

            after = _top_keys(nodes)
            jaccard = _jaccard(before, after) if before else 0.0
            stability = _rank_stability(before, after) if before else 0.0
            step_statuses[step.step_id] = decision
            trace.append(
                {
                    "step_index": index,
                    "step": step.model_dump(),
                    "input_candidate_count": input_count,
                    "raw_result_count": raw_count,
                    "output_candidate_count": len(nodes),
                    "decision": decision,
                    "decision_reason": decision_reason,
                    "effective_role": effective_role,
                    "quality_metrics": {
                        key: round(value, 4) if isinstance(value, float) else value
                        for key, value in metrics.items()
                    },
                    "elapsed_seconds": round(time.perf_counter() - step_started, 3),
                    "top_k_jaccard": round(jaccard, 4),
                    "rank_stability": round(stability, 4),
                    "top_moments": after[:5],
                    "added_to_top": [key for key in after if key not in before],
                    "removed_from_top": [key for key in before if key not in after],
                    "tool_trace": tool_trace,
                }
            )
            if (
                decision == "accepted"
                and index >= 1
                and before
                and jaccard >= plan.early_stop_threshold
                and stability >= plan.early_stop_threshold
            ):
                stop_reason = "ranking_stable"
                break

        if max_steps is not None and max_steps < len(plan.steps) and stop_reason == "plan_completed":
            stop_reason = "paused_after_step"
        results = self._serialize_results(nodes, plan.result_limit)
        outcome = {
            "execution_id": execution_id,
            "plan": plan.model_dump(),
            "executed_steps": len(trace),
            "stop_reason": stop_reason,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "count": len(results),
            "above_count": sum(1 for item in results if item.get("above_threshold")),
            "accepted_steps": sum(1 for item in trace if item["decision"] == "accepted"),
            "skipped_steps": sum(1 for item in trace if item["decision"] == "skipped"),
            "rolled_back_steps": sum(1 for item in trace if item["decision"] == "rolled_back"),
            "results": results,
            "trace": trace,
        }
        self.orchestrator._write_trace(
            {
                "trace_type": "snapmind_planner_lab",
                "query": query,
                "video_ids": video_ids,
                **outcome,
            }
        )
        return outcome
