import React, { useMemo, useState } from "react";
import {
  api, CandidatePlan, EvidenceRole, PlanSetResponse, PlannerExecution, PlanStep,
  PlannerLabCapabilities, PlannerMode, SearchResult, Video,
} from "./api";

function clock(seconds: number) {
  const value = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const rest = value % 60;
  return `${hours ? `${String(hours).padStart(2, "0")}:` : ""}${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

const examples = [
  "厨师在厨房做菜并与其他人交谈",
  "演讲者展示产品后，观众开始鼓掌",
  "画面出现价格或菜单，同时有人介绍菜品",
];

const modeMeta: Record<PlannerMode, { name: string; icon: string; description: string }> = {
  guide: { name: "引导", icon: "◎", description: "计划只读，逐步解释和确认" },
  assist: { name: "协作", icon: "✦", description: "自由编辑，可单步或连续运行" },
  auto: { name: "自动", icon: "↗", description: "计划只读，一键运行并早停" },
};

const planMeta: Record<string, { icon: string; eyebrow: string; accent: string }> = {
  fast: { icon: "↯", eyebrow: "低延迟探索", accent: "amber" },
  balanced: { icon: "◈", eyebrow: "推荐策略", accent: "violet" },
  deep: { icon: "≋", eyebrow: "高质量深挖", accent: "cyan" },
};

const toolGlyph: Record<string, string> = {
  "visual.search": "◉", "face.search": "◎", "asr.search": "≋",
  "ocr.search": "▤", "confidence.filter": "⌁", "vlm.rerank": "✦",
};

const roleMeta: Record<EvidenceRole, { label: string; hint: string }> = {
  primary: { label: "主要证据", hint: "可以创建候选，决定基础排名" },
  support: { label: "补充证据", hint: "只能增强已有候选，不能独立召回" },
  constraint: { label: "必要约束", hint: "检查条件，过严时自动回退" },
  verifier: { label: "模型验证", hint: "只重排已有候选，失败时保留原排序" },
  fallback: { label: "备用召回", hint: "主通道质量不佳时才启动" },
};

const decisionMeta: Record<string, string> = {
  accepted: "已接受", skipped: "已跳过", rolled_back: "已回退", downweighted: "已降权",
};

function clone<T>(value: T): T {
  return JSON.parse(JSON.stringify(value)) as T;
}

export function PlannerLabPage({ videos, capability, setNotice }: {
  videos: Video[];
  capability: PlannerLabCapabilities;
  setNotice: (value: string) => void;
}) {
  const ready = videos.filter(video => video.status === "ready" || video.indexed_modalities.length);
  const [query, setQuery] = useState("");
  const [image, setImage] = useState<File>();
  const [mode, setMode] = useState<PlannerMode>("assist");
  const [selectedVideoIds, setSelectedVideoIds] = useState<string[]>([]);
  const [scopeOpen, setScopeOpen] = useState(false);
  const [scopeSearch, setScopeSearch] = useState("");
  const [capabilitiesOpen, setCapabilitiesOpen] = useState(false);
  const [planSet, setPlanSet] = useState<PlanSetResponse>();
  const [originalPlanSet, setOriginalPlanSet] = useState<PlanSetResponse>();
  const [selectedPlanId, setSelectedPlanId] = useState("balanced");
  const [execution, setExecution] = useState<PlannerExecution>();
  const [executionHistory, setExecutionHistory] = useState<PlannerExecution[]>([]);
  const [planning, setPlanning] = useState(false);
  const [running, setRunning] = useState(false);
  const [playing, setPlaying] = useState<SearchResult>();
  const [expandedResult, setExpandedResult] = useState("");

  const selectedPlan = planSet?.plans.find(plan => plan.plan_id === selectedPlanId);
  const nextStep = execution?.executed_steps ?? 0;
  const canEdit = mode === "assist";
  const scope = selectedVideoIds.length ? selectedVideoIds : undefined;
  const filteredVideos = ready.filter(video => video.name.toLowerCase().includes(scopeSearch.toLowerCase()));
  const toolMap = useMemo(
    () => Object.fromEntries(capability.capabilities.map(item => [item.tool_id, item])),
    [capability],
  );

  const generate = async () => {
    if (!query.trim()) return setNotice("先描述你想找到的画面、人物或事件");
    if (!ready.length) return setNotice("还没有已建立索引的视频");
    setPlanning(true);
    setExecution(undefined);
    setExecutionHistory([]);
    try {
      const value = await api.plannerLabPlans({
        queryText: query.trim(), queryImage: image, videoIds: scope, mode,
      });
      setPlanSet(value);
      setOriginalPlanSet(clone(value));
      setSelectedPlanId("balanced");
      window.setTimeout(() => document.getElementById("strategy-section")?.scrollIntoView({ behavior: "smooth", block: "start" }), 80);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "计划生成失败");
    } finally { setPlanning(false); }
  };

  const updatePlan = (planId: string, transform: (plan: CandidatePlan) => CandidatePlan) => {
    setPlanSet(value => value ? {
      ...value, plans: value.plans.map(plan => plan.plan_id === planId ? transform(plan) : plan),
    } : value);
    setExecution(undefined);
    setExecutionHistory([]);
  };

  const updateStep = (stepId: string, update: Partial<PlanStep>) => {
    if (!selectedPlan) return;
    updatePlan(selectedPlan.plan_id, plan => ({
      ...plan, steps: plan.steps.map(step => step.step_id === stepId ? { ...step, ...update } : step),
    }));
  };

  const moveStep = (index: number, direction: -1 | 1) => {
    if (!selectedPlan) return;
    const target = index + direction;
    if (target < 0 || target >= selectedPlan.steps.length) return;
    const steps = [...selectedPlan.steps];
    [steps[index], steps[target]] = [steps[target], steps[index]];
    updatePlan(selectedPlan.plan_id, plan => ({ ...plan, steps }));
  };

  const resetSelectedPlan = () => {
    const original = originalPlanSet?.plans.find(plan => plan.plan_id === selectedPlanId);
    if (original) updatePlan(selectedPlanId, () => clone(original));
  };

  const runPlan = async (plan: CandidatePlan, maxSteps?: number, preserveHistory = true) => {
    setRunning(true);
    try {
      const value = await api.plannerLabExecute({
        queryText: query.trim(), queryImage: image, videoIds: scope,
        plan, maxSteps,
      });
      if (preserveHistory && execution) setExecutionHistory(value => [...value, execution]);
      setExecution(value);
      window.setTimeout(() => document.getElementById("planner-output")?.scrollIntoView({ behavior: "smooth", block: "start" }), 80);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "计划执行失败");
    } finally { setRunning(false); }
  };

  const execute = async (runAll = false) => {
    if (!selectedPlan) return;
    const maxSteps = mode === "auto" || runAll ? undefined : Math.min(nextStep + 1, selectedPlan.steps.length);
    await runPlan(selectedPlan, maxSteps);
  };

  const reviseLastStep = async (action: "skip" | "downweight") => {
    if (!selectedPlan || !execution?.trace.length) return;
    const last = execution.trace[execution.trace.length - 1];
    const stepId = last.step.step_id as string;
    const changed = {
      ...selectedPlan,
      steps: selectedPlan.steps.map(step => step.step_id !== stepId ? step : action === "skip"
        ? { ...step, enabled: false }
        : { ...step, weight: Math.max(0, step.weight * 0.5) }),
    };
    setPlanSet(value => value ? {
      ...value, plans: value.plans.map(plan => plan.plan_id === changed.plan_id ? changed : plan),
    } : value);
    await runPlan(changed, execution.executed_steps);
  };

  const rollbackExecution = () => {
    const previous = executionHistory[executionHistory.length - 1];
    if (!previous) return;
    setExecution(previous);
    setExecutionHistory(value => value.slice(0, -1));
  };

  const toggleVideo = (id: string) => setSelectedVideoIds(value =>
    value.includes(id) ? value.filter(item => item !== id) : [...value, id]
  );

  return <div className="planner-lab-v2">
    <section className="lab-hero">
      <div className="lab-hero-orb orb-one" /><div className="lab-hero-orb orb-two" />
      <div className="lab-hero-copy">
        <span className="lab-kicker"><i /> MOMENTSEEK PLANNER LAB</span>
        <h2>把一个模糊想法，变成<br /><em>可执行的检索策略</em></h2>
        <p>Qwen3.5 负责思考，MomentSeek 负责行动。你可以比较策略、调整每一步，并看到结果为何出现。</p>
        <div className="lab-trust-row">
          <span>✓ 固定工具边界</span><span>✓ 可解释融合</span><span>✓ 全程可回放</span>
        </div>
      </div>
      <div className="lab-runtime-card">
        <div className="runtime-head"><span className="runtime-pulse" /><b>系统就绪</b><em>LIVE</em></div>
        <div className="runtime-model"><span>Q</span><div><b>{capability.llm_enabled ? "Qwen3.5 vLLM" : "Heuristic Planner"}</b><small>{capability.llm_enabled ? "规划与多模态重排在线" : "当前使用备用规划器"}</small></div></div>
        <div className="runtime-stats"><div><strong>{capability.capabilities.length}</strong><small>可用工具</small></div><div><strong>3</strong><small>候选策略</small></div><div><strong>∞</strong><small>可回放</small></div></div>
      </div>
    </section>

    <section className="lab-composer">
      <div className="composer-main">
        <div className="section-title"><span>01</span><div><h3>描述你的检索目标</h3><p>场景、人物、台词、画面文字和事件顺序都可以组合</p></div></div>
        <div className={`query-canvas ${planning ? "is-planning" : ""}`}>
          <textarea value={query} onChange={event => setQuery(event.target.value)} placeholder="例如：找到厨师一边做菜，一边和客人讨论菜品的片段…" />
          <div className="query-canvas-footer">
            <label className={`image-attach ${image ? "attached" : ""}`}><input type="file" accept="image/*" onChange={event => setImage(event.target.files?.[0])} /><span>{image ? "✓" : "+"}</span><div><b>{image ? image.name : "添加参考图"}</b><small>{image ? "点击更换图片" : "人物、物体或视觉风格"}</small></div></label>
            <button type="button" className="scope-trigger" onClick={() => setScopeOpen(value => !value)}><span>▣</span><div><b>{selectedVideoIds.length ? `${selectedVideoIds.length} 个视频` : "全部可检索视频"}</b><small>{ready.length} 个视频已就绪</small></div><em>{scopeOpen ? "⌃" : "⌄"}</em></button>
          </div>
          {scopeOpen && <div className="scope-popover">
            <div className="scope-search"><span>⌕</span><input value={scopeSearch} onChange={event => setScopeSearch(event.target.value)} placeholder="搜索视频名称" /><button type="button" onClick={() => setSelectedVideoIds([])}>选择全部</button></div>
            <div className="scope-list">{filteredVideos.slice(0, 80).map(video => <label key={video.id} className={selectedVideoIds.includes(video.id) ? "selected" : ""}><input type="checkbox" checked={selectedVideoIds.includes(video.id)} onChange={() => toggleVideo(video.id)} /><span className="scope-check">✓</span><div><b>{video.name}</b><small>{clock(video.duration)} · {video.indexed_modalities.join(" / ")}</small></div></label>)}</div>
            <div className="scope-footer"><span>{selectedVideoIds.length ? `已选择 ${selectedVideoIds.length} 个视频` : "未选择时检索全部视频"}</span><button type="button" onClick={() => setScopeOpen(false)}>完成</button></div>
          </div>}
        </div>
        <div className="query-examples"><span>试试这些</span>{examples.map(item => <button type="button" key={item} onClick={() => setQuery(item)}>{item}</button>)}</div>
      </div>

      <aside className="composer-side">
        <div className="section-title compact"><span>02</span><div><h3>选择协作方式</h3><p>你希望对计划掌控到什么程度？</p></div></div>
        <div className="mode-selector">{(Object.keys(modeMeta) as PlannerMode[]).map(item => <button type="button" key={item} className={mode === item ? "selected" : ""} onClick={() => { setMode(item); setExecution(undefined); }}><span>{modeMeta[item].icon}</span><div><b>{modeMeta[item].name}</b><small>{modeMeta[item].description}</small></div><i /></button>)}</div>
        <button type="button" className="capability-toggle" onClick={() => setCapabilitiesOpen(value => !value)}><span>能力注册表</span><b>{capability.capabilities.length} 个工具可用</b><em>{capabilitiesOpen ? "−" : "+"}</em></button>
        {capabilitiesOpen && <div className="capability-drawer">{capability.capabilities.map(tool => <div key={tool.tool_id}><span>{toolGlyph[tool.tool_id] || "◇"}</span><div><b>{tool.label}</b><small>{tool.description}</small></div><em className={`latency-${tool.latency}`}>{tool.latency}</em></div>)}</div>}
        <button type="button" className="generate-button" disabled={planning} onClick={generate}><span>{planning ? "" : "✦"}</span><div><b>{planning ? "Qwen 正在设计策略" : "生成检索策略"}</b><small>{planning ? "理解意图 · 选择工具 · 估算成本" : "获得 Fast / Balanced / Deep 三套方案"}</small></div><em>{planning ? <i className="button-loader" /> : "→"}</em></button>
      </aside>
    </section>

    {planning && <section className="planning-state"><div className="thinking-orbit"><i /><i /><i /><span>Q</span></div><div><b>正在把你的目标拆成可执行步骤</b><p>分析查询意图、检索范围与可用模态，通常需要 30–80 秒</p></div><div className="thinking-steps"><span className="done">理解意图</span><span className="active">组合工具</span><span>生成策略</span></div></section>}

    {planSet && !planning && <section id="strategy-section" className="strategy-section">
      <div className="strategy-heading"><div className="section-title"><span>03</span><div><h3>选择一条检索路线</h3><p>三套方案使用不同的速度、覆盖度与精度取舍</p></div></div><div className="generated-by"><span className={planSet.planner_trace.status === "ok" ? "ok" : "fallback"}>✦</span><div><b>{planSet.planner_trace.status === "ok" ? "Qwen3.5 已生成" : "已使用备用计划"}</b><small>{planSet.query_intent}</small></div></div></div>
      <div className="strategy-cards">{planSet.plans.map(plan => {
        const meta = planMeta[plan.plan_id] || planMeta.balanced;
        const tools = [...new Set(plan.steps.map(step => step.tool_id))];
        const hasRerank = plan.steps.some(step => step.tool_id === "vlm.rerank");
        return <button type="button" key={plan.plan_id} className={`strategy-card ${meta.accent} ${selectedPlanId === plan.plan_id ? "selected" : ""}`} onClick={() => { setSelectedPlanId(plan.plan_id); setExecution(undefined); }}>
          {plan.plan_id === "balanced" && <span className="recommend-badge">RECOMMENDED</span>}
          <div className="strategy-card-head"><span className="strategy-icon">{meta.icon}</span><div><small>{meta.eyebrow}</small><h4>{plan.label}</h4></div><i className="strategy-radio" /></div>
          <p>{plan.description}</p>
          <div className="strategy-metrics"><div><strong>{plan.steps.length}</strong><small>步骤</small></div><div><strong>{tools.length}</strong><small>工具</small></div><div><strong>{hasRerank ? "有" : "无"}</strong><small>VLM 重排</small></div><div><strong>{plan.estimated_cost === "low" ? "低" : plan.estimated_cost === "medium" ? "中" : "高"}</strong><small>成本</small></div></div>
          <div className="strategy-tool-row">{tools.slice(0, 5).map(tool => <span key={tool} title={toolMap[tool]?.label}>{toolGlyph[tool] || "◇"}</span>)}<em>{plan.fusion.toUpperCase()}</em></div>
        </button>;
      })}</div>

      {selectedPlan && <div className={`plan-workbench mode-${mode}`}>
        <div className="workbench-head"><div><span className={`strategy-icon ${planMeta[selectedPlan.plan_id]?.accent}`}>{planMeta[selectedPlan.plan_id]?.icon}</span><div><small>{canEdit ? "协作编辑器" : "计划预览"}</small><h3>{selectedPlan.label}</h3></div></div><div className="workbench-actions">{canEdit && <button type="button" onClick={resetSelectedPlan}>↺ 恢复初始计划</button>}<span>{selectedPlan.steps.length} 个步骤</span></div></div>
        <div className={`mode-explainer ${mode}`}><span>{modeMeta[mode].icon}</span><div><b>{mode === "guide" ? "引导模式：先理解，再决定是否继续" : mode === "assist" ? "协作模式：这份计划现在由你和 Agent 共同编辑" : "自动模式：确认策略即可，Agent 将自行运行和早停"}</b><small>{mode === "guide" ? "参数已锁定。每次只执行一个新步骤，并展示候选与排名变化。" : mode === "assist" ? "所有参数均可调整；可以验证下一步，也可以直接运行剩余计划。" : "参数已锁定。系统会连续执行所有步骤，排名稳定时提前结束。"}</small></div><em>{canEdit ? "EDITABLE" : "READ ONLY"}</em></div>
        <div className="workbench-grid">
          <div className="pipeline-editor">{selectedPlan.steps.map((step, index) => {
            const tool = toolMap[step.tool_id];
            const role = step.role || (step.operation === "filter" ? "constraint" : step.tool_id === "vlm.rerank" ? "verifier" : "primary");
            const complete = execution && index < execution.executed_steps;
            const active = running && index === Math.min(nextStep, selectedPlan.steps.length - 1);
            return <div className={`pipeline-step tool-${tool?.modality || "aggregate"} role-${role} ${step.enabled === false ? "disabled" : ""} ${complete ? "complete" : ""} ${active ? "active" : ""}`} key={step.step_id}>
              <div className="pipeline-rail"><span>{complete ? "✓" : index + 1}</span>{index < selectedPlan.steps.length - 1 && <i />}</div>
              <div className="pipeline-card">
                <div className="pipeline-card-head"><div className="pipeline-tool-icon">{toolGlyph[step.tool_id] || "◇"}</div><div><span>{step.operation.toUpperCase()} · {tool?.latency || "LOW"} LATENCY</span><h4>{tool?.label || step.tool_id}</h4></div>{canEdit && <div className="step-order"><button type="button" title={step.enabled === false ? "启用步骤" : "禁用步骤"} onClick={() => updateStep(step.step_id, { enabled: step.enabled === false })}>{step.enabled === false ? "○" : "●"}</button><button type="button" disabled={index === 0} onClick={() => moveStep(index, -1)}>↑</button><button type="button" disabled={index === selectedPlan.steps.length - 1} onClick={() => moveStep(index, 1)}>↓</button></div>}</div>
                <div className="pipeline-role-row"><span className={`role-badge ${role}`}>{roleMeta[role].label}</span><small>{roleMeta[role].hint}</small>{canEdit && step.operation === "search" && role !== "fallback" && <select value={role} onChange={event => updateStep(step.step_id, { role: event.target.value as EvidenceRole })}><option value="primary">主要证据</option><option value="support">补充证据</option></select>}</div>
                <label className="step-query-label"><span>{canEdit ? "这个工具要找什么？" : "该步骤的检索目标"}</span><input disabled={!canEdit} value={step.query} onChange={event => updateStep(step.step_id, { query: event.target.value })} /></label>
                <div className="step-controls"><label><span>影响权重 <b>{step.weight.toFixed(2)}</b></span><input disabled={!canEdit} type="range" min="0" max="3" step="0.05" value={step.weight} onChange={event => updateStep(step.step_id, { weight: Number(event.target.value) })} /></label><label><span>候选数量</span><input disabled={!canEdit} type="number" min="1" max="300" value={step.top_k} onChange={event => updateStep(step.step_id, { top_k: Math.max(1, Number(event.target.value)) })} /></label></div>
                {role === "support" && <label className="support-cap"><span>补充奖励上限 <b>{Math.round((step.support_bonus_cap ?? 0.4) * 100)}%</b></span><input disabled={!canEdit} type="range" min="0" max="1" step="0.05" value={step.support_bonus_cap ?? 0.4} onChange={event => updateStep(step.step_id, { support_bonus_cap: Number(event.target.value) })} /></label>}
                <p>{step.rationale || tool?.description}</p>
              </div>
            </div>;
          })}</div>
          <aside className="plan-settings">
            <div className="settings-block"><span>全局融合方式 {canEdit ? "" : "· 已锁定"}</span><div className="fusion-tabs">{(["rrf", "combsum", "combmnz"] as CandidatePlan["fusion"][]).map(item => <button disabled={!canEdit} type="button" key={item} className={selectedPlan.fusion === item ? "selected" : ""} onClick={() => updatePlan(selectedPlan.plan_id, value => ({ ...value, fusion: item }))}>{item === "rrf" ? "RRF" : item === "combsum" ? "SUM" : "MNZ"}</button>)}</div><p>{selectedPlan.fusion === "rrf" ? "按各工具排名融合，对分数尺度差异最稳健。" : selectedPlan.fusion === "combsum" ? "归一化分数相加，突出强单路命中。" : "奖励多工具共同命中，强调跨模态共识。"}</p></div>
            <div className="settings-block"><span>稳定后提前停止 {canEdit ? "" : "· 已锁定"}</span><div className="threshold-display"><strong>{Math.round(selectedPlan.early_stop_threshold * 100)}%</strong><div><i style={{ width: `${selectedPlan.early_stop_threshold * 100}%` }} /></div></div><input disabled={!canEdit} type="range" min="0.5" max="1" step="0.01" value={selectedPlan.early_stop_threshold} onChange={event => updatePlan(selectedPlan.plan_id, value => ({ ...value, early_stop_threshold: Number(event.target.value) }))} /><p>Top-K 集合与顺序同时稳定后，跳过剩余步骤。</p></div>
            <div className="plan-cost-summary"><div><span>预计调用</span><b>{selectedPlan.steps.length} 步</b></div><div><span>结果上限</span><b>{selectedPlan.result_limit}</b></div><div><span>VLM 调用</span><b>{selectedPlan.steps.some(step => step.tool_id === "vlm.rerank") ? `Top ${selectedPlan.steps.find(step => step.tool_id === "vlm.rerank")?.top_k}` : "不调用"}</b></div></div>
          </aside>
        </div>
        <div className="run-dock"><div className="run-plan-ident"><span>{planMeta[selectedPlan.plan_id]?.icon}</span><div><b>{selectedPlan.label} · {modeMeta[mode].name}模式</b><small>{mode === "guide" ? `请确认第 ${Math.min(nextStep + 1, selectedPlan.steps.length)} 步；系统会在执行后解释排名变化` : mode === "assist" ? "先单步验证修改是否有效，或直接运行整份协作计划" : "连续运行全部步骤，排名稳定时自动停止"}</small></div></div><div className="run-progress">{selectedPlan.steps.map((_, index) => <i key={index} className={execution && index < execution.executed_steps ? "done" : ""} />)}</div><div className="run-actions">{mode === "assist" && <button type="button" className="run-secondary" disabled={running || nextStep >= selectedPlan.steps.length} onClick={() => execute(false)}>{nextStep ? "验证下一步" : "先验证第一步"}</button>}<button type="button" disabled={running || (mode === "guide" && nextStep >= selectedPlan.steps.length)} onClick={() => execute(mode !== "guide")}>{running ? <><i className="button-loader" /> 正在执行</> : mode === "guide" ? <>确认并执行第 {Math.min(nextStep + 1, selectedPlan.steps.length)} 步 <span>→</span></> : mode === "assist" ? <>运行剩余计划 <span>→</span></> : <>一键自动运行 <span>→</span></>}</button></div></div>
      </div>}
    </section>}

    {execution && <Output execution={execution} toolMap={toolMap} expandedResult={expandedResult} setExpandedResult={setExpandedResult} setPlaying={setPlaying} canRollback={executionHistory.length > 0} onRollback={rollbackExecution} onRevise={reviseLastStep} running={running} />}
    {playing && <div className="clip-modal" onClick={() => setPlaying(undefined)}><div className="clip-dialog" onClick={event => event.stopPropagation()}><div className="clip-dialog-head"><div><span>VIDEO MOMENT</span><b>{playing.video_name}</b><small>{clock(playing.start_time)} – {clock(playing.end_time)}</small></div><button type="button" onClick={() => setPlaying(undefined)}>×</button></div><video src={playing.clip_url || playing.media_url} controls autoPlay /><div className="clip-dialog-foot"><div>{playing.modalities.map(item => <span key={item}>{item}</span>)}</div><b>融合分 {playing.score.toFixed(3)}</b></div></div></div>}
  </div>;
}

function Output({ execution, toolMap, expandedResult, setExpandedResult, setPlaying, canRollback, onRollback, onRevise, running }: {
  execution: PlannerExecution;
  toolMap: Record<string, any>;
  expandedResult: string;
  setExpandedResult: (value: string) => void;
  setPlaying: (value: SearchResult) => void;
  canRollback: boolean;
  onRollback: () => void;
  onRevise: (action: "skip" | "downweight") => void;
  running: boolean;
}) {
  return <section id="planner-output" className="lab-output">
    <div className="output-heading"><div className="section-title"><span>04</span><div><h3>检索完成，看看为什么是这些结果</h3><p>每一个片段都能追溯到具体工具、分数和执行步骤</p></div></div><span className="execution-id">ID · {execution.execution_id.slice(0, 10)}</span></div>
    <div className="output-kpis"><div><span>⌕</span><p><strong>{execution.count}</strong><small>候选片段</small></p></div><div><span>✓</span><p><strong>{execution.above_count}</strong><small>超过阈值</small></p></div><div><span>◷</span><p><strong>{execution.elapsed_seconds.toFixed(1)}s</strong><small>端到端耗时</small></p></div><div><span>↳</span><p><strong>{execution.executed_steps}</strong><small>已执行步骤</small></p></div><div className="stop-kpi"><span>✦</span><p><strong>{execution.stop_reason === "ranking_stable" ? "排名已稳定" : execution.stop_reason === "paused_after_step" ? "等待下一步" : "计划已完成"}</strong><small>停止原因</small></p></div></div>
    <div className="execution-decisions"><div><b>{execution.accepted_steps ?? 0}</b><span>接受</span></div><div><b>{execution.skipped_steps ?? 0}</b><span>跳过</span></div><div><b>{execution.rolled_back_steps ?? 0}</b><span>自动回退</span></div><div className="decision-actions"><button type="button" disabled={running || !execution.trace.length} onClick={() => onRevise("downweight")}>降低最后一步权重</button><button type="button" disabled={running || !execution.trace.length} onClick={() => onRevise("skip")}>忽略最后一步</button><button type="button" disabled={running || !canRollback} onClick={onRollback}>回到上次结果</button></div></div>
    <div className="trace-panel">
      <div className="trace-panel-head"><div><b>执行轨迹</b><small>观察候选集如何随每个工具变化</small></div><span>TOP-K CONVERGENCE</span></div>
      <div className="trace-flow">{execution.trace.map((item, index) => <React.Fragment key={index}><article>
        <div className="trace-icon">{toolGlyph[item.step.tool_id] || "◇"}<i>{index + 1}</i></div>
        <div className="trace-copy"><span>{item.step.operation} · {roleMeta[(item.effective_role || item.step.role || "primary") as EvidenceRole]?.label}</span><b>{toolMap[item.step.tool_id]?.label || item.step.tool_id}</b><small>{item.raw_result_count} 召回 → {item.output_candidate_count} 候选 · {item.elapsed_seconds}s</small><em className={`decision-${item.decision}`}>{decisionMeta[item.decision] || item.decision} · {item.decision_reason}</em></div>
        <div className="trace-scores"><label><span>集合重合度 <b>{Math.round(item.top_k_jaccard * 100)}%</b></span><i><em style={{ width: `${item.top_k_jaccard * 100}%` }} /></i></label><label><span>顺序稳定度 <b>{Math.round(item.rank_stability * 100)}%</b></span><i><em style={{ width: `${item.rank_stability * 100}%` }} /></i></label></div>
      </article>{index < execution.trace.length - 1 && <div className="trace-arrow">→</div>}</React.Fragment>)}</div>
    </div>
    <div className="results-heading"><div><b>高相关片段</b><small>按融合得分排序 · 点击卡片查看证据</small></div><div className="result-legend"><span><i className="visual" />视觉</span><span><i className="asr" />语音</span><span><i className="rerank" />VLM 重排</span></div></div>
    <div className="evidence-results">{execution.results.map((result, index) => {
      const key = `${result.video_id}-${result.start_time}`;
      const sources = Object.entries(result.planner_evidence?.source_contrib || {});
      const expanded = expandedResult === key;
      return <article className={`evidence-card ${expanded ? "expanded" : ""}`} key={key}>
        <button type="button" className="evidence-media" onClick={() => setPlaying(result)}>{result.thumbnail_url ? <img src={result.thumbnail_url} /> : <span>NO PREVIEW</span>}<i className="media-shade" /><span className="result-rank">#{index + 1}</span><span className="result-time">{clock(result.start_time)} – {clock(result.end_time)}</span><span className="play-button">▶</span><b className="score-pill">{Math.round(result.score * 100)}</b></button>
        <div className="evidence-body" onClick={() => setExpandedResult(expanded ? "" : key)}><div className="evidence-title"><div><h4>{result.video_name}</h4><p>{result.planner_evidence?.source_count || sources.length} 个独立证据源</p></div><span>{expanded ? "⌃" : "⌄"}</span></div><div className="source-pills">{sources.map(([name, score]) => <span key={name} className={`source-${name.split(".")[0]}`}><i>{toolGlyph[name] || "◇"}</i>{toolMap[name]?.label || name}<b>{Number(score).toFixed(3)}</b></span>)}</div>{expanded && <div className="evidence-details"><div><span>原始工具分数</span>{Object.entries(result.planner_evidence?.raw_scores || {}).map(([name, score]) => <label key={name}><b>{name}</b><em>{Number(score).toFixed(4)}</em></label>)}</div><div><span>命中证据</span>{result.evidence.slice(0, 3).map((item, evidenceIndex) => <p key={evidenceIndex}>{item.detail || item.text || `${item.modality} evidence`}</p>)}</div></div>}</div>
      </article>;
    })}</div>
  </section>;
}
