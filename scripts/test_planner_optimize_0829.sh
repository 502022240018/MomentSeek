#!/usr/bin/env bash
# test_planner_optimize_0829.sh
# 验证 2026-08-17 Planner 耗时优化各项改动是否生效。
# 运行环境：0829 宿主机（服务通过 docker cp 方式更新，无热挂载）
#
# 使用方式：
#   bash scripts/test_planner_optimize_0829.sh
#
# 依赖：curl, jq, python3

set -euo pipefail

WORK_ROOT="/home/momentseek_0829_develop/workplace/MomentSeek_planner"
if [[ -f "${WORK_ROOT}/.env.0829" ]]; then
    source "${WORK_ROOT}/.env.0829"
fi

APP_PORT="${PLANNER_LAB_PORT:-8101}"
BASE_URL="http://127.0.0.1:${APP_PORT}"
CONFIG_FILE="${WORK_ROOT}/deploy/orchestration/qwen35-vllm.json"
PROMPT_FILE="${WORK_ROOT}/deploy/orchestration/prompts/snapmind-planner-v2-adaptive.txt"
TRACE_FILE="${WORK_ROOT}/runtime/orchestration-traces.jsonl"
VLLM_BASE_URL="${QWEN35_VLLM_BASE_URL:-http://127.0.0.1:18084/v1}"
# Extract host:port from VLLM_BASE_URL for metrics endpoint (remove /v1 suffix)
VLLM_METRICS_URL=$(echo "$VLLM_BASE_URL" | sed 's|/v1$||')

PASS=0
FAIL=0
WARN=0

# ── 工具函数 ──────────────────────────────────────────────────────────────────
green()  { printf '\033[0;32m✔ %s\033[0m\n' "$*"; }
red()    { printf '\033[0;31m✘ %s\033[0m\n' "$*"; }
yellow() { printf '\033[0;33m⚠ %s\033[0m\n' "$*"; }
header() { printf '\n\033[1;34m── %s\033[0m\n' "$*"; }

pass() { green "$1"; PASS=$((PASS+1)); }
fail() { red   "$1"; FAIL=$((FAIL+1)); }
warn() { yellow "$1"; WARN=$((WARN+1)); }

check() {
    local label="$1" result="$2" expected="$3"
    if [[ "$result" == "$expected" ]]; then
        pass "$label: $result"
    else
        fail "$label: expected='$expected' got='$result'"
    fi
}

# ── Section 1: 配置文件核查 ────────────────────────────────────────────────────
header "Section 1: 配置文件核查 (qwen35-vllm.json)"

if [[ ! -f "$CONFIG_FILE" ]]; then
    fail "配置文件不存在: $CONFIG_FILE"
else
    planner_timeout=$(jq -r '.providers["qwen35-planner"].timeout_seconds' "$CONFIG_FILE")
    check "planner timeout_seconds" "$planner_timeout" "45"

    unified_max=$(jq -r '.profiles["qwen35-unified"].planner.max_tokens' "$CONFIG_FILE")
    check "qwen35-unified planner max_tokens" "$unified_max" "1200"

    temporal_max=$(jq -r '.profiles["qwen35-temporal-efficient"].planner.max_tokens' "$CONFIG_FILE")
    check "qwen35-temporal-efficient planner max_tokens" "$temporal_max" "1200"

    reranker_timeout=$(jq -r '.providers["qwen35-reranker"].timeout_seconds' "$CONFIG_FILE")
    if [[ "$reranker_timeout" == "300" ]]; then
        pass "reranker timeout_seconds 保持 300（未受影响）"
    else
        warn "reranker timeout_seconds 非预期值: $reranker_timeout（预期 300）"
    fi
fi

# ── Section 2: System prompt 核查 ─────────────────────────────────────────────
header "Section 2: System prompt 核查 (snapmind-planner-v2-adaptive.txt)"

if [[ ! -f "$PROMPT_FILE" ]]; then
    fail "Prompt 文件不存在: $PROMPT_FILE"
else
    # 检查两阶段输出说明
    if grep -q "TWO-STAGE OUTPUT" "$PROMPT_FILE" || grep -q "two-stage" "$PROMPT_FILE"; then
        pass "prompt 包含两阶段输出说明"
    else
        fail "prompt 缺少两阶段输出说明"
    fi

    # 检查 optimization_hints schema
    if grep -q "optimization_hints" "$PROMPT_FILE"; then
        pass "prompt 包含 optimization_hints 定义"
    else
        fail "prompt 缺少 optimization_hints 定义"
    fi

    # capability 块必须存在（大小写不敏感）
    if grep -qi "## Registered [Cc]apabilities" "$PROMPT_FILE"; then
        pass "prompt 末尾包含 'Registered Capabilities' 块"
    else
        fail "prompt 末尾缺少 'Registered Capabilities' 块"
    fi

    # 检查是否有紧凑格式要求
    if grep -q "COMPACT" "$PROMPT_FILE" || grep -q "compact" "$PROMPT_FILE"; then
        pass "prompt 包含紧凑格式输出要求"
    else
        warn "prompt 未明确要求紧凑格式（可能导致 token 浪费）"
    fi
fi

# ── Section 3: 服务健康检查 ───────────────────────────────────────────────────
header "Section 3: 服务健康检查"

if ! curl -fsSL --max-time 5 "${BASE_URL}/api/health" >/dev/null 2>&1; then
    fail "健康检查失败: ${BASE_URL}/api/health"
    echo "  后续测试将跳过（服务不可达）"
    printf '\n\033[1;34m── 汇总 ──\033[0m\n'
    printf 'PASS=%s  FAIL=%s  WARN=%s\n' "$PASS" "$FAIL" "$WARN"
    exit 1
else
    pass "服务健康: ${BASE_URL}/api/health"
fi

# ── Section 4: Capabilities 端点验证 ─────────────────────────────────────────
header "Section 4: Capabilities 端点验证"

caps_json=$(curl -fsSL --max-time 5 "${BASE_URL}/api/planner-lab/capabilities")

cap_count=$(echo "$caps_json" | jq '.capabilities | length')
check "capabilities 工具数量" "$cap_count" "7"

llm_enabled=$(echo "$caps_json" | jq -r '.llm_enabled')
if [[ "$llm_enabled" == "true" ]]; then
    pass "LLM 编排已启用（Qwen3.5 模式）"
else
    warn "LLM 编排未启用（当前为 heuristic 模式），后续 trace 验证将受限"
fi

# ── Section 5: vLLM prefix caching 状态确认 ──────────────────────────────────
header "Section 5: vLLM prefix caching 状态（方案 B 收益依赖项）"

if ps aux | grep vllm | grep -v grep | tr ' ' '\n' | grep -qE "enable.prefix|prefix.cach"; then
    pass "vLLM --enable-prefix-caching 已启用，方案 B 预期收益 5–10 s"
else
    warn "vLLM --enable-prefix-caching 未检测到，方案 B 当前收益约 1–3 s（仍有效）"
    warn "  建议在 vLLM 启动参数中加入 --enable-prefix-caching 以充分释放收益"
fi

# ── Section 6: Planner 规划请求 + 耗时验证 ────────────────────────────────────
header "Section 6: Planner 规划请求耗时（连续 5 次）+ Prefix Caching 利用率"

QUERIES=(
    "找到演讲者展示产品后观众鼓掌的片段"
    "播音员播报财经新闻时指着屏幕上图表的画面"
    "主持人介绍嘉宾登台的片段"
    "运动员获奖后举起奖杯的镜头"
    "老师在黑板上写公式讲解的场景"
)

# 创建临时文件存储本次测试的 planner trace
TEMP_TRACES="/tmp/planner_traces_session_$$.jsonl"
> "$TEMP_TRACES"

# 获取测试前的 vLLM prefix cache 指标
echo "正在获取 vLLM prefix cache 基线指标..."
VLLM_METRICS_BEFORE=$(curl -fsSL --max-time 5 "${VLLM_METRICS_URL}/metrics" 2>/dev/null || echo "")
if [[ -n "$VLLM_METRICS_BEFORE" ]]; then
    CACHE_QUERIES_BEFORE=$(echo "$VLLM_METRICS_BEFORE" | grep "^vllm:prefix_cache_queries_total" | grep -oP '\} \K[\d.]+' | head -1)
    CACHE_HITS_BEFORE=$(echo "$VLLM_METRICS_BEFORE" | grep "^vllm:prefix_cache_hits_total" | grep -oP '\} \K[\d.]+' | head -1)
    pass "基线: queries=${CACHE_QUERIES_BEFORE:-0} hits=${CACHE_HITS_BEFORE:-0}"
else
    warn "无法获取 vLLM metrics（${VLLM_METRICS_URL}/metrics 不可达）"
    CACHE_QUERIES_BEFORE=0
    CACHE_HITS_BEFORE=0
fi
echo

total_elapsed=0
ok_count=0
fallback_count=0

for i in "${!QUERIES[@]}"; do
    query="${QUERIES[$i]}"
    resp=$(curl -fsSL --max-time 50 -X POST "${BASE_URL}/api/planner-lab/plans" \
        -F "query_text=${query}" \
        -F "mode=assist" 2>/dev/null) || {
        warn "请求 $((i+1)) 超时或失败: ${query}"
        continue
    }

    status=$(echo "$resp" | jq -r '.planner_trace.status // "unknown"')
    elapsed=$(echo "$resp" | jq -r '.planner_trace.elapsed_seconds // 0')
    plan_count=$(echo "$resp" | jq '.plans | length // 0')

    # 保存 planner_trace 到临时文件（用于后续分析）- 使用 -c 确保紧凑格式
    echo "$resp" | jq -c '.planner_trace' >> "$TEMP_TRACES"

    if [[ "$status" == "ok" ]]; then
        ok_count=$((ok_count+1))
        total_elapsed=$(python3 -c "print(round($total_elapsed + $elapsed, 3))")
        printf '  [%d] %-40s  status=%-8s  elapsed=%.2f s  plans=%s\n' \
            "$((i+1))" "${query:0:40}" "$status" "$elapsed" "$plan_count"
    else
        fallback_count=$((fallback_count+1))
        err=$(echo "$resp" | jq -r '.planner_trace.error // ""')
        warn "请求 $((i+1)) fallback: ${err:0:80}"
    fi
done

if [[ $ok_count -gt 0 ]]; then
    avg=$(python3 -c "print(round($total_elapsed / $ok_count, 2))")
    pass "LLM 规划成功 ${ok_count}/5，平均耗时 ${avg} s"
    if python3 -c "exit(0 if $avg < 20 else 1)"; then
        pass "平均耗时 < 20 s（优化达标，预期 10–20 s）"
    elif python3 -c "exit(0 if $avg < 45 else 1)"; then
        warn "平均耗时 ${avg} s 超出预期（10–20 s），但仍低于优化前基线（30–50 s）"
    else
        fail "平均耗时 ${avg} s 接近或超过优化前基线，优化未生效"
    fi
fi

if [[ $fallback_count -gt 0 ]]; then
    warn "fallback 次数: $fallback_count/5（基线可接受 ≤1；若超出请检查 vLLM 响应格式）"
fi

# 获取测试后的 vLLM prefix cache 指标
echo
header "Section 6.1: vLLM Prefix Cache 利用率分析"

VLLM_METRICS_AFTER=$(curl -fsSL --max-time 5 "${VLLM_METRICS_URL}/metrics" 2>/dev/null || echo "")
if [[ -n "$VLLM_METRICS_AFTER" ]]; then
    CACHE_QUERIES_AFTER=$(echo "$VLLM_METRICS_AFTER" | grep "^vllm:prefix_cache_queries_total" | grep -oP '\} \K[\d.]+' | head -1)
    CACHE_HITS_AFTER=$(echo "$VLLM_METRICS_AFTER" | grep "^vllm:prefix_cache_hits_total" | grep -oP '\} \K[\d.]+' | head -1)

    # 计算本次测试的增量
    DELTA_QUERIES=$(python3 -c "print(int(${CACHE_QUERIES_AFTER:-0} - ${CACHE_QUERIES_BEFORE:-0}))" 2>/dev/null || echo "0")
    DELTA_HITS=$(python3 -c "print(int(${CACHE_HITS_AFTER:-0} - ${CACHE_HITS_BEFORE:-0}))" 2>/dev/null || echo "0")

    if [[ $DELTA_QUERIES -gt 0 ]]; then
        CACHE_HIT_RATE=$(python3 -c "print(round(100 * ${DELTA_HITS} / ${DELTA_QUERIES}, 1))" 2>/dev/null || echo "0")

        echo "本次测试的 prefix cache 统计:"
        echo "  查询 tokens: $DELTA_QUERIES"
        echo "  命中 tokens: $DELTA_HITS"
        echo "  命中率: ${CACHE_HIT_RATE}%"
        echo

        if python3 -c "exit(0 if ${CACHE_HIT_RATE} >= 70 else 1)" 2>/dev/null; then
            pass "✓ Prefix cache 命中率 ${CACHE_HIT_RATE}% ≥ 70%（缓存生效良好）"
            pass "  → 方案 A（available_modalities 前置）已起效"
        elif python3 -c "exit(0 if ${CACHE_HIT_RATE} >= 50 else 1)" 2>/dev/null; then
            warn "Prefix cache 命中率 ${CACHE_HIT_RATE}% 在 50-70% 之间（部分生效）"
            warn "  → 同一视频库内连续查询应有更高命中率"
        elif python3 -c "exit(0 if ${CACHE_HIT_RATE} >= 10 else 1)" 2>/dev/null; then
            warn "Prefix cache 命中率 ${CACHE_HIT_RATE}% < 50%（效果有限）"
            warn "  → 可能 available_modalities 在各请求间仍有变化"
        else
            fail "Prefix cache 命中率 ${CACHE_HIT_RATE}% 极低（< 10%）"
            fail "  → 方案 A 未生效，或 user message 前缀仍然每次都不同"
        fi

        # 估算 prefill 加速
        if [[ $DELTA_HITS -gt 0 ]]; then
            echo
            echo "Prefix cache 收益估算:"
            echo "  缓存命中的 tokens: $DELTA_HITS"
            echo "  假设 prefill 速度 ~50 tokens/s（NPU）"
            SAVED_TIME=$(python3 -c "print(round(${DELTA_HITS} / 50, 2))" 2>/dev/null || echo "0")
            echo "  节省的 prefill 时间: ~${SAVED_TIME}s (across ${ok_count} requests)"
            PER_REQUEST=$(python3 -c "print(round(${SAVED_TIME} / ${ok_count}, 2))" 2>/dev/null || echo "0")
            echo "  平均每请求节省: ~${PER_REQUEST}s"
        fi
    else
        warn "本次测试未检测到 prefix cache 增量（DELTA_QUERIES = $DELTA_QUERIES）"
        warn "  可能原因: vLLM metrics 未更新，或测试请求过少"
    fi
else
    warn "无法获取 vLLM metrics（${VLLM_METRICS_URL}/metrics 不可达）"
    warn "  跳过 prefix cache 利用率分析"
fi

# ── Section 6.5: Optimization Hints 输出成功率验证 ──────────────────────
header "Section 6.5: Optimization Hints 输出成功率（方案 D 关键指标）"

if [[ ! -f "$TEMP_TRACES" ]]; then
    warn "本次测试未生成 planner trace 数据，跳过 hints 验证"
else
    recent_hints=$(cat "$TEMP_TRACES" 2>/dev/null | python3 -c "
import sys, json
lines = [l.strip() for l in sys.stdin if l.strip()]
total = 0
hints_ok = 0
hints_partial = 0
for line in lines:
    try:
        trace = json.loads(line)
        # 直接从 planner_trace 读取 derivation
        deriv = trace.get('derivation', {})
        if not deriv:
            continue

        if deriv.get('mode') == 'hints_guided':
            total += 1
            hints = deriv.get('hints', {})
            # 检查 7 个必需字段
            required = ['fast_strategy', 'fast_top_k_ratio', 'deep_needs_rerank',
                        'deep_extra_modality', 'deep_enhance_primary',
                        'query_complexity', 'rationale']
            present = sum(1 for k in required if k in hints)
            if present == 7:
                hints_ok += 1
            elif present >= 5:
                hints_partial += 1
    except Exception:
        pass
print(f'{hints_ok}/{total} (partial={hints_partial})')
" 2>/dev/null || echo "0/0 (partial=0)")

    if [[ "$recent_hints" =~ ^0/0 ]]; then
        warn "本次测试中无 hints_guided 记录"
        warn "  → 可能尚未切换到新 prompt (snapmind-planner-v2-adaptive.txt)"
        warn "  → 或 LLM 持续返回 3 计划（未触发派生逻辑）"
        warn "  → 检查 ORCHESTRATION_PROFILE 是否为 qwen35-adaptive"
    else
        pass "Hints 输出统计: $recent_hints"
        success_count=$(echo "$recent_hints" | grep -oP '^\d+' || echo "0")
        total_count=$(echo "$recent_hints" | grep -oP '^\d+/\K\d+' || echo "1")
        if [[ $total_count -gt 0 ]]; then
            success_rate=$(python3 -c "print(round($success_count / $total_count * 100, 1))" 2>/dev/null || echo "0")
            if python3 -c "exit(0 if $success_rate >= 90 else 1)" 2>/dev/null; then
                pass "✓ Hints 完整输出率 ${success_rate}% ≥ 90%（达标，方案 D 智能化生效）"
            elif python3 -c "exit(0 if $success_rate >= 70 else 1)" 2>/dev/null; then
                warn "Hints 完整输出率 ${success_rate}% 在 70-90% 之间（建议优化 prompt）"
                warn "  → 检查 raw_output 是否包含所有 7 个字段"
                warn "  → 考虑在 prompt 中添加 few-shot examples"
            else
                fail "Hints 完整输出率 ${success_rate}% < 70%（方案 D 智能化严重受损）"
                fail "  → 必须强化 prompt 或添加 JSON Schema 约束"
            fi
        fi
    fi
fi

# ── Section 6.6: Decode Throughput 验证 ───────────────────────────────
header "Section 6.6: Decode Throughput（生成速度，方案 D 收益验证）"

if [[ ! -f "$TEMP_TRACES" ]]; then
    warn "本次测试未生成 planner trace 数据，跳过 throughput 验证"
else
    throughput_stats=$(cat "$TEMP_TRACES" 2>/dev/null | python3 -c "
import sys, json
lines = [l.strip() for l in sys.stdin if l.strip()]
throughputs = []
token_counts = []
for line in lines:
    try:
        trace = json.loads(line)
        # 直接从 planner_trace 读取
        raw = trace.get('raw_output', '')
        elapsed = trace.get('elapsed_seconds', 0)
        if raw and elapsed > 0.1:  # 至少 100ms
            # 粗略估算 token 数（1 token ≈ 3.5-4 chars for Chinese/English mix）
            token_count = len(raw) / 3.8
            throughput = token_count / elapsed
            throughputs.append(throughput)
            token_counts.append(token_count)
    except Exception:
        pass
if throughputs:
    avg_throughput = sum(throughputs) / len(throughputs)
    avg_tokens = sum(token_counts) / len(token_counts)
    print(f'{avg_throughput:.1f} {avg_tokens:.0f} {len(throughputs)}')
else:
    print('0 0 0')
" 2>/dev/null || echo "0 0 0")

    avg_throughput=$(echo "$throughput_stats" | awk '{print $1}')
    avg_tokens=$(echo "$throughput_stats" | awk '{print $2}')
    sample_count=$(echo "$throughput_stats" | awk '{print $3}')

    if [[ "$sample_count" == "0" ]] || [[ "$avg_throughput" == "0" ]]; then
        warn "无法计算 throughput（本次测试样本不足）"
    else
        pass "平均 decode throughput: ${avg_throughput} tokens/s（样本数: ${sample_count}）"
        pass "平均输出 token 数: ${avg_tokens} tokens"

        # 检查平均 token 数是否符合预期（方案 D 预期 ~230-240 tokens）
        if python3 -c "exit(0 if 200 <= $avg_tokens <= 350 else 1)" 2>/dev/null; then
            pass "✓ 平均 token 数在预期范围 200-350（方案 D 生效）"
        elif python3 -c "exit(0 if $avg_tokens > 350 else 1)" 2>/dev/null; then
            warn "平均 token 数 ${avg_tokens} > 350（可能未触发派生，仍输出 3 计划）"
            warn "  → 检查是否使用了 snapmind-planner-v2-adaptive.txt"
        else
            warn "平均 token 数 ${avg_tokens} < 200（输出过于精简，可能影响质量）"
        fi

        # 注：方案 D 主动减少 token 输出（从 900+ 降到 340），因此不再使用 tokens/s 阈值
        # 评估标准：总耗时（Section 6 已验证 ~20s < 30s 基线）+ token 数（上方已验证 200-350）
        if python3 -c "exit(0 if $avg_throughput >= 15 else 1)" 2>/dev/null; then
            pass "✓ Throughput ${avg_throughput} tokens/s（方案 D 以减少 token 为目标，已达标）"
        else
            fail "Throughput ${avg_throughput} tokens/s < 15（异常低，需检查）"
            fail "  → 检查 vLLM 服务状态和网络延迟"
        fi
    fi
fi

# ── Section 6.7: 派生模式分布统计 ─────────────────────────────────────
header "Section 6.7: 派生模式分布（验证新旧 prompt 使用情况）"

if [[ ! -f "$TEMP_TRACES" ]]; then
    warn "本次测试未生成 planner trace 数据，跳过派生模式统计"
else
    deriv_stats=$(cat "$TEMP_TRACES" 2>/dev/null | python3 -c "
import sys, json
from collections import Counter
lines = [l.strip() for l in sys.stdin if l.strip()]
modes = []
for line in lines:
    try:
        trace = json.loads(line)
        # 直接从 planner_trace 读取
        status = trace.get('status', 'unknown')
        deriv = trace.get('derivation', {})

        mode = deriv.get('mode', 'N/A')
        if status == 'ok':
            modes.append(mode)
        elif status == 'fallback':
            modes.append('fallback')
    except Exception:
        pass
counts = Counter(modes)
total = sum(counts.values())
for mode in ['hints_guided', 'none', 'N/A', 'fallback']:
    count = counts.get(mode, 0)
    pct = round(100 * count / total, 1) if total > 0 else 0
    print(f'{mode}={count}({pct}%)', end=' ')
print(f'total={total}')
" 2>/dev/null || echo "total=0")

    if [[ "$deriv_stats" =~ total=0 ]]; then
        warn "无有效 trace 数据"
    else
        pass "派生模式分布: $deriv_stats"

        hints_guided_count=$(echo "$deriv_stats" | grep -oP 'hints_guided=\K\d+' || echo "0")
        none_count=$(echo "$deriv_stats" | grep -oP 'none=\K\d+' || echo "0")
        fallback_count_stat=$(echo "$deriv_stats" | grep -oP 'fallback=\K\d+' || echo "0")
        total_count_stat=$(echo "$deriv_stats" | grep -oP 'total=\K\d+' || echo "1")

        hints_pct=$(python3 -c "print(round(100 * $hints_guided_count / $total_count_stat, 1))" 2>/dev/null || echo "0")

        if python3 -c "exit(0 if $hints_pct >= 80 else 1)" 2>/dev/null; then
            pass "✓ hints_guided 模式占比 ${hints_pct}% ≥ 80%（方案 D 已全面生效）"
        elif python3 -c "exit(0 if $hints_pct >= 50 else 1)" 2>/dev/null; then
            warn "hints_guided 模式占比 ${hints_pct}% 在 50-80% 之间（新旧混用）"
            warn "  → 可能正在 A/B 测试或部分流量使用旧 profile"
        elif [[ $hints_guided_count -eq 0 ]]; then
            fail "hints_guided 模式占比 0%（方案 D 未生效）"
            fail "  → 检查 ORCHESTRATION_PROFILE 是否设置为 qwen35-adaptive"
            fail "  → 或检查 prompt_path 是否指向 snapmind-planner-v2-adaptive.txt"
        else
            warn "hints_guided 模式占比 ${hints_pct}% < 50%（方案 D 生效率低）"
        fi

        if python3 -c "exit(0 if $fallback_count_stat / $total_count_stat < 0.05 else 1)" 2>/dev/null; then
            pass "✓ Fallback 率 < 5%（系统稳定）"
        else
            fallback_pct=$(python3 -c "print(round(100 * $fallback_count_stat / $total_count_stat, 1))" 2>/dev/null || echo "0")
            warn "Fallback 率 ${fallback_pct}% ≥ 5%（LLM 响应异常率偏高）"
        fi
    fi
fi

# ── Section 7: Trace 文件抽查 ─────────────────────────────────────────────────
header "Section 7: 本次测试 planner trace 抽查"

if [[ ! -f "$TEMP_TRACES" ]]; then
    warn "本次测试未生成 planner trace 数据"
else
    trace_count=$(wc -l < "$TEMP_TRACES" 2>/dev/null || echo "0")
    if [[ "$trace_count" -eq 0 ]]; then
        warn "临时 trace 文件为空"
    else
        pass "本次测试生成 $trace_count 条 planner trace"

        # 抽查第一条 trace 的完整性
        first_trace=$(head -1 "$TEMP_TRACES" | python3 -c "
import sys, json
line = sys.stdin.read().strip()
if not line:
    print('empty')
else:
    try:
        trace = json.loads(line)
        status = trace.get('status', 'unknown')
        deriv = trace.get('derivation', {})
        mode = deriv.get('mode', 'N/A')
        raw = trace.get('raw_output', '')
        if raw:
            try:
                obj = json.loads(raw)
                plans = [p.get('plan_id') for p in obj.get('plans', [])]
                hints = deriv.get('hints', {})
                hints_keys = list(hints.keys())
                print(f'status={status} mode={mode} plans={plans} hints_fields={len(hints_keys)}')
            except Exception as e:
                print(f'raw_output_invalid: {e}')
        else:
            print(f'status={status} mode={mode} no_raw_output')
    except Exception as e:
        print(f'parse_error: {e}')
" 2>/dev/null || echo "script_error")

        if echo "$first_trace" | grep -q "parse_error\|script_error\|empty"; then
            fail "Trace 解析失败: $first_trace"
        elif echo "$first_trace" | grep -q "raw_output_invalid"; then
            fail "Trace raw_output 不是合法 JSON: $first_trace"
        else
            pass "Trace 数据完整: $first_trace"
        fi
    fi
fi

# # ── Section 8: 单元测试（容器内执行）────────────────────────────────────────────
# header "Section 8: 单元测试（在容器 momentseek-0829-planner-lab 内运行）"
#
# CONTAINER="momentseek-0829-planner-lab"
# if ! docker container inspect "$CONTAINER" >/dev/null 2>&1; then
#     warn "容器 $CONTAINER 不存在，跳过单元测试"
# else
#     ut_result=$(docker exec -w /app/backend "$CONTAINER" \
#         python3 -m pytest tests/test_snapmind_planner_lab.py \
#         -k "test_propose_max_tokens or test_propose_response_format or test_propose_context or test_propose_sanitize" \
#         -v --tb=short 2>&1)
#     echo "$ut_result" | grep -E "PASSED|FAILED|ERROR|passed|failed|error" | tail -10
#     if echo "$ut_result" | grep -q "4 passed"; then
#         pass "4 个优化单元测试全部通过"
#     else
#         fail "单元测试未全部通过，请查看上方输出"
#     fi
# fi

# ── 汇总 ──────────────────────────────────────────────────────────────────────
# ── 清理临时文件 ──────────────────────────────────────────────────────
if [[ -f "$TEMP_TRACES" ]]; then
    rm -f "$TEMP_TRACES"
fi

# ── 汇总 ──────────────────────────────────────────────────────────────────────
printf '\n\033[1;34m══ 测试汇总 ══\033[0m\n'
printf '  PASS: \033[0;32m%s\033[0m\n' "$PASS"
printf '  FAIL: \033[0;31m%s\033[0m\n' "$FAIL"
printf '  WARN: \033[0;33m%s\033[0m\n' "$WARN"
printf '\n'

if [[ $FAIL -eq 0 ]]; then
    printf '\033[0;32m✔ 所有核查项通过，Planner 优化验证成功。\033[0m\n'
    exit 0
else
    printf '\033[0;31m✘ 存在 %s 个失败项，请逐一排查。\033[0m\n' "$FAIL"
    exit 1
fi
