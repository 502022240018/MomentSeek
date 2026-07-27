import { useEffect, useMemo, useState } from "react";

import {
  api,
  ColorGradingCapability,
  ColorGradingTask,
  Video,
} from "./api";


function statusText(status: string) {
  return ({
    submitting: "正在提交",
    queued: "排队中",
    running: "运行中",
    finalizing: "正在收尾",
    succeeded: "已完成",
    failed: "失败",
    submission_unknown: "提交状态未知",
  } as Record<string, string>)[status] || status;
}

type TaskFilter = "all" | "active" | "succeeded" | "failed";

const activeStatuses = new Set(["submitting", "queued", "running", "finalizing"]);

function taskProgress(status: string) {
  return ({
    submitting: 8,
    queued: 20,
    running: 68,
    finalizing: 90,
    succeeded: 100,
    failed: 100,
    submission_unknown: 100,
  } as Record<string, number>)[status] || 0;
}

function taskHint(task: ColorGradingTask) {
  if (task.status === "submitting") return "正在把任务发送到仿色服务";
  if (task.status === "queued") {
    return task.queue_position
      ? `前方还有 ${task.queue_position} 个任务`
      : "任务已进入队列，等待可用算力";
  }
  if (task.status === "running") return "正在分析参考风格并逐帧应用色彩";
  if (task.status === "finalizing") return "正在校验成片并合并原视频音轨";
  if (task.status === "succeeded") return "成片已生成，可以预览、下载或加入素材库";
  if (task.status === "submission_unknown") return "暂时无法确认上游是否收到任务";
  if (task.status === "failed") return task.error_message || "任务处理失败";
  return task.stage || "等待状态更新";
}

function formatDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return new Intl.DateTimeFormat("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  }).format(date);
}


export function ColorGradingPage({ status, tasks, videos, refresh, setNotice }: {
  status: ColorGradingCapability;
  tasks: ColorGradingTask[];
  videos: Video[];
  refresh: () => Promise<void>;
  setNotice: (value: string) => void;
}) {
  const [inputVideoId, setInputVideoId] = useState("");
  const [referenceType, setReferenceType] = useState<"image" | "video">("image");
  const [referenceVideoId, setReferenceVideoId] = useState("");
  const [referenceImage, setReferenceImage] = useState<File>();
  const [submitting, setSubmitting] = useState(false);
  const [taskFilter, setTaskFilter] = useState<TaskFilter>("all");
  const referenceImageUrl = useMemo(
    () => referenceImage ? URL.createObjectURL(referenceImage) : "",
    [referenceImage],
  );
  useEffect(() => () => {
    if (referenceImageUrl) URL.revokeObjectURL(referenceImageUrl);
  }, [referenceImageUrl]);
  useEffect(() => {
    if (!inputVideoId && videos.length) setInputVideoId(videos[0].id);
  }, [videos.length, inputVideoId]);
  useEffect(() => {
    if (referenceVideoId === inputVideoId) setReferenceVideoId("");
  }, [inputVideoId, referenceVideoId]);
  const inputVideo = videos.find(video => video.id === inputVideoId);
  const referenceVideo = videos.find(video => video.id === referenceVideoId);
  const activeTaskCount = tasks.filter(task => activeStatuses.has(task.status)).length;
  const succeededTaskCount = tasks.filter(task => task.status === "succeeded").length;
  const failedTaskCount = tasks.filter(task => ["failed", "submission_unknown"].includes(task.status)).length;
  const filteredTasks = tasks.filter(task => {
    if (taskFilter === "active") return activeStatuses.has(task.status);
    if (taskFilter === "succeeded") return task.status === "succeeded";
    if (taskFilter === "failed") return ["failed", "submission_unknown"].includes(task.status);
    return true;
  });
  const referenceReady = referenceType === "image" ? !!referenceImage : !!referenceVideoId;
  const canSubmit = status.available && !!inputVideoId && referenceReady && !submitting;
  const submit = async () => {
    if (!status.available) return setNotice(status.reason || "仿色服务尚未就绪");
    if (!inputVideoId) return setNotice("请选择原视频");
    if (referenceType === "image" && !referenceImage) return setNotice("请选择参考图片");
    if (referenceType === "video" && !referenceVideoId) return setNotice("请选择参考视频");
    setSubmitting(true);
    try {
      await api.createColorGradingTask({
        inputVideoId,
        referenceType,
        referenceImage,
        referenceVideoId: referenceType === "video" ? referenceVideoId : undefined,
      });
      setReferenceImage(undefined);
      await refresh();
      setNotice("仿色任务已进入队列");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "仿色任务提交失败");
    } finally {
      setSubmitting(false);
    }
  };
  const importResult = async (task: ColorGradingTask) => {
    try {
      await api.importColorGradingResult(task.id);
      await refresh();
      setNotice("仿色结果已加入视频资产");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "加入素材库失败");
    }
  };
  return <div className="grading-page">
    <section className={`grading-hero ${status.available ? "available" : "unavailable"}`}>
      <div className="grading-hero-copy">
        <span className="panel-label">AI COLOR TRANSFER</span>
        <h2>把参考画面的色彩语言<br />迁移到你的原片</h2>
        <p>保留原视频内容与声音，根据参考图片或视频生成 LUT 并完成整片调色。</p>
        <div className="grading-service-line">
          <span className={`service-dot ${status.available ? "online" : "offline"}`} />
          <b>{status.available ? "仿色服务在线" : "仿色服务不可用"}</b>
          <span>{status.available ? `${status.device || "加速设备"} · 串行处理` : status.reason}</span>
        </div>
      </div>
      <div className="grading-overview">
        <div><b>{activeTaskCount}</b><span>处理中</span></div>
        <div><b>{succeededTaskCount}</b><span>已完成</span></div>
        <div><b>{videos.length}</b><span>可用原片</span></div>
      </div>
    </section>
    <div className="grading-workspace">
      <div className="panel grading-form">
        <div className="grading-form-head">
          <div><span className="panel-label">NEW GRADING TASK</span><h2>创建仿色任务</h2></div>
          <span className="grading-step-count">2 步完成</span>
        </div>
        <div className="grading-step">
          <div className="grading-step-title"><span>1</span><div><b>选择需要调色的原片</b><small>画面内容和原始音轨都会保留</small></div></div>
          <label className="grading-select">
            <span>原视频</span>
            <select value={inputVideoId} onChange={event => setInputVideoId(event.target.value)}>
              <option value="">选择原视频</option>
              {videos.map(video => <option value={video.id} key={video.id}>{video.name}</option>)}
            </select>
          </label>
          <div className={`grading-preview ${inputVideo ? "has-media" : ""}`}>
            {inputVideo
              ? <><video controls preload="metadata" src={`/api/videos/${inputVideo.id}/media`} /><span className="preview-badge">原片</span></>
              : <div className="grading-placeholder"><span>▶</span><b>请选择原视频</b></div>}
          </div>
          {inputVideo && <div className="grading-media-meta">
            <b title={inputVideo.name}>{inputVideo.name}</b>
            <span>{Math.round(inputVideo.duration)} 秒 · {inputVideo.width} × {inputVideo.height}</span>
          </div>}
        </div>
        <div className="grading-step">
          <div className="grading-step-title"><span>2</span><div><b>选择参考风格</b><small>图片适合定向风格，视频适合动态光线</small></div></div>
          <div className="reference-tabs" role="tablist" aria-label="参考素材类型">
            <button type="button" role="tab" aria-selected={referenceType === "image"} className={referenceType === "image" ? "selected" : ""} onClick={() => setReferenceType("image")}><span>▧</span>参考图片</button>
            <button type="button" role="tab" aria-selected={referenceType === "video"} className={referenceType === "video" ? "selected" : ""} onClick={() => setReferenceType("video")}><span>▶</span>参考视频</button>
          </div>
          {referenceType === "image"
            ? <label className={`image-drop grading-reference ${referenceImage ? "has-image" : ""}`}>
                <input type="file" accept=".jpg,.jpeg,.png,.webp,image/*" onChange={event => setReferenceImage(event.target.files?.[0])} />
                {referenceImage
                  ? <><img src={referenceImageUrl} alt="参考图片预览" /><span><b>{referenceImage.name}</b><small>点击可重新选择</small></span></>
                  : <><span className="upload-glyph">＋</span><b>上传参考图片</b><small>JPG / PNG / WebP，最大 25 MB</small></>}
              </label>
            : <>
                <label className="grading-select">
                  <span>参考视频</span>
                  <select value={referenceVideoId} onChange={event => setReferenceVideoId(event.target.value)}>
                    <option value="">选择参考视频</option>
                    {videos.filter(video => video.id !== inputVideoId).map(video => <option value={video.id} key={video.id}>{video.name}</option>)}
                  </select>
                </label>
                <div className={`grading-preview ${referenceVideo ? "has-media" : ""}`}>
                  {referenceVideo
                    ? <><video controls preload="metadata" src={`/api/videos/${referenceVideo.id}/media`} /><span className="preview-badge reference">参考</span></>
                    : <div className="grading-placeholder"><span>◇</span><b>请选择参考视频</b></div>}
                </div>
              </>}
        </div>
        <div className={`grading-submit-summary ${canSubmit ? "ready" : ""}`}>
          <span>{!status.available ? "服务离线" : !inputVideoId ? "请选择原视频" : !referenceReady ? "还差一个参考素材" : "素材已就绪，可以开始仿色"}</span>
          <i>{referenceType === "image" ? "图片参考" : "视频参考"}</i>
        </div>
        <button className="primary grading-submit" disabled={!canSubmit} onClick={submit}>
          <span>{submitting ? "正在提交任务…" : "开始生成仿色视频"}</span>
          {submitting ? <span className="spinner" /> : <span>→</span>}
        </button>
      </div>
      <div className="grading-history">
        <div className="grading-history-head">
          <div><span className="panel-label">TASK HISTORY</span><h2>处理记录</h2><p>任务状态会自动刷新，完成后可直接对比原片与成片。</p></div>
          <span>{tasks.length} 个任务</span>
        </div>
        <div className="grading-filters" role="tablist" aria-label="筛选仿色任务">
          {([
            ["all", "全部", tasks.length],
            ["active", "处理中", activeTaskCount],
            ["succeeded", "已完成", succeededTaskCount],
            ["failed", "失败", failedTaskCount],
          ] as [TaskFilter, string, number][]).map(([value, label, count]) =>
            <button type="button" role="tab" aria-selected={taskFilter === value} className={taskFilter === value ? "selected" : ""} onClick={() => setTaskFilter(value)} key={value}>{label}<span>{count}</span></button>
          )}
        </div>
        <div className="grading-task-list">
          {filteredTasks.map(task => <article className={`panel grading-task ${activeStatuses.has(task.status) ? "is-active" : ""}`} key={task.id}>
            <div className="grading-task-head">
              <div className="grading-task-title">
                <span className="grading-task-icon">◐</span>
                <div><b title={task.input_video_name}>{task.input_video_name}</b><small>{formatDate(task.created_at)} · {task.reference_type === "image" ? "图片参考" : `视频参考：${task.reference_video_name || "—"}`}</small></div>
              </div>
              <span className={`status ${task.status}`}>{statusText(task.status)}</span>
            </div>
            <div className="grading-task-progress">
              <div><span>{taskHint(task)}</span><b>{taskProgress(task.status)}%</b></div>
              <div className={`grading-progress-track ${["failed", "submission_unknown"].includes(task.status) ? "failed" : ""}`}><span style={{ width: `${taskProgress(task.status)}%` }} /></div>
            </div>
            {task.error_message && task.status !== "failed" && <p className="grading-error">{task.error_message}</p>}
            {task.media_url && <div className="grading-compare">
              <div><span>原片</span><video controls preload="metadata" src={`/api/videos/${task.input_video_id}/media`} /></div>
              <div><span>仿色结果</span><video controls preload="metadata" src={task.media_url} /></div>
            </div>}
            {task.status === "succeeded" && <div className="grading-actions">
              {task.media_url && <a className="outline action-primary" href={task.media_url} download>↓ 下载成片</a>}
              {task.lut_url && <a className="outline" href={task.lut_url} download>下载 LUT</a>}
              <button className="outline" disabled={!!task.imported_video_id} onClick={() => importResult(task)}>{task.imported_video_id ? "✓ 已加入素材库" : "＋ 加入素材库"}</button>
            </div>}
          </article>)}
          {!filteredTasks.length && <div className="panel grading-empty">
            <span>◌</span>
            <b>{tasks.length ? "这个筛选下还没有任务" : "还没有仿色任务"}</b>
            <p>{tasks.length ? "切换其他状态查看处理记录。" : "从左侧选择原视频和参考风格，创建第一个任务。"}</p>
          </div>}
        </div>
      </div>
    </div>
  </div>;
}
