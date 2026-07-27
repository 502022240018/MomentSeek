import { useEffect, useState } from "react";

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
  useEffect(() => {
    if (!inputVideoId && videos.length) setInputVideoId(videos[0].id);
  }, [videos.length, inputVideoId]);
  useEffect(() => {
    if (referenceVideoId === inputVideoId) setReferenceVideoId("");
  }, [inputVideoId, referenceVideoId]);
  const inputVideo = videos.find(video => video.id === inputVideoId);
  const referenceVideo = videos.find(video => video.id === referenceVideoId);
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
    <div className={`grading-health panel ${status.available ? "available" : "unavailable"}`}>
      <div><span className="panel-label">OPTIONAL SERVICE</span><h2>{status.available ? "仿色服务已就绪" : "仿色服务正在启动或不可用"}</h2><p>{status.available ? `${status.device || "加速设备"} · 独立容器串行推理` : status.reason}</p></div>
      <span className={`status ${status.available ? "succeeded" : "failed"}`}>{status.available ? "可用" : "不可用"}</span>
    </div>
    <div className="grading-workspace">
      <div className="panel grading-form">
        <span className="panel-label">NEW GRADING TASK</span><h2>创建视频仿色</h2>
        <label>原视频<select value={inputVideoId} onChange={event => setInputVideoId(event.target.value)}><option value="">选择原视频</option>{videos.map(video => <option value={video.id} key={video.id}>{video.name}</option>)}</select></label>
        <div className="grading-preview">{inputVideo ? <video controls preload="metadata" src={`/api/videos/${inputVideo.id}/media`} /> : <span>请选择原视频</span>}</div>
        <div className="reference-tabs"><button className={referenceType === "image" ? "selected" : ""} onClick={() => setReferenceType("image")}>参考图片</button><button className={referenceType === "video" ? "selected" : ""} onClick={() => setReferenceType("video")}>参考视频</button></div>
        {referenceType === "image" ? <label className={`image-drop grading-reference ${referenceImage ? "has-image" : ""}`}><input type="file" accept=".jpg,.jpeg,.png,.webp,image/*" onChange={event => setReferenceImage(event.target.files?.[0])} />{referenceImage ? <><img src={URL.createObjectURL(referenceImage)} /><span>{referenceImage.name}</span></> : <><span className="upload-glyph">↥</span><b>选择参考图片</b><small>JPG / PNG / WebP，最大 25 MB</small></>}</label> : <><label>参考视频<select value={referenceVideoId} onChange={event => setReferenceVideoId(event.target.value)}><option value="">选择参考视频</option>{videos.filter(video => video.id !== inputVideoId).map(video => <option value={video.id} key={video.id}>{video.name}</option>)}</select></label><div className="grading-preview">{referenceVideo ? <video controls preload="metadata" src={`/api/videos/${referenceVideo.id}/media`} /> : <span>请选择参考视频</span>}</div></>}
        <button className="primary" disabled={submitting || !status.available || !videos.length} onClick={submit}><span>{submitting ? "正在提交…" : "开始仿色"}</span>{submitting ? <span className="spinner" /> : <span>→</span>}</button>
      </div>
      <div className="grading-history">
        <div className="section-head"><div><span className="panel-label">TASK HISTORY</span><h2>仿色任务</h2></div><span>{tasks.length} tasks</span></div>
        <div className="grading-task-list">{tasks.map(task => <article className="panel grading-task" key={task.id}>
          <div className="grading-task-head"><div><b>{task.input_video_name}</b><small>{task.reference_type === "image" ? "参考图片" : `参考视频：${task.reference_video_name || "—"}`}</small></div><span className={`status ${task.status}`}>{statusText(task.status)}</span></div>
          {task.status === "queued" && <p>队列位置：{task.queue_position || "等待更新"}</p>}
          {task.status === "running" && <p>模型正在生成 LUT 并逐帧应用颜色风格。</p>}
          {task.status === "finalizing" && <p>正在校验结果并合并原视频音轨。</p>}
          {task.error_message && <p className="grading-error">{task.error_message}</p>}
          {task.media_url && <video className="grading-result" controls preload="metadata" src={task.media_url} />}
          {task.status === "succeeded" && <div className="grading-actions"><a className="outline" href={task.media_url || "#"} download>下载视频</a><a className="outline" href={task.lut_url || "#"} download>下载 LUT</a><button className="outline" disabled={!!task.imported_video_id} onClick={() => importResult(task)}>{task.imported_video_id ? "已加入素材库" : "加入素材库"}</button></div>}
        </article>)}{!tasks.length && <div className="panel empty-list">还没有仿色任务</div>}</div>
      </div>
    </div>
  </div>;
}
