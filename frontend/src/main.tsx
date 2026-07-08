import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, Entity, Job, SearchResult, Video } from "./api";
import "./styles.css";

type Page = "overview" | "indexes" | "assets" | "entities" | "search";
type VisualSegmentPresetKey = "fixed" | "general" | "fast-cut" | "talking-head" | "custom";
type VisualShotDetector = "simple" | "pyscenedetect_content" | "pyscenedetect_adaptive";

const visualShotDetectorLabels: Record<VisualShotDetector, string> = {
  simple: "Simple 帧差",
  pyscenedetect_content: "PySceneDetect Content",
  pyscenedetect_adaptive: "PySceneDetect Adaptive",
};

const visualSegmentPresets: Record<VisualSegmentPresetKey, {
  label: string;
  strategy: "fixed" | "shot";
  detector: VisualShotDetector;
  minSeconds: number;
  maxSeconds: number;
  threshold: number;
}> = {
  fixed: { label: "固定分段", strategy: "fixed", detector: "simple", minSeconds: 0.8, maxSeconds: 8, threshold: 0.2 },
  general: { label: "通用镜头", strategy: "shot", detector: "simple", minSeconds: 0.8, maxSeconds: 8, threshold: 0.2 },
  "fast-cut": { label: "广告 / MV", strategy: "shot", detector: "pyscenedetect_adaptive", minSeconds: 0.5, maxSeconds: 6, threshold: 0.16 },
  "talking-head": { label: "访谈长镜头", strategy: "shot", detector: "pyscenedetect_content", minSeconds: 1.5, maxSeconds: 12, threshold: 0.28 },
  custom: { label: "自定义", strategy: "shot", detector: "simple", minSeconds: 0.8, maxSeconds: 8, threshold: 0.2 },
};

const icons: Record<Page, string> = {
  overview: "⌂",
  indexes: "▣",
  assets: "+",
  entities: "◎",
  search: "⌕",
};

const labels: Record<Page, string> = {
  overview: "概览",
  indexes: "索引任务",
  assets: "视频资产",
  entities: "人物库",
  search: "检索",
};

function formatTime(seconds: number) {
  const value = Math.max(0, Math.floor(seconds));
  const hours = Math.floor(value / 3600);
  const minutes = Math.floor((value % 3600) / 60);
  const rest = value % 60;
  return `${hours ? `${String(hours).padStart(2, "0")}:` : ""}${String(minutes).padStart(2, "0")}:${String(rest).padStart(2, "0")}`;
}

function formatDuration(seconds?: number | null) {
  if (seconds == null || Number.isNaN(seconds)) return "—";
  if (seconds < 1) return `${Math.round(seconds * 1000)}ms`;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return `${minutes}m ${rest}s`;
}

function statusText(status: string) {
  return ({ uploaded: "待索引", indexing: "索引中", ready: "可检索", failed: "失败", queued: "排队中", running: "运行中", completed: "已完成" } as Record<string, string>)[status] || status;
}

function App() {
  const [page, setPage] = useState<Page>("search");
  const [videos, setVideos] = useState<Video[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [entities, setEntities] = useState<Entity[]>([]);
  const [notice, setNotice] = useState<string>("");

  const refresh = async () => {
    try {
      const [nextVideos, nextJobs, nextEntities] = await Promise.all([api.videos(), api.jobs(), api.entities()]);
      setVideos(nextVideos); setJobs(nextJobs); setEntities(nextEntities);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "服务连接失败");
    }
  };

  useEffect(() => {
    refresh();
    const timer = window.setInterval(refresh, 3000);
    return () => window.clearInterval(timer);
  }, []);

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><span className="brand-mark">M</span><span>MomentSeek</span></div>
        <button className="new-button" onClick={() => setPage("assets")}><span>上传视频</span><b>＋</b></button>
        <nav>
          {(["overview", "indexes", "assets", "entities", "search"] as Page[]).map(item => (
            <button key={item} className={page === item ? "active" : ""} onClick={() => setPage(item)}>
              <span className="nav-icon">{icons[item]}</span>{labels[item]}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="resource-pill"><span className="pulse" />模型按任务加载</div>
          <small>空闲时不占用 NPU 显存</small>
        </div>
      </aside>

      <main>
        <header className="topbar">
          <div><span className="eyebrow">PRIVATE VIDEO INTELLIGENCE</span><h1>{labels[page]}</h1></div>
          <div className="top-actions"><a href="/docs" target="_blank">API Docs</a><span className="avatar">MS</span></div>
        </header>
        {notice && <div className="notice" onClick={() => setNotice("")}>{notice}<span>×</span></div>}
        <section className="page-content">
          {page === "search" && <SearchPage videos={videos} setNotice={setNotice} />}
          {page === "assets" && <AssetsPage videos={videos} refresh={refresh} setNotice={setNotice} />}
          {page === "indexes" && <IndexesPage jobs={jobs} videos={videos} />}
          {page === "entities" && <EntitiesPage entities={entities} refresh={refresh} setNotice={setNotice} />}
          {page === "overview" && <Overview videos={videos} jobs={jobs} entities={entities} setPage={setPage} />}
        </section>
      </main>
    </div>
  );
}

function SearchPage({ videos, setNotice }: { videos: Video[]; setNotice: (value: string) => void }) {
  const ready = videos.filter(video => video.status === "ready" || video.indexed_modalities.length);
  const [selected, setSelected] = useState<string[]>([]);
  const [query, setQuery] = useState("");
  const [image, setImage] = useState<File>();
  const [modalities, setModalities] = useState(["visual", "face", "asr", "ocr"]);
  const [alpha, setAlpha] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [searchElapsed, setSearchElapsed] = useState<number | undefined>();
  const [playing, setPlaying] = useState<SearchResult>();

  useEffect(() => { if (!selected.length && ready.length) setSelected(ready.map(video => video.id)); }, [ready.length]);
  const toggleMode = (mode: string) => setModalities(value => value.includes(mode) ? value.filter(item => item !== mode) : [...value, mode]);
  const submit = async () => {
    if (!query.trim() && !image) return setNotice("请输入文字或上传参考图");
    if (!selected.length) return setNotice("请先选择至少一个已建立索引的视频");
    if (!modalities.length) return setNotice("请至少启用一个检索通道");
    setLoading(true);
    try {
      const response = await api.search({ queryText: query.trim(), queryImage: image, modalities, videoIds: selected, alpha });
      setResults(response.results);
      setSearchElapsed(response.elapsed_seconds);
      if (!response.results.length) setNotice("没有超过当前阈值的片段，可以换个描述或通道再试");
    } catch (error) { setNotice(error instanceof Error ? error.message : "检索失败"); }
    finally { setLoading(false); }
  };

  const aboveCount = results.filter(result => result.above_threshold !== false).length;
  const belowCount = results.length - aboveCount;
  const firstBelow = results.findIndex(result => result.above_threshold === false);

  return <div className="search-layout">
    <div className="query-panel panel">
      <div className="panel-label">SEARCH BUILDER</div>
      <label>检索范围</label>
      <div className="index-select">
        {ready.length ? ready.map(video => <label className="check-row" key={video.id}>
          <input type="checkbox" checked={selected.includes(video.id)} onChange={() => setSelected(value => value.includes(video.id) ? value.filter(id => id !== video.id) : [...value, video.id])} />
          <span className="video-dot" />
          <span><b>{video.name}</b><small>{formatTime(video.duration)} · {video.indexed_modalities.join(" / ")}</small></span>
        </label>) : <div className="empty-mini">还没有可检索的视频，请先上传并建立索引。</div>}
      </div>

      <label>查询文字</label>
      <textarea value={query} onChange={event => setQuery(event.target.value)} placeholder="例如：a person speaking on stage / 提到电影投资的片段" />

      <label>参考图 <em>可选</em></label>
      <label className={`image-drop ${image ? "has-image" : ""}`}>
        <input type="file" accept="image/*" onChange={event => setImage(event.target.files?.[0])} />
        {image ? <><img src={URL.createObjectURL(image)} /><span>{image.name}</span></> : <><span className="upload-glyph">↥</span><b>添加人物、物体或场景参考图</b><small>JPG / PNG / WEBP</small></>}
      </label>

      <label>检索通道</label>
      <div className="mode-grid">
        {[['visual','Visual','场景与物体'],['face','Face','同一人物'],['asr','ASR','语音内容'],['ocr','OCR','画面文字']].map(([value,title,sub]) =>
          <button key={value} className={modalities.includes(value) ? "selected" : ""} onClick={() => toggleMode(value)}>
            <span>{modalities.includes(value) ? "✓" : "+"}</span><b>{title}</b><small>{sub}</small>
          </button>)}
      </div>
      {query && image && modalities.includes("visual") && <div className="alpha-control"><span>文字权重 {Math.round(alpha * 100)}%</span><input type="range" min="0" max="1" step="0.05" value={alpha} onChange={event => setAlpha(Number(event.target.value))} /></div>}
      <button className="primary" disabled={loading} onClick={submit}>{loading ? <><span className="spinner" />正在检索</> : <>开始检索 <span>→</span></>}</button>
    </div>

    <div className="results-panel">
      <div className="results-head"><div><span className="panel-label">MOMENTS</span><h2>{results.length ? `${results.length} 个相关片段` : "在视频中找到那个瞬间"}</h2>{searchElapsed !== undefined && <small className="time-note">检索耗时 {formatDuration(searchElapsed)}{belowCount > 0 ? ` · ${aboveCount} 命中 / ${belowCount} 低于阈值` : ""}</small>}</div>{results.length > 0 && <button className="text-button" onClick={() => setResults([])}>清空结果</button>}</div>
      {!results.length ? <div className="examples">
        <Example title="参考图中的人物" tags={["FACE", "VISUAL"]} text="上传一张清晰正脸，找出人物所有出现片段" setQuery={setQuery} />
        <Example title="舞台上讲话" tags={["VISUAL"]} text="a person speaking on a stage" setQuery={setQuery} />
        <Example title="语音中提到某话题" tags={["ASR", "LEXICAL"]} text="电影投资" setQuery={setQuery} />
        <Example title="画面中出现文字" tags={["OCR", "TEXT"]} text="logo on screen" setQuery={setQuery} />
        <Example title="红色行李箱" tags={["VISUAL", "IMAGE"]} text="a red suitcase" setQuery={setQuery} />
      </div> : <div className="result-grid">{results.map((result, index) => <React.Fragment key={`${result.video_id}-${result.start_time}-${index}`}>{index === firstBelow && <div className="threshold-divider"><span>以下片段低于阈值 · 仅供参考</span></div>}<ResultCard result={result} onPlay={() => setPlaying(result)} /></React.Fragment>)}</div>}
    </div>
    {playing && <PlayerModal result={playing} onClose={() => setPlaying(undefined)} />}
  </div>;
}

function Example({ title, tags, text, setQuery }: { title: string; tags: string[]; text: string; setQuery: (value: string) => void }) {
  return <button className="example-card" onClick={() => setQuery(text)}><span className="spark">✦</span><h3>{title}</h3><p>{text}</p><div>{tags.map(tag => <span key={tag}>{tag}</span>)}</div></button>;
}

function ResultCard({ result, onPlay }: { result: SearchResult; onPlay: () => void }) {
  const below = result.above_threshold === false;
  return <article className={`result-card${below ? " below" : ""}`} onClick={onPlay}>
    <div className="result-thumb">{result.thumbnail_url ? <img src={result.thumbnail_url} /> : <div className="thumb-placeholder">M</div>}<button>▶</button><span>{formatTime(result.start_time)} — {formatTime(result.end_time)}</span></div>
    <div className="result-body"><div className="result-title"><h3>{result.video_name}</h3><b>{Math.round(result.score * 100)}%</b></div><div className="chips">{result.modalities.map(mode => <span className={`chip ${mode}`} key={mode}>{mode}</span>)}{below && <span className="chip below-tag">低于阈值</span>}</div><p>{result.evidence.find(item => item.detail)?.detail || "视觉向量相似度命中"}</p></div>
  </article>;
}

function PlayerModal({ result, onClose }: { result: SearchResult; onClose: () => void }) {
  const ref = useRef<HTMLVideoElement>(null);
  const sourceUrl = result.clip_url || result.media_url;
  const sourceStart = result.clip_url ? 0 : result.start_time;
  const sourceEnd = result.clip_url ? Math.max(0.25, result.end_time - result.start_time) : result.end_time;
  // Restrict playback to the matched [start, end] window: seek in on load, and
  // loop back to start once playback passes the segment end.
  const clampToSegment = () => {
    const video = ref.current;
    if (!video) return;
    if (video.currentTime >= sourceEnd || video.currentTime < sourceStart - 0.5) {
      video.currentTime = sourceStart;
    }
  };
  return <div className="modal-backdrop" onMouseDown={onClose}><div className="player-modal" onMouseDown={event => event.stopPropagation()}><button className="close" onClick={onClose}>×</button><video ref={ref} src={sourceUrl} controls autoPlay onLoadedMetadata={() => { if (ref.current) ref.current.currentTime = sourceStart; }} onTimeUpdate={clampToSegment} /><div className="player-info"><div><span className="panel-label">MATCHED MOMENT</span><h3>{result.video_name}</h3></div><b>{formatTime(result.start_time)} — {formatTime(result.end_time)} · 仅循环播放命中片段</b></div></div></div>;
}

function AssetsPage({ videos, refresh, setNotice }: { videos: Video[]; refresh: () => Promise<void>; setNotice: (value: string) => void }) {
  const [videoFile, setVideoFile] = useState<File>();
  const [transcript, setTranscript] = useState<File>();
  const [visualModel, setVisualModel] = useState("siglip2-so400m-384");
  const [visualSampleFps, setVisualSampleFps] = useState(5);
  const [visualSegmentSeconds, setVisualSegmentSeconds] = useState(5);
  const [visualSegmentPreset, setVisualSegmentPreset] = useState<VisualSegmentPresetKey>("fixed");
  const [visualMinSegmentSeconds, setVisualMinSegmentSeconds] = useState(0.8);
  const [visualMaxSegmentSeconds, setVisualMaxSegmentSeconds] = useState(8);
  const [visualShotDetector, setVisualShotDetector] = useState<VisualShotDetector>("simple");
  const [visualShotThreshold, setVisualShotThreshold] = useState(0.2);
  const [faceSampleFps, setFaceSampleFps] = useState(2);
  const [includeOcr, setIncludeOcr] = useState(false);
  const [ocrSampleFps, setOcrSampleFps] = useState(0.05);
  const [asrModel, setAsrModel] = useState("small");
  const [asrLanguage, setAsrLanguage] = useState("zh");
  const [uploading, setUploading] = useState(false);
  const useShotSegments = visualSegmentPresets[visualSegmentPreset].strategy === "shot";
  const applyVisualSegmentPreset = (next: VisualSegmentPresetKey) => {
    setVisualSegmentPreset(next);
    if (next === "custom") return;
    const preset = visualSegmentPresets[next];
    setVisualMinSegmentSeconds(preset.minSeconds);
    setVisualMaxSegmentSeconds(preset.maxSeconds);
    setVisualShotDetector(preset.detector);
    setVisualShotThreshold(preset.threshold);
  };
  const upload = async () => {
    if (!videoFile) return setNotice("请先选择视频");
    setUploading(true);
    try { await api.uploadVideo(videoFile, transcript); setVideoFile(undefined); setTranscript(undefined); await refresh(); setNotice("视频上传完成，可以开始建立索引"); }
    catch (error) { setNotice(error instanceof Error ? error.message : "上传失败"); }
    finally { setUploading(false); }
  };
  const index = async (id: string) => {
    if (useShotSegments && visualMinSegmentSeconds > visualMaxSegmentSeconds) {
      return setNotice("镜头最短时长不能大于最长时长");
    }
    try {
      const selectedModalities = includeOcr ? ["visual", "face", "asr", "ocr"] : ["visual", "face", "asr"];
      await api.indexVideo(id, selectedModalities, {
        visualModel,
        visualSampleFps,
        visualSegmentSeconds,
        visualSegmentStrategy: useShotSegments ? "shot" : "fixed",
        visualMinSegmentSeconds: useShotSegments ? visualMinSegmentSeconds : undefined,
        visualMaxSegmentSeconds: useShotSegments ? visualMaxSegmentSeconds : undefined,
        visualShotDetector: useShotSegments ? visualShotDetector : undefined,
        visualShotThreshold: useShotSegments ? visualShotThreshold : undefined,
        faceSampleFps,
        ocrSampleFps: includeOcr ? ocrSampleFps : undefined,
        asrModel,
        asrLanguage,
      });
      await refresh();
      const segmentLabel = useShotSegments
        ? `${visualSegmentPresets[visualSegmentPreset].label} ${visualShotDetectorLabels[visualShotDetector]} ${visualMinSegmentSeconds}-${visualMaxSegmentSeconds}s / 阈值 ${visualShotThreshold}`
        : `固定 ${visualSegmentSeconds}s`;
      setNotice(`索引任务已进入队列：Visual ${visualModel} · ${visualSampleFps}fps / ${segmentLabel}，Face ${faceSampleFps}fps${includeOcr ? `，OCR ${ocrSampleFps}fps` : ""}，ASR ${asrModel}/${asrLanguage}`);
    }
    catch (error) { setNotice(error instanceof Error ? error.message : "任务创建失败"); }
  };
  const rename = async (video: Video) => {
    const next = window.prompt("重命名视频", video.name);
    if (next === null) return;
    const name = next.trim();
    if (!name || name === video.name) return;
    try { await api.renameVideo(video.id, name); await refresh(); setNotice("已重命名"); }
    catch (error) { setNotice(error instanceof Error ? error.message : "重命名失败"); }
  };
  const remove = async (video: Video) => {
    if (!window.confirm(`确定删除「${video.name}」？将同时删除其索引、缩略图和上传文件，且不可恢复。`)) return;
    try { await api.deleteVideo(video.id); await refresh(); setNotice("已删除视频及其索引"); }
    catch (error) { setNotice(error instanceof Error ? error.message : "删除失败"); }
  };
  return <div className="stack-page"><div className="upload-panel panel"><div><span className="panel-label">NEW ASSET</span><h2>添加视频素材</h2><p>上传后可一次建立 Face、Visual 和 ASR 三路索引。</p></div><label className="file-line"><input type="file" accept="video/*" onChange={event => setVideoFile(event.target.files?.[0])} /><span>{videoFile?.name || "选择视频文件"}</span><b>浏览</b></label><label className="file-line secondary"><input type="file" accept=".json,.srt,.vtt" onChange={event => setTranscript(event.target.files?.[0])} /><span>{transcript?.name || "可选：已有字幕 JSON / SRT / VTT"}</span><b>添加</b></label><button className="primary compact" onClick={upload} disabled={uploading}>{uploading ? "正在上传…" : "上传素材"}</button></div>
    <div className="index-options panel">
      <div>
        <span className="panel-label">INDEX OPTIONS</span>
        <h2>索引参数</h2>
        <p>Visual 默认 5fps；OCR 需显式勾选，服务器优先走 NPU/CANN，长视频建议先低采样；ASR 默认 small。</p>
      </div>
      <label>Visual model<select value={visualModel} onChange={event => setVisualModel(event.target.value)}><option value="siglip2-so400m-384">SigLIP2 So400m-384 默认</option><option value="chinese-clip-vit-b16">ChineseCLIP ViT-B/16</option><option value="openclip-vit-b32">OpenCLIP ViT-B/32</option><option value="openclip-vit-b16">OpenCLIP ViT-B/16</option><option value="openclip-vit-l14">OpenCLIP ViT-L/14</option></select></label>
      <label>Visual fps<input type="number" min="0.2" max="10" step="0.5" value={visualSampleFps} onChange={event => setVisualSampleFps(Number(event.target.value))} /></label>
      <label>Visual 分段<select value={visualSegmentPreset} onChange={event => applyVisualSegmentPreset(event.target.value as VisualSegmentPresetKey)}>{(Object.keys(visualSegmentPresets) as VisualSegmentPresetKey[]).map(key => <option key={key} value={key}>{visualSegmentPresets[key].label}</option>)}</select></label>
      <label>{useShotSegments ? "固定回退秒数" : "Visual 分段秒数"}<input type="number" min="1" max="60" step="1" value={visualSegmentSeconds} onChange={event => setVisualSegmentSeconds(Number(event.target.value))} /></label>
      {useShotSegments && <>
        <label>镜头检测器<select value={visualShotDetector} onChange={event => setVisualShotDetector(event.target.value as VisualShotDetector)}>{(Object.keys(visualShotDetectorLabels) as VisualShotDetector[]).map(key => <option key={key} value={key}>{visualShotDetectorLabels[key]}</option>)}</select></label>
        <label>镜头最短秒数<input type="number" min="0.2" max="30" step="0.1" value={visualMinSegmentSeconds} onChange={event => setVisualMinSegmentSeconds(Number(event.target.value))} /></label>
        <label>镜头最长秒数<input type="number" min="1" max="60" step="0.5" value={visualMaxSegmentSeconds} onChange={event => setVisualMaxSegmentSeconds(Number(event.target.value))} /></label>
        <label>切分阈值<input type="number" min="0.05" max="0.6" step="0.01" value={visualShotThreshold} onChange={event => setVisualShotThreshold(Number(event.target.value))} /></label>
      </>}
      <label>Face fps<input type="number" min="0.2" max="15" step="0.5" value={faceSampleFps} onChange={event => setFaceSampleFps(Number(event.target.value))} /></label>
      <label className="inline-check"><input type="checkbox" checked={includeOcr} onChange={event => setIncludeOcr(event.target.checked)} />包含 OCR</label>
      <label>OCR fps<input type="number" min="0.02" max="5" step="0.05" value={ocrSampleFps} onChange={event => setOcrSampleFps(Number(event.target.value))} /></label>
      <label>ASR 模型<select value={asrModel} onChange={event => setAsrModel(event.target.value)}><option value="base">base 更快</option><option value="small">small 推荐</option><option value="medium">medium 更准更慢</option><option value="large-v3">large-v3 高风险</option></select></label>
      <label>ASR 语言<select value={asrLanguage} onChange={event => setAsrLanguage(event.target.value)}><option value="zh">中文</option><option value="en">English</option><option value="auto">Auto</option></select></label>
    </div>
    <div className="table-panel panel"><div className="section-head"><div><span className="panel-label">LIBRARY</span><h2>视频资产</h2></div><span>{videos.length} items</span></div><div className="asset-list">{videos.map(video => <div className="asset-row" key={video.id}><div className="asset-icon">▶</div><div className="asset-main"><b>{video.name}</b><small>{formatTime(video.duration)} · {video.width}×{video.height} · {video.fps.toFixed(1)} fps</small></div><div className="chips">{video.indexed_modalities.map(mode => <span className={`chip ${mode}`} key={mode}>{mode}</span>)}</div><span className={`status ${video.status}`}>{statusText(video.status)}</span><div className="asset-actions">{video.status !== "indexing" && <button className="outline" onClick={() => index(video.id)}>{video.indexed_modalities.length ? "重建索引" : "建立索引"}</button>}<button className="outline" onClick={() => rename(video)}>重命名</button><button className="outline danger" disabled={video.status === "indexing"} onClick={() => remove(video)}>删除</button></div></div>)}{!videos.length && <div className="empty-list">还没有视频素材</div>}</div></div>
  </div>;
}

function IndexesPage({ jobs, videos }: { jobs: Job[]; videos: Video[] }) {
  const names = Object.fromEntries(videos.map(video => [video.id, video.name]));
  return <div className="table-panel panel"><div className="section-head"><div><span className="panel-label">PIPELINE</span><h2>索引任务</h2></div><span>阶段子进程退出后释放模型</span></div><div className="job-list">{jobs.map(job => {
    const stages = job.metrics?.stages || {};
    return <div className="job-row" key={job.id}><div className={`job-state ${job.status}`}>{job.status === "completed" ? "✓" : job.status === "failed" ? "!" : "↻"}</div><div className="job-main"><b>{names[job.video_id] || job.video_id}</b><small>{job.modalities.join(" → ")} · 当前：{job.stage} · 总耗时 {formatDuration(job.metrics?.total_elapsed_seconds)}</small><div className="stage-times">{["visual", "face", "asr", "ocr"].filter(stage => job.modalities.includes(stage) || stages[stage]).map(stage => <span key={stage}>{stage}: {formatDuration(stages[stage]?.elapsed_seconds)}</span>)}</div><div className="progress"><span style={{ width: `${job.progress * 100}%` }} /></div></div><span className={`status ${job.status}`}>{statusText(job.status)}</span>{job.error && <p>{job.error}</p>}</div>;
  })}{!jobs.length && <div className="empty-list">还没有索引任务</div>}</div></div>;
}

function EntitiesPage({ entities, refresh, setNotice }: { entities: Entity[]; refresh: () => Promise<void>; setNotice: (value: string) => void }) {
  const [name, setName] = useState(""); const [image, setImage] = useState<File>(); const [saving, setSaving] = useState(false);
  const save = async () => { if (!name.trim() || !image) return setNotice("请输入人物名称并选择清晰正脸"); setSaving(true); try { await api.createEntity(name.trim(), image); setName(""); setImage(undefined); await refresh(); setNotice("人物已登记，以后可以直接用名字检索"); } catch (error) { setNotice(error instanceof Error ? error.message : "人物登记失败"); } finally { setSaving(false); } };
  return <div className="entity-layout"><div className="entity-create panel"><span className="panel-label">NEW ENTITY</span><h2>登记人物</h2><p>人物名称与参考脸绑定后，可用“内马尔”等名称调用 face_index。</p><input className="text-input" value={name} onChange={event => setName(event.target.value)} placeholder="人物或明星名称" /><label className="portrait-drop"><input type="file" accept="image/*" onChange={event => setImage(event.target.files?.[0])} />{image ? <img src={URL.createObjectURL(image)} /> : <><span>◎</span><b>选择清晰正脸</b></>}</label><button className="primary compact" disabled={saving} onClick={save}>{saving ? "正在提取人脸…" : "登记人物"}</button></div><div className="entity-library"><div className="section-head"><div><span className="panel-label">FACE LIBRARY</span><h2>人物库</h2></div><span>{entities.length} entities</span></div><div className="entity-grid">{entities.map(entity => <article key={entity.id}><img src={`/api/entities/${entity.id}/reference`} /><div><b>{entity.name}</b><small>{entity.embedding_path ? "Face embedding ready" : "Processing"}</small></div></article>)}{!entities.length && <div className="empty-list">还没有登记人物</div>}</div></div></div>;
}

function Overview({ videos, jobs, entities, setPage }: { videos: Video[]; jobs: Job[]; entities: Entity[]; setPage: (page: Page) => void }) {
  const ready = videos.filter(video => video.status === "ready").length;
  return <div className="overview"><div className="hero panel"><div><span className="panel-label">MVP BASELINE</span><h2>让长视频变成<br />可以搜索的素材库。</h2><p>Face / Visual / ASR / OCR 四路独立索引，保留时间证据；模型只在索引阶段短暂加载。</p><button className="primary compact" onClick={() => setPage("assets")}>添加第一条视频 <span>→</span></button></div><div className="hero-visual"><div className="orbit one">Face</div><div className="orbit two">Visual</div><div className="orbit three">ASR</div><span className="core">M</span></div></div><div className="stats"><article><span>视频资产</span><b>{videos.length}</b><small>{ready} 条可检索</small></article><article><span>人物实体</span><b>{entities.length}</b><small>参考脸向量</small></article><article><span>索引任务</span><b>{jobs.length}</b><small>{jobs.filter(job => job.status === "running").length} 个正在运行</small></article><article><span>NPU 常驻</span><b>0</b><small>任务结束即释放</small></article></div></div>;
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
