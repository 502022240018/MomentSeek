import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, Entity, Job, OrchestrationProfile, SearchResult, SpeakerView, Video, VoiceHit } from "./api";
import {
  defaultIndexConfiguration,
  IndexConfiguration,
  IndexOptionsPanel,
  indexActionLabel,
  toIndexOptions,
  validateIndexConfiguration,
} from "./indexing";
import "./styles.css";

type Page = "overview" | "indexes" | "assets" | "entities" | "search";

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
  return ({ uploaded: "待索引", indexing: "索引中", ready: "可检索", failed: "失败", cancelled: "已取消", queued: "排队中", running: "运行中", completed: "已完成" } as Record<string, string>)[status] || status;
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
          <div className="resource-pill"><span className="pulse" />索引模型常驻复用</div>
          <small>单队列串行调度</small>
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
          {page === "indexes" && <IndexesPage jobs={jobs} videos={videos} refresh={refresh} setNotice={setNotice} />}
          {page === "entities" && <EntitiesPage entities={entities} videos={videos} refresh={refresh} setNotice={setNotice} />}
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
  const [orchestrationProfiles, setOrchestrationProfiles] = useState<OrchestrationProfile[]>([]);
  const [orchestrationProfile, setOrchestrationProfile] = useState("");
  const [orchestrationStatus, setOrchestrationStatus] = useState("正在加载编排配置…");
  const [plannerMode, setPlannerMode] = useState<"auto" | "off" | "force">("auto");
  const [rerankerMode, setRerankerMode] = useState<"auto" | "off" | "force">("auto");
  const [execution, setExecution] = useState<Record<string, any>>();

  useEffect(() => { if (!selected.length && ready.length) setSelected(ready.map(video => video.id)); }, [ready.length]);
  useEffect(() => {
    api.orchestrationProfiles().then(value => {
      if (!value.enabled) {
        setOrchestrationStatus("后端未启用编排");
        return;
      }
      setOrchestrationProfiles(value.profiles);
      setOrchestrationProfile(value.default_profile);
      setOrchestrationStatus(value.profiles.length ? "已连接" : "没有可用 Profile");
    }).catch(() => setOrchestrationStatus("编排配置加载失败，请刷新页面"));
  }, []);
  const toggleMode = (mode: string) => setModalities(value => value.includes(mode) ? value.filter(item => item !== mode) : [...value, mode]);
  const submit = async () => {
    if (!query.trim() && !image) return setNotice("请输入文字或上传参考图");
    if (!selected.length) return setNotice("请先选择至少一个已建立索引的视频");
    if (!modalities.length) return setNotice("请至少启用一个检索通道");
    setLoading(true);
    try {
      const response = await api.search({
        queryText: query.trim(), queryImage: image, modalities, videoIds: selected, alpha,
        orchestrationProfile: orchestrationProfile || undefined, plannerMode, rerankerMode,
      });
      setResults(response.results);
      setSearchElapsed(response.elapsed_seconds);
      setExecution(response.execution);
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
      <div className="orchestration-controls">
        <div><b>智能检索编排 <span className={orchestrationProfiles.length ? "connected" : ""}>{orchestrationStatus}</span></b><small>记录通路、参数、Prompt 和精排结果</small></div>
        <label>实验 Profile<select disabled={!orchestrationProfiles.length} value={orchestrationProfile} onChange={event => setOrchestrationProfile(event.target.value)}>
          {orchestrationProfiles.length ? orchestrationProfiles.map(profile => <option value={profile.name} key={profile.name}>{profile.name}</option>) : <option>等待后端配置</option>}
        </select></label>
        <label>Planner<select disabled={!orchestrationProfiles.length} value={plannerMode} onChange={event => setPlannerMode(event.target.value as "auto" | "off" | "force")}>
          <option value="auto">自动</option><option value="off">关闭</option><option value="force">强制</option>
        </select></label>
        <label>Reranker<select disabled={!orchestrationProfiles.length} value={rerankerMode} onChange={event => setRerankerMode(event.target.value as "auto" | "off" | "force")}>
          <option value="auto">按计划</option><option value="off">关闭</option><option value="force">强制</option>
        </select></label>
      </div>
      {query && image && modalities.includes("visual") && <div className="alpha-control"><span>文字权重 {Math.round(alpha * 100)}%</span><input type="range" min="0" max="1" step="0.05" value={alpha} onChange={event => setAlpha(Number(event.target.value))} /></div>}
      <button className="primary" disabled={loading} onClick={submit}>{loading ? <><span className="spinner" />正在检索</> : <>开始检索 <span>→</span></>}</button>
    </div>

    <div className="results-panel">
      <div className="results-head"><div><span className="panel-label">MOMENTS</span><h2>{results.length ? `${results.length} 个相关片段` : "在视频中找到那个瞬间"}</h2>{searchElapsed !== undefined && <small className="time-note">检索耗时 {formatDuration(searchElapsed)}{belowCount > 0 ? ` · ${aboveCount} 命中 / ${belowCount} 低于阈值` : ""}</small>}{execution && <details className="execution-trace"><summary>查看本次 Planner / Reranker 执行记录</summary><pre>{JSON.stringify(execution, null, 2)}</pre></details>}</div>{results.length > 0 && <button className="text-button" onClick={() => { setResults([]); setExecution(undefined); }}>清空结果</button>}</div>
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
    <div className="result-body"><div className="result-title"><h3>{result.video_name}</h3><b>{Math.round(result.score * 100)}%</b></div><div className="chips">{result.modalities.map(mode => <span className={`chip ${mode}`} key={mode}>{mode}</span>)}{result.rerank_score !== undefined && result.rerank_score !== null && <span className="chip reranked">VLM {Math.round(result.rerank_score * 100)}%</span>}{below && <span className="chip below-tag">低于阈值</span>}</div><p>{result.evidence.find(item => item.detail)?.detail || "视觉向量相似度命中"}</p></div>
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
  const [indexConfiguration, setIndexConfiguration] = useState<IndexConfiguration>(defaultIndexConfiguration);
  const [uploading, setUploading] = useState(false);
  const upload = async () => {
    if (!videoFile) return setNotice("请先选择视频");
    setUploading(true);
    try { await api.uploadVideo(videoFile, transcript); setVideoFile(undefined); setTranscript(undefined); await refresh(); setNotice("视频上传完成，可以开始建立索引"); }
    catch (error) { setNotice(error instanceof Error ? error.message : "上传失败"); }
    finally { setUploading(false); }
  };
  const index = async (video: Video) => {
    const validationError = validateIndexConfiguration(indexConfiguration);
    if (validationError) return setNotice(validationError);
    try {
      await api.indexVideo(video.id, indexConfiguration.modalities, toIndexOptions(indexConfiguration));
      await refresh();
      setNotice(`${indexActionLabel(video, indexConfiguration.modalities)}任务已进入队列`);
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
  return <div className="stack-page"><div className="upload-panel panel"><div><span className="panel-label">NEW ASSET</span><h2>添加视频素材</h2><p>上传后按需选择 Visual、Face、ASR 或 OCR 通道。</p></div><label className="file-line"><input type="file" accept="video/*" onChange={event => setVideoFile(event.target.files?.[0])} /><span>{videoFile?.name || "选择视频文件"}</span><b>浏览</b></label><label className="file-line secondary"><input type="file" accept=".json,.srt,.vtt" onChange={event => setTranscript(event.target.files?.[0])} /><span>{transcript?.name || "可选：已有字幕 JSON / SRT / VTT"}</span><b>添加</b></label><button className="primary compact" onClick={upload} disabled={uploading}>{uploading ? "正在上传…" : "上传素材"}</button></div>
    <IndexOptionsPanel value={indexConfiguration} onChange={setIndexConfiguration} />
    <div className="table-panel panel"><div className="section-head"><div><span className="panel-label">LIBRARY</span><h2>视频资产</h2></div><span>{videos.length} items</span></div><div className="asset-list">{videos.map(video => <div className="asset-row" key={video.id}><div className="asset-icon">▶</div><div className="asset-main"><b>{video.name}</b><small>{formatTime(video.duration)} · {video.width}×{video.height} · {video.fps.toFixed(1)} fps</small></div><div className="chips">{video.indexed_modalities.map(mode => <span className={`chip ${mode}`} key={mode}>{mode}</span>)}</div><span className={`status ${video.status}`}>{statusText(video.status)}</span><div className="asset-actions">{video.status !== "indexing" && <button className="outline index-action" disabled={!indexConfiguration.modalities.length} onClick={() => index(video)}>{indexActionLabel(video, indexConfiguration.modalities)}</button>}<button className="outline" onClick={() => rename(video)}>重命名</button><button className="outline danger" disabled={video.status === "indexing"} onClick={() => remove(video)}>删除</button></div></div>)}{!videos.length && <div className="empty-list">还没有视频素材</div>}</div></div>
  </div>;
}

function IndexesPage({ jobs, videos, refresh, setNotice }: { jobs: Job[]; videos: Video[]; refresh: () => Promise<void>; setNotice: (value: string) => void }) {
  const names = Object.fromEntries(videos.map(video => [video.id, video.name]));
  const cancel = async (job: Job) => {
    if (!window.confirm(`确定取消“${names[job.video_id] || job.video_id}”的索引任务吗？`)) return;
    try {
      await api.cancelJob(job.id);
      setNotice("索引任务已取消");
      await refresh();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "取消任务失败");
    }
  };
  return <div className="table-panel panel"><div className="section-head"><div><span className="panel-label">PIPELINE</span><h2>索引任务</h2></div><span>单队列串行执行 · 支持取消</span></div><div className="job-list">{jobs.map(job => {
    const stages = job.metrics?.stages || {};
    return <div className="job-row" key={job.id}><div className={`job-state ${job.status}`}>{job.status === "completed" ? "✓" : job.status === "failed" ? "!" : job.status === "cancelled" ? "×" : "↻"}</div><div className="job-main"><b>{names[job.video_id] || job.video_id}</b><small>{job.modalities.join(" → ")} · 当前：{job.stage} · 总耗时 {formatDuration(job.metrics?.total_elapsed_seconds)}</small><div className="stage-times">{(["visual", "face", "asr", "speaker", "ocr"] as const).filter(stage => stages[stage] || (stage !== "speaker" && job.modalities.includes(stage))).map(stage => <span key={stage}>{stage}: {formatDuration(stages[stage]?.elapsed_seconds)}</span>)}</div><div className="progress"><span style={{ width: `${job.progress * 100}%` }} /></div></div><span className={`status ${job.status}`}>{statusText(job.status)}</span>{["queued", "running"].includes(job.status) && <button className="outline danger" onClick={() => cancel(job)}>取消</button>}{job.error && <p>{job.error}</p>}</div>;
  })}{!jobs.length && <div className="empty-list">还没有索引任务</div>}</div></div>;
}

function SpeakersPage({ videos, setNotice }: { videos: Video[]; setNotice: (value: string) => void }) {
  const indexed = videos.filter(video => video.speaker_indexed);
  const [videoId, setVideoId] = useState("");
  const [view, setView] = useState<SpeakerView>();
  const [hits, setHits] = useState<VoiceHit[]>([]);
  const [voiceFile, setVoiceFile] = useState<File>();
  const [entities, setEntities] = useState<Entity[]>([]);
  const [selectedEntity, setSelectedEntity] = useState<Record<number, string>>({});
  const [loading, setLoading] = useState(false);
  useEffect(() => { if (!videoId && indexed.length) setVideoId(indexed[0].id); }, [indexed.length]);
  const load = async (id = videoId) => {
    if (!id) return;
    setLoading(true);
    try { setView(await api.speakers(id)); }
    catch (error) { setView(undefined); setNotice(error instanceof Error ? error.message : "无法读取 Speaker 索引"); }
    finally { setLoading(false); }
  };
  useEffect(() => { load(videoId); setHits([]); }, [videoId]);
  useEffect(() => { api.entities().then(setEntities).catch(() => undefined); }, []);
  const rename = async (trackId: number, current: string) => {
    const name = window.prompt("Speaker 名称", current); if (name == null) return;
    setView(await api.updateSpeaker(videoId, trackId, { display_name: name.trim() || undefined }));
  };
  const move = async (utteranceIndex: number, trackId: number | null, searchable: boolean) => {
    setView(await api.updateUtterance(videoId, utteranceIndex, { corrected_track_id: trackId, searchable }));
  };
  const search = async (utteranceIndex: number) => {
    setLoading(true); try { setHits((await api.voiceSearch(videoId, utteranceIndex)).results); }
    catch (error) { setNotice(error instanceof Error ? error.message : "声纹搜索失败"); }
    finally { setLoading(false); }
  };
  const searchUpload = async () => {
    if (!voiceFile) return;
    setLoading(true); try { setHits((await api.voiceSearchUpload(voiceFile)).results); }
    catch (error) { setNotice(error instanceof Error ? error.message : "上传声音搜索失败"); }
    finally { setLoading(false); }
  };
  const addToLibrary = async (trackId: number, utteranceIndex: number) => {
    const entityId = selectedEntity[trackId]; if (!entityId) return setNotice("请先选择人物");
    try { await api.addVoiceSample(entityId, videoId, utteranceIndex, trackId); setNotice("代表声音已加入人物库并绑定 Speaker"); await load(); }
    catch (error) { setNotice(error instanceof Error ? error.message : "加入人物库失败"); }
  };
  const utteranceByIndex = Object.fromEntries((view?.utterances || []).map(item => [item.index, item]));
  return <div className="speaker-page">
    <div className="panel speaker-toolbar"><div><span className="panel-label">VOICE WORKSPACE</span><h2>视频内说话人</h2></div><label>视频<select value={videoId} onChange={event => setVideoId(event.target.value)}><option value="">选择已建立 Speaker 索引的视频</option>{indexed.map(video => <option key={video.id} value={video.id}>{video.name}</option>)}</select></label><span>{loading ? "处理中…" : `${view?.tracks.length || 0} speakers`}</span><label className="voice-upload">上传参考声音<input type="file" accept="audio/*,video/*" onChange={event => setVoiceFile(event.target.files?.[0])} /></label><button className="primary compact" disabled={!voiceFile || loading} onClick={searchUpload}>搜索上传声音</button></div>
    {!indexed.length && <div className="panel empty-list">请先为视频选择 ASR + Speaker 通道建立索引</div>}
    <div className="speaker-grid">{view?.tracks.filter(track => !track.hidden).map(track => {
      const representative = utteranceByIndex[track.representative_utterance_index];
      return <article className="panel speaker-card" key={track.track_id}><div className="speaker-card-head"><div><span>Speaker {track.track_id}</span><h3>{track.label}</h3><small>{track.utterance_count} 句 · {formatDuration(track.duration_ms / 1000)}</small></div><button className="outline" onClick={() => rename(track.track_id, track.label)}>改名</button></div>{representative && <><div className="representative"><audio controls preload="none" src={representative.clip_url} /><button className="primary compact" onClick={() => search(representative.index)}>用代表声音搜索</button></div><div className="voice-library-bind"><select value={selectedEntity[track.track_id] || track.entity_id || ""} onChange={event => setSelectedEntity(value => ({ ...value, [track.track_id]: event.target.value }))}><option value="">选择人物库身份</option>{entities.map(entity => <option key={entity.id} value={entity.id}>{entity.name}</option>)}</select><button className="outline" onClick={() => addToLibrary(track.track_id, representative.index)}>加入声音库并绑定</button></div></>}<div className="utterance-list">{track.utterance_indices.map(index => { const utterance = utteranceByIndex[index]; if (!utterance) return null; return <div className="utterance-row" key={index}><div><b>{formatTime(utterance.start_ms / 1000)}–{formatTime(utterance.end_ms / 1000)}</b><p>{utterance.text || "（无文本）"}</p></div><audio controls preload="none" src={utterance.clip_url} /><select value={utterance.track_id ?? -1} onChange={event => move(index, Number(event.target.value), utterance.searchable)}><option value={-1}>未归属</option>{view.tracks.map(option => <option key={option.track_id} value={option.track_id}>{option.label}</option>)}</select><button className="outline" onClick={() => search(index)}>搜索同声</button><label className="searchable-check"><input type="checkbox" checked={utterance.searchable} onChange={event => move(index, utterance.track_id, event.target.checked)} />可检索</label></div>})}</div></article>;
    })}</div>
    {!!hits.length && <div className="panel voice-results"><div className="section-head"><div><span className="panel-label">VOICE MATCHES</span><h2>同声纹片段</h2></div><span>{hits.length} results</span></div>{hits.map(hit => <div className="voice-hit" key={`${hit.video_id}:${hit.utterance_index}`}><b>{hit.video_name}</b><span>{formatTime(hit.start_ms / 1000)} · Speaker {hit.track_id ?? "?"}</span><strong>{(hit.score * 100).toFixed(1)}%</strong><p>{hit.text || "（无文本）"}</p><audio controls preload="none" src={hit.clip_url} /></div>)}</div>}
  </div>;
}

function EntitiesPage({ entities, videos, refresh, setNotice }: { entities: Entity[]; videos: Video[]; refresh: () => Promise<void>; setNotice: (value: string) => void }) {
  const [name, setName] = useState(""); const [image, setImage] = useState<File>(); const [saving, setSaving] = useState(false);
  const save = async () => { if (!name.trim()) return setNotice("请输入人物名称"); setSaving(true); try { image ? await api.createEntity(name.trim(), image) : await api.createVoiceEntity(name.trim()); setName(""); setImage(undefined); await refresh(); setNotice(image ? "人物与参考脸已登记" : "已创建声音人物，可从 Speaker 页面添加声音"); } catch (error) { setNotice(error instanceof Error ? error.message : "人物登记失败"); } finally { setSaving(false); } };
  const rename = async (entity: Entity) => { const next = window.prompt("人物名称", entity.name); if (!next?.trim() || next.trim() === entity.name) return; try { await api.renameEntity(entity.id, next.trim()); await refresh(); setNotice("人物已重命名"); } catch (error) { setNotice(error instanceof Error ? error.message : "重命名失败"); } };
  const remove = async (entity: Entity) => { if (!window.confirm(`删除人物“${entity.name}”？相关人脸、声音样本和绑定也会删除。`)) return; try { await api.deleteEntity(entity.id); await refresh(); setNotice("人物已删除"); } catch (error) { setNotice(error instanceof Error ? error.message : "删除失败"); } };
  return <div><div className="entity-layout"><div className="entity-create panel"><span className="panel-label">NEW IDENTITY</span><h2>登记人物</h2><p>参考脸可选；没有图片时可先创建声音人物，再从下方视频说话人添加代表声音。</p><input className="text-input" value={name} onChange={event => setName(event.target.value)} placeholder="人物或明星名称" /><label className="portrait-drop"><input type="file" accept="image/*" onChange={event => setImage(event.target.files?.[0])} />{image ? <img src={URL.createObjectURL(image)} /> : <><span>◎</span><b>可选：添加清晰正脸</b></>}</label><button className="primary compact" disabled={saving} onClick={save}>{saving ? "正在登记…" : image ? "登记人脸人物" : "创建声音人物"}</button></div><div className="entity-library"><div className="section-head"><div><span className="panel-label">IDENTITY LIBRARY</span><h2>人物库</h2></div><span>{entities.length} entities</span></div><div className="entity-grid">{entities.map(entity => <article key={entity.id}>{entity.reference_path ? <img src={`/api/entities/${entity.id}/reference`} /> : <div className="voice-only-avatar">◉</div>}<div><b>{entity.name}</b><small>{entity.embedding_path ? "人脸" : "无参考脸"} · {entity.voice_sample_count || 0} 条声音</small><div className="asset-actions"><button className="outline" onClick={() => rename(entity)}>重命名</button><button className="outline danger" onClick={() => remove(entity)}>删除</button></div></div></article>)}{!entities.length && <div className="empty-list">还没有登记人物</div>}</div></div></div><SpeakersPage videos={videos} setNotice={setNotice} /></div>;
}

function Overview({ videos, jobs, entities, setPage }: { videos: Video[]; jobs: Job[]; entities: Entity[]; setPage: (page: Page) => void }) {
  const ready = videos.filter(video => video.status === "ready").length;
  return <div className="overview"><div className="hero panel"><div><span className="panel-label">MVP BASELINE</span><h2>让长视频变成<br />可以搜索的素材库。</h2><p>Face / Visual / ASR / OCR 四路独立索引，保留时间证据；模型只在索引阶段短暂加载。</p><button className="primary compact" onClick={() => setPage("assets")}>添加第一条视频 <span>→</span></button></div><div className="hero-visual"><div className="orbit one">Face</div><div className="orbit two">Visual</div><div className="orbit three">ASR</div><span className="core">M</span></div></div><div className="stats"><article><span>视频资产</span><b>{videos.length}</b><small>{ready} 条可检索</small></article><article><span>人物实体</span><b>{entities.length}</b><small>参考脸向量</small></article><article><span>索引任务</span><b>{jobs.length}</b><small>{jobs.filter(job => job.status === "running").length} 个正在运行</small></article><article><span>NPU 常驻</span><b>0</b><small>任务结束即释放</small></article></div></div>;
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);
