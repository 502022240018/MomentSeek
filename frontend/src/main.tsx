import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, Entity, Job, SearchResult, Video } from "./api";
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
  const [modalities, setModalities] = useState(["visual", "face", "asr"]);
  const [alpha, setAlpha] = useState(0.5);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
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
      if (!response.results.length) setNotice("没有超过当前阈值的片段，可以换个描述或通道再试");
    } catch (error) { setNotice(error instanceof Error ? error.message : "检索失败"); }
    finally { setLoading(false); }
  };

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
        {[['visual','Visual','场景与物体'],['face','Face','同一人物'],['asr','ASR','语音内容']].map(([value,title,sub]) =>
          <button key={value} className={modalities.includes(value) ? "selected" : ""} onClick={() => toggleMode(value)}>
            <span>{modalities.includes(value) ? "✓" : "+"}</span><b>{title}</b><small>{sub}</small>
          </button>)}
      </div>
      {query && image && modalities.includes("visual") && <div className="alpha-control"><span>文字权重 {Math.round(alpha * 100)}%</span><input type="range" min="0" max="1" step="0.05" value={alpha} onChange={event => setAlpha(Number(event.target.value))} /></div>}
      <button className="primary" disabled={loading} onClick={submit}>{loading ? <><span className="spinner" />正在检索</> : <>开始检索 <span>→</span></>}</button>
    </div>

    <div className="results-panel">
      <div className="results-head"><div><span className="panel-label">MOMENTS</span><h2>{results.length ? `${results.length} 个相关片段` : "在视频中找到那个瞬间"}</h2></div>{results.length > 0 && <button className="text-button" onClick={() => setResults([])}>清空结果</button>}</div>
      {!results.length ? <div className="examples">
        <Example title="参考图中的人物" tags={["FACE", "VISUAL"]} text="上传一张清晰正脸，找出人物所有出现片段" setQuery={setQuery} />
        <Example title="舞台上讲话" tags={["VISUAL"]} text="a person speaking on a stage" setQuery={setQuery} />
        <Example title="语音中提到某话题" tags={["ASR", "LEXICAL"]} text="电影投资" setQuery={setQuery} />
        <Example title="红色行李箱" tags={["VISUAL", "IMAGE"]} text="a red suitcase" setQuery={setQuery} />
      </div> : <div className="result-grid">{results.map((result, index) => <ResultCard key={`${result.video_id}-${result.start_time}-${index}`} result={result} onPlay={() => setPlaying(result)} />)}</div>}
    </div>
    {playing && <PlayerModal result={playing} onClose={() => setPlaying(undefined)} />}
  </div>;
}

function Example({ title, tags, text, setQuery }: { title: string; tags: string[]; text: string; setQuery: (value: string) => void }) {
  return <button className="example-card" onClick={() => setQuery(text)}><span className="spark">✦</span><h3>{title}</h3><p>{text}</p><div>{tags.map(tag => <span key={tag}>{tag}</span>)}</div></button>;
}

function ResultCard({ result, onPlay }: { result: SearchResult; onPlay: () => void }) {
  return <article className="result-card" onClick={onPlay}>
    <div className="result-thumb">{result.thumbnail_url ? <img src={result.thumbnail_url} /> : <div className="thumb-placeholder">M</div>}<button>▶</button><span>{formatTime(result.start_time)} — {formatTime(result.end_time)}</span></div>
    <div className="result-body"><div className="result-title"><h3>{result.video_name}</h3><b>{Math.round(result.score * 100)}%</b></div><div className="chips">{result.modalities.map(mode => <span className={`chip ${mode}`} key={mode}>{mode}</span>)}</div><p>{result.evidence.find(item => item.detail)?.detail || "视觉向量相似度命中"}</p></div>
  </article>;
}

function PlayerModal({ result, onClose }: { result: SearchResult; onClose: () => void }) {
  const ref = useRef<HTMLVideoElement>(null);
  return <div className="modal-backdrop" onMouseDown={onClose}><div className="player-modal" onMouseDown={event => event.stopPropagation()}><button className="close" onClick={onClose}>×</button><video ref={ref} src={result.media_url} controls autoPlay onLoadedMetadata={() => { if (ref.current) ref.current.currentTime = result.start_time; }} /><div className="player-info"><div><span className="panel-label">MATCHED MOMENT</span><h3>{result.video_name}</h3></div><b>{formatTime(result.start_time)} — {formatTime(result.end_time)}</b></div></div></div>;
}

function AssetsPage({ videos, refresh, setNotice }: { videos: Video[]; refresh: () => Promise<void>; setNotice: (value: string) => void }) {
  const [videoFile, setVideoFile] = useState<File>();
  const [transcript, setTranscript] = useState<File>();
  const [uploading, setUploading] = useState(false);
  const upload = async () => {
    if (!videoFile) return setNotice("请先选择视频");
    setUploading(true);
    try { await api.uploadVideo(videoFile, transcript); setVideoFile(undefined); setTranscript(undefined); await refresh(); setNotice("视频上传完成，可以开始建立索引"); }
    catch (error) { setNotice(error instanceof Error ? error.message : "上传失败"); }
    finally { setUploading(false); }
  };
  const index = async (id: string) => {
    try { await api.indexVideo(id, ["visual", "face", "asr"]); await refresh(); setNotice("索引任务已进入队列，模型会按阶段加载并自动释放"); }
    catch (error) { setNotice(error instanceof Error ? error.message : "任务创建失败"); }
  };
  return <div className="stack-page"><div className="upload-panel panel"><div><span className="panel-label">NEW ASSET</span><h2>添加视频素材</h2><p>上传后可一次建立 Face、Visual 和 ASR 三路索引。</p></div><label className="file-line"><input type="file" accept="video/*" onChange={event => setVideoFile(event.target.files?.[0])} /><span>{videoFile?.name || "选择视频文件"}</span><b>浏览</b></label><label className="file-line secondary"><input type="file" accept=".json,.srt,.vtt" onChange={event => setTranscript(event.target.files?.[0])} /><span>{transcript?.name || "可选：已有字幕 JSON / SRT / VTT"}</span><b>添加</b></label><button className="primary compact" onClick={upload} disabled={uploading}>{uploading ? "正在上传…" : "上传素材"}</button></div>
    <div className="table-panel panel"><div className="section-head"><div><span className="panel-label">LIBRARY</span><h2>视频资产</h2></div><span>{videos.length} items</span></div><div className="asset-list">{videos.map(video => <div className="asset-row" key={video.id}><div className="asset-icon">▶</div><div className="asset-main"><b>{video.name}</b><small>{formatTime(video.duration)} · {video.width}×{video.height} · {video.fps.toFixed(1)} fps</small></div><div className="chips">{video.indexed_modalities.map(mode => <span className={`chip ${mode}`} key={mode}>{mode}</span>)}</div><span className={`status ${video.status}`}>{statusText(video.status)}</span>{!(["indexing"].includes(video.status)) && <button className="outline" onClick={() => index(video.id)}>{video.indexed_modalities.length ? "重建索引" : "建立索引"}</button>}</div>)}{!videos.length && <div className="empty-list">还没有视频素材</div>}</div></div>
  </div>;
}

function IndexesPage({ jobs, videos }: { jobs: Job[]; videos: Video[] }) {
  const names = Object.fromEntries(videos.map(video => [video.id, video.name]));
  return <div className="table-panel panel"><div className="section-head"><div><span className="panel-label">PIPELINE</span><h2>索引任务</h2></div><span>阶段子进程退出后释放模型</span></div><div className="job-list">{jobs.map(job => <div className="job-row" key={job.id}><div className={`job-state ${job.status}`}>{job.status === "completed" ? "✓" : job.status === "failed" ? "!" : "↻"}</div><div className="job-main"><b>{names[job.video_id] || job.video_id}</b><small>{job.modalities.join(" → ")} · 当前：{job.stage}</small><div className="progress"><span style={{ width: `${job.progress * 100}%` }} /></div></div><span className={`status ${job.status}`}>{statusText(job.status)}</span>{job.error && <p>{job.error}</p>}</div>)}{!jobs.length && <div className="empty-list">还没有索引任务</div>}</div></div>;
}

function EntitiesPage({ entities, refresh, setNotice }: { entities: Entity[]; refresh: () => Promise<void>; setNotice: (value: string) => void }) {
  const [name, setName] = useState(""); const [image, setImage] = useState<File>(); const [saving, setSaving] = useState(false);
  const save = async () => { if (!name.trim() || !image) return setNotice("请输入人物名称并选择清晰正脸"); setSaving(true); try { await api.createEntity(name.trim(), image); setName(""); setImage(undefined); await refresh(); setNotice("人物已登记，以后可以直接用名字检索"); } catch (error) { setNotice(error instanceof Error ? error.message : "人物登记失败"); } finally { setSaving(false); } };
  return <div className="entity-layout"><div className="entity-create panel"><span className="panel-label">NEW ENTITY</span><h2>登记人物</h2><p>人物名称与参考脸绑定后，可用“内马尔”等名称调用 face_index。</p><input className="text-input" value={name} onChange={event => setName(event.target.value)} placeholder="人物或明星名称" /><label className="portrait-drop"><input type="file" accept="image/*" onChange={event => setImage(event.target.files?.[0])} />{image ? <img src={URL.createObjectURL(image)} /> : <><span>◎</span><b>选择清晰正脸</b></>}</label><button className="primary compact" disabled={saving} onClick={save}>{saving ? "正在提取人脸…" : "登记人物"}</button></div><div className="entity-library"><div className="section-head"><div><span className="panel-label">FACE LIBRARY</span><h2>人物库</h2></div><span>{entities.length} entities</span></div><div className="entity-grid">{entities.map(entity => <article key={entity.id}><img src={`/api/entities/${entity.id}/reference`} /><div><b>{entity.name}</b><small>{entity.embedding_path ? "Face embedding ready" : "Processing"}</small></div></article>)}{!entities.length && <div className="empty-list">还没有登记人物</div>}</div></div></div>;
}

function Overview({ videos, jobs, entities, setPage }: { videos: Video[]; jobs: Job[]; entities: Entity[]; setPage: (page: Page) => void }) {
  const ready = videos.filter(video => video.status === "ready").length;
  return <div className="overview"><div className="hero panel"><div><span className="panel-label">MVP BASELINE</span><h2>让长视频变成<br />可以搜索的素材库。</h2><p>三路独立索引，保留时间证据；模型只在索引阶段短暂加载。</p><button className="primary compact" onClick={() => setPage("assets")}>添加第一条视频 <span>→</span></button></div><div className="hero-visual"><div className="orbit one">Face</div><div className="orbit two">Visual</div><div className="orbit three">ASR</div><span className="core">M</span></div></div><div className="stats"><article><span>视频资产</span><b>{videos.length}</b><small>{ready} 条可检索</small></article><article><span>人物实体</span><b>{entities.length}</b><small>参考脸向量</small></article><article><span>索引任务</span><b>{jobs.length}</b><small>{jobs.filter(job => job.status === "running").length} 个正在运行</small></article><article><span>NPU 常驻</span><b>0</b><small>任务结束即释放</small></article></div></div>;
}

createRoot(document.getElementById("root")!).render(<React.StrictMode><App /></React.StrictMode>);

