import type { IndexModality, IndexOptions, Video } from "./api";

export type VisualSegmentPresetKey = "fixed" | "general" | "fast-cut" | "talking-head" | "custom";
export type VisualShotDetector = "simple" | "pyscenedetect_content" | "pyscenedetect_adaptive";

type VisualSegmentPreset = {
  label: string;
  strategy: "fixed" | "shot";
  detector: VisualShotDetector;
  minSeconds: number;
  maxSeconds: number;
  threshold: number;
};

export type IndexConfiguration = {
  modalities: IndexModality[];
  visualModel: string;
  visualSampleFps: number;
  visualSegmentSeconds: number;
  visualSegmentPreset: VisualSegmentPresetKey;
  visualMinSegmentSeconds: number;
  visualMaxSegmentSeconds: number;
  visualShotDetector: VisualShotDetector;
  visualShotThreshold: number;
  faceSampleFps: number;
  ocrSampleFps: number;
  asrModel: string;
  asrLanguage: string;
  asrSpeakerEnabled: boolean;
};

const channelDefinitions: Array<{ id: IndexModality; label: string; detail: string }> = [
  { id: "visual", label: "Visual", detail: "画面语义" },
  { id: "face", label: "Face", detail: "人脸轨迹" },
  { id: "asr", label: "ASR", detail: "语音文本" },
  { id: "ocr", label: "OCR", detail: "画面文字" },
];

const visualShotDetectorLabels: Record<VisualShotDetector, string> = {
  simple: "Simple 帧差",
  pyscenedetect_content: "PySceneDetect Content",
  pyscenedetect_adaptive: "PySceneDetect Adaptive",
};

const visualSegmentPresets: Record<VisualSegmentPresetKey, VisualSegmentPreset> = {
  fixed: { label: "固定分段", strategy: "fixed", detector: "simple", minSeconds: 0.8, maxSeconds: 8, threshold: 0.2 },
  general: { label: "通用镜头", strategy: "shot", detector: "simple", minSeconds: 0.8, maxSeconds: 8, threshold: 0.2 },
  "fast-cut": { label: "广告 / MV", strategy: "shot", detector: "pyscenedetect_adaptive", minSeconds: 0.5, maxSeconds: 6, threshold: 0.16 },
  "talking-head": { label: "访谈长镜头", strategy: "shot", detector: "pyscenedetect_content", minSeconds: 1.5, maxSeconds: 12, threshold: 0.28 },
  custom: { label: "自定义", strategy: "shot", detector: "simple", minSeconds: 0.8, maxSeconds: 8, threshold: 0.2 },
};

export const defaultIndexConfiguration: IndexConfiguration = {
  modalities: ["visual", "face", "asr"],
  visualModel: "siglip2-so400m-384",
  visualSampleFps: 5,
  visualSegmentSeconds: 5,
  visualSegmentPreset: "fixed",
  visualMinSegmentSeconds: 0.8,
  visualMaxSegmentSeconds: 8,
  visualShotDetector: "simple",
  visualShotThreshold: 0.2,
  faceSampleFps: 2,
  ocrSampleFps: 1,
  asrModel: "turbo",
  asrLanguage: "auto",
  asrSpeakerEnabled: true,
};

function channelNames(modalities: IndexModality[]) {
  return channelDefinitions
    .filter(channel => modalities.includes(channel.id))
    .map(channel => channel.label)
    .join("+");
}

export function indexActionLabel(video: Video, modalities: IndexModality[]) {
  if (!modalities.length) return "选择索引通道";
  const existing = new Set(video.indexed_modalities);
  const build = modalities.filter(modality => !existing.has(modality));
  const rebuild = modalities.filter(modality => existing.has(modality));
  return [
    build.length ? `构建 ${channelNames(build)}` : "",
    rebuild.length ? `重建 ${channelNames(rebuild)}` : "",
  ].filter(Boolean).join(" · ");
}

export function toIndexOptions(configuration: IndexConfiguration): IndexOptions {
  const useShotSegments = visualSegmentPresets[configuration.visualSegmentPreset].strategy === "shot";
  return {
    visualModel: configuration.visualModel,
    visualSampleFps: configuration.visualSampleFps,
    visualSegmentSeconds: configuration.visualSegmentSeconds,
    visualSegmentStrategy: useShotSegments ? "shot" : "fixed",
    visualMinSegmentSeconds: useShotSegments ? configuration.visualMinSegmentSeconds : undefined,
    visualMaxSegmentSeconds: useShotSegments ? configuration.visualMaxSegmentSeconds : undefined,
    visualShotDetector: useShotSegments ? configuration.visualShotDetector : undefined,
    visualShotThreshold: useShotSegments ? configuration.visualShotThreshold : undefined,
    faceSampleFps: configuration.faceSampleFps,
    ocrSampleFps: configuration.ocrSampleFps,
    asrModel: configuration.asrModel,
    asrLanguage: configuration.asrLanguage,
    asrSpeakerEnabled: configuration.asrSpeakerEnabled,
  };
}

export function validateIndexConfiguration(configuration: IndexConfiguration): string | null {
  if (!configuration.modalities.length) return "请至少选择一条索引通道";
  const useShotSegments = visualSegmentPresets[configuration.visualSegmentPreset].strategy === "shot";
  if (
    configuration.modalities.includes("visual")
    && useShotSegments
    && configuration.visualMinSegmentSeconds > configuration.visualMaxSegmentSeconds
  ) {
    return "镜头最短时长不能大于最长时长";
  }
  return null;
}

export function IndexOptionsPanel({
  value,
  onChange,
}: {
  value: IndexConfiguration;
  onChange: (value: IndexConfiguration) => void;
}) {
  const useShotSegments = visualSegmentPresets[value.visualSegmentPreset].strategy === "shot";
  const selected = new Set(value.modalities);
  const update = <K extends keyof IndexConfiguration>(key: K, next: IndexConfiguration[K]) => {
    onChange({ ...value, [key]: next });
  };
  const toggleModality = (modality: IndexModality) => {
    const next = new Set(value.modalities);
    if (next.has(modality)) next.delete(modality);
    else next.add(modality);
    update("modalities", channelDefinitions.map(channel => channel.id).filter(channel => next.has(channel)));
  };
  const applyVisualSegmentPreset = (presetKey: VisualSegmentPresetKey) => {
    if (presetKey === "custom") {
      onChange({ ...value, visualSegmentPreset: presetKey });
      return;
    }
    const preset = visualSegmentPresets[presetKey];
    onChange({
      ...value,
      visualSegmentPreset: presetKey,
      visualMinSegmentSeconds: preset.minSeconds,
      visualMaxSegmentSeconds: preset.maxSeconds,
      visualShotDetector: preset.detector,
      visualShotThreshold: preset.threshold,
    });
  };

  return <div className="index-options panel">
    <div className="index-options-head">
      <div>
        <span className="panel-label">INDEX CHANNELS</span>
        <h2>索引通道与参数</h2>
      </div>
      <span className="selection-count">已选 {value.modalities.length} / 4</span>
    </div>
    <div className="index-channel-picker" role="group" aria-label="选择索引通道">
      {channelDefinitions.map(channel => <label className={`channel-toggle ${selected.has(channel.id) ? "selected" : ""}`} key={channel.id}>
        <input type="checkbox" checked={selected.has(channel.id)} onChange={() => toggleModality(channel.id)} />
        <span><b>{channel.label}</b><small>{channel.detail}</small></span>
      </label>)}
    </div>

    {selected.has("visual") && <div className="channel-settings">
      <div className="channel-settings-title"><span className="chip visual">visual</span><b>画面语义</b></div>
      <label>模型<select value={value.visualModel} onChange={event => update("visualModel", event.target.value)}><option value="siglip2-so400m-384">SigLIP2 So400m-384 默认</option><option value="chinese-clip-vit-b16">ChineseCLIP ViT-B/16</option><option value="openclip-vit-b32">OpenCLIP ViT-B/32</option><option value="openclip-vit-b16">OpenCLIP ViT-B/16</option><option value="openclip-vit-l14">OpenCLIP ViT-L/14</option></select></label>
      <label>采样 fps<input type="number" min="0.2" max="10" step="0.5" value={value.visualSampleFps} onChange={event => update("visualSampleFps", Number(event.target.value))} /></label>
      <label>分段方式<select value={value.visualSegmentPreset} onChange={event => applyVisualSegmentPreset(event.target.value as VisualSegmentPresetKey)}>{(Object.keys(visualSegmentPresets) as VisualSegmentPresetKey[]).map(key => <option key={key} value={key}>{visualSegmentPresets[key].label}</option>)}</select></label>
      <label>{useShotSegments ? "固定回退秒数" : "分段秒数"}<input type="number" min="1" max="60" step="1" value={value.visualSegmentSeconds} onChange={event => update("visualSegmentSeconds", Number(event.target.value))} /></label>
      {useShotSegments && <>
        <label>镜头检测器<select value={value.visualShotDetector} onChange={event => update("visualShotDetector", event.target.value as VisualShotDetector)}>{(Object.keys(visualShotDetectorLabels) as VisualShotDetector[]).map(key => <option key={key} value={key}>{visualShotDetectorLabels[key]}</option>)}</select></label>
        <label>最短秒数<input type="number" min="0.2" max="30" step="0.1" value={value.visualMinSegmentSeconds} onChange={event => update("visualMinSegmentSeconds", Number(event.target.value))} /></label>
        <label>最长秒数<input type="number" min="1" max="60" step="0.5" value={value.visualMaxSegmentSeconds} onChange={event => update("visualMaxSegmentSeconds", Number(event.target.value))} /></label>
        <label>切分阈值<input type="number" min="0.05" max="0.6" step="0.01" value={value.visualShotThreshold} onChange={event => update("visualShotThreshold", Number(event.target.value))} /></label>
      </>}
    </div>}

    {selected.has("face") && <div className="channel-settings compact-settings">
      <div className="channel-settings-title"><span className="chip face">face</span><b>人脸轨迹</b></div>
      <label>采样 fps<input type="number" min="0.2" max="15" step="0.5" value={value.faceSampleFps} onChange={event => update("faceSampleFps", Number(event.target.value))} /></label>
    </div>}

    {selected.has("asr") && <div className="channel-settings compact-settings">
      <div className="channel-settings-title"><span className="chip asr">asr</span><b>语音文本</b></div>
      <label>模型<select value={value.asrModel} onChange={event => update("asrModel", event.target.value)}><option value="turbo">turbo 默认</option><option value="small">small 更快</option><option value="medium">medium 更准更慢</option><option value="large-v3">large-v3 高风险</option></select></label>
      <label>语言<select value={value.asrLanguage} onChange={event => update("asrLanguage", event.target.value)}><option value="auto">Auto</option><option value="zh">中文</option><option value="yue">粤语/方言</option><option value="en">English</option><option value="es">Español</option><option value="pt">Português</option></select></label>
      <label className="inline-option"><span>Speaker 声纹</span><input type="checkbox" checked={value.asrSpeakerEnabled} onChange={event => update("asrSpeakerEnabled", event.target.checked)} /></label>
    </div>}

    {selected.has("ocr") && <div className="channel-settings compact-settings">
      <div className="channel-settings-title"><span className="chip ocr">ocr</span><b>画面文字</b></div>
      <label>采样 fps<input type="number" min="0.05" max="5" step="0.05" value={value.ocrSampleFps} onChange={event => update("ocrSampleFps", Number(event.target.value))} /></label>
    </div>}
  </div>;
}
