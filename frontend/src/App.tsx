import React, { useState, useEffect, useRef, useCallback } from "react";
import axios from "axios";
import {
  Camera, Settings, Activity, BoxSelect, Cpu, ScanSearch, CheckCircle2,
  Video, XCircle, Globe, X, Image as ImageIcon, Server, ShieldAlert,
  Database, RefreshCw, Save, ChevronRight, ChevronLeft,
  Lock
} from "lucide-react";
import { useAppStore } from "./store";
import type { ImageLog, CameraData, VideoFrameResult, ImageInspection } from "./store";
import { locales } from "./locales";

export type StepStatus = 'idle' | 'processing' | 'success' | 'error';
export type ViewItem = { type: 'camera'; id: string } | { type: 'image'; id: string };
type StreamModelMode = 'real' | 'mock';
type StreamConnectionState = 'idle' | 'connecting' | 'streaming' | 'error';

const DASHBOARD_WS_URL = import.meta.env.VITE_WS_URL || (
  typeof window !== 'undefined'
    ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`
    : 'ws://localhost:8080/ws'
);

function buildSourceWsUrl(cameraId: string, modelMode: StreamModelMode, scenario: string) {
  const base = DASHBOARD_WS_URL.replace(/\/$/, '');
  const sourceBase = base.endsWith('/ws') ? `${base}/source` : `${base}/ws/source`;
  const url = new URL(sourceBase);
  url.searchParams.set('cameraId', cameraId);
  url.searchParams.set('model_mode', modelMode);
  if (scenario.trim()) url.searchParams.set('scenario', scenario.trim());
  return url.toString();
}

function parseTargetOrder(value: string) {
  return value.split(',').map(item => item.trim()).filter(Boolean);
}

// ─── UI 생동감 표시기 ─────────────────────────────────────────────────────────
function LivenessIndicator() {
  const speeds = [0.7, 0.45, 0.9, 0.55, 1.0, 0.4, 0.75];
  return (
    <div className="flex items-end gap-[3px] h-5" title="UI 정상 동작 중">
      {speeds.map((speed, i) => (
        <div
          key={i}
          className="w-[3px] bg-[#22c55e]"
          style={{
            height: '3px',
            animation: `liveness-bar ${speed}s ease-in-out infinite alternate`,
            animationDelay: `${i * 0.08}s`,
          }}
        />
      ))}
    </div>
  );
}

// ─── 이미지 패널 ──────────────────────────────────────────────────────────────
function ImagePanel({
  img, onRemove, isSplit, onOpenDetail,
}: {
  img: ImageLog; onRemove?: () => void; isSplit?: boolean; onOpenDetail: () => void;
}) {
  const { language } = useAppStore();
  const t = locales[language];
  const sc = img.status === 'success' ? '#22c55e' : img.status === 'error' ? '#ef4444' : '#f59e0b';
  return (
    <section className="flex flex-col bg-[#27272a] border border-[#3f3f46] shadow-lg h-full overflow-hidden relative">
      <div className={`flex justify-between items-center bg-[#3f3f46] px-2 border-b border-[#52525b] ${isSplit ? 'py-1' : 'py-2'}`}>
        <div className="flex items-center gap-2">
          <ImageIcon size={14} className="text-[#E50012]" />
          <h2 className="font-semibold tracking-wide text-zinc-100 text-xs">{img.id}</h2>
        </div>
        <div className="flex items-center gap-2 font-mono text-zinc-400 text-[10px]">
          <span className="px-1.5 py-0.5 border border-zinc-600">T{img.target_idx}</span>
          <span>{img.cam}</span>
          {isSplit && onRemove && (
            <button onClick={(e) => { e.stopPropagation(); onRemove(); }} className="ml-1 text-zinc-500 hover:text-red-400 px-1">✕</button>
          )}
        </div>
      </div>
      <div
        className="flex-1 relative flex items-center justify-center overflow-hidden bg-black cursor-pointer select-none"
        onDoubleClick={onOpenDetail}
      >
        <div className="relative flex items-center justify-center w-full max-h-full" style={{ aspectRatio: '16/9', maxHeight: '100%', maxWidth: '100%' }}>
          {img.image_url ? (
            <img src={img.image_url} alt={img.id} className="w-full h-full object-contain" />
          ) : (
            <div className="absolute inset-0 flex flex-col items-center justify-center opacity-40">
              <ImageIcon size={isSplit ? 28 : 48} className="text-zinc-500 mb-2" />
              <span className="font-mono text-zinc-500 text-sm">NO IMAGE</span>
            </div>
          )}
        </div>
        <div className="absolute top-2 left-2 bg-[#18181b]/90 px-2 py-0.5 border border-[#3f3f46] font-mono text-zinc-300 z-10 text-[10px]">{img.cam}</div>
        <div className="absolute top-2 right-2 bg-[#18181b]/90 px-2 py-0.5 border font-mono z-10 text-[10px]" style={{ borderColor: sc, color: sc }}>T{img.target_idx} · {img.reason}</div>
        <div className="absolute bottom-3 inset-x-0 flex justify-center z-10 pointer-events-none">
          <div className="px-4 py-1 border text-sm font-black uppercase tracking-widest bg-[#18181b]/90" style={{ color: sc, borderColor: sc }}>
            {img.status === 'success' ? t.statusSuccess : img.status === 'error' ? t.statusError : t.statusProcessing}
          </div>
        </div>
        <div className="absolute bottom-1 right-2 text-[8px] text-zinc-800 font-mono z-10 pointer-events-none">더블클릭 • 상세보기</div>
      </div>
      <div className={`bg-[#27272a] border-t border-[#3f3f46] flex ${isSplit ? 'p-0.5 gap-0.5' : 'p-1.5 gap-1.5'}`}>
        {[1, 2, 3, 4].map((s) => {
          const isTarget = s === img.target_idx;
          const style = isTarget
            ? img.status === 'success' ? 'border-[#22c55e] bg-[#22c55e]/15 text-[#22c55e]'
            : img.status === 'error' ? 'border-[#ef4444] bg-[#ef4444]/15 text-[#ef4444]'
            : 'border-[#f59e0b] bg-[#f59e0b]/15 text-[#f59e0b] animate-pulse'
            : 'border-[#52525b] bg-zinc-800/30 text-zinc-600 opacity-50';
          return (
            <div key={s} className={`flex-1 border flex items-center justify-between relative transition-all px-1.5 py-1 ${style}`}>
              <span className="text-[9px] font-bold tracking-wider">T{s}</span>
              {isTarget && (img.status === 'success' ? <CheckCircle2 size={10} /> : img.status === 'error' ? <XCircle size={10} /> : <Activity size={10} />)}
            </div>
          );
        })}
      </div>
    </section>
  );
}

// ─── 카메라 패널 ──────────────────────────────────────────────────────────────
function CameraPanel({
  cameraId, title, stepStatuses, latestData, onRemove, isSplit, onOpenDetail,
}: {
  cameraId: string; title: string; stepStatuses: StepStatus[];
  latestData: CameraData | null; onRemove?: () => void; isSplit?: boolean; onOpenDetail: () => void;
}) {
  const { language } = useAppStore();
  const t = locales[language];
  const steps = [
    { id: 1, title: 'T1', icon: BoxSelect },
    { id: 2, title: 'T2', icon: Cpu },
    { id: 3, title: 'T3', icon: ScanSearch },
    { id: 4, title: 'T4', icon: CheckCircle2 },
  ];
  return (
    <section className="flex flex-col bg-[#27272a] border border-[#3f3f46] shadow-lg h-full overflow-hidden relative">
      <div className={`flex justify-between items-center bg-[#3f3f46] px-2 border-b border-[#52525b] ${isSplit ? 'py-1' : 'py-2'}`}>
        <div className="flex items-center gap-2">
          <Camera size={14} className="text-[#E50012]" />
          <h2 className="font-semibold tracking-wide text-zinc-100 text-xs">{cameraId}</h2>
          {!isSplit && <span className="text-zinc-500 text-[10px] font-mono truncate max-w-[160px]">{title}</span>}
        </div>
        <div className="flex items-center gap-2 font-mono text-zinc-400 text-[10px]">
          {!isSplit && <span>60fps</span>}
          {isSplit && onRemove && (
            <button onClick={(e) => { e.stopPropagation(); onRemove(); }} className="text-zinc-500 hover:text-red-400 px-1">✕</button>
          )}
        </div>
      </div>
      <div
        className="flex-1 relative flex items-center justify-center overflow-hidden bg-black cursor-pointer select-none"
        onDoubleClick={onOpenDetail}
      >
        <div className="relative bg-zinc-900/40 flex items-center justify-center w-full max-h-full" style={{ aspectRatio: '16/9', maxHeight: '100%', maxWidth: '100%' }}>
          {cameraId === 'CAM_00' ? (
            <iframe 
              src={(latestData?.display as any)?.url || "https://www.youtube.com/embed/dQw4w9WgXcQ?autoplay=1&mute=1"}
              className="absolute inset-0 w-full h-full border-none z-0"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
            />
          ) : (
            <>
              <img id={`video-stream-${cameraId}`} className="absolute inset-0 w-full h-full object-contain hidden opacity-100 z-0" alt={cameraId} />
              {!latestData?.inference && (
                <div className="absolute inset-0 flex flex-col items-center justify-center opacity-30 z-10">
                  <Activity size={isSplit ? 28 : 48} className="text-zinc-500 mb-2" />
                  <span className={`font-mono text-zinc-500 ${isSplit ? 'text-xs' : 'text-base'}`}>{t.videoOffline}</span>
                </div>
              )}
            </>
          )}
        </div>
        <div className="absolute top-2 left-2 bg-[#18181b]/90 px-2 py-0.5 border border-[#3f3f46] font-mono text-zinc-300 z-10 text-[10px]">{cameraId}</div>
        {latestData?.inference && (
          <div className="absolute top-2 right-2 bg-[#18181b]/90 px-2 py-0.5 border border-[#f59e0b] text-[#f59e0b] font-mono flex items-center gap-1.5 z-10 text-[10px] animate-pulse">
            <div className="bg-[#f59e0b] rounded-full w-1.5 h-1.5" />
            INFERENCE
          </div>
        )}
        {latestData?.inference && (
          <div className="absolute bottom-3 inset-x-0 w-full flex justify-center z-10 pointer-events-none">
            <div className={`px-4 py-1 border text-sm font-black uppercase tracking-widest bg-[#18181b]/90 ${latestData.logic && (latestData.logic as Record<string, unknown>).allowed_transition === false ? 'text-[#ef4444] border-[#ef4444]' : 'text-[#22c55e] border-[#3f3f46]'}`}>
              {(latestData.display as Record<string, string> | undefined)?.system_message || 'ANALYZING...'}
            </div>
          </div>
        )}
        <div className="absolute bottom-1 right-2 text-[8px] text-zinc-800 font-mono z-10 pointer-events-none">더블클릭 • 상세보기</div>
      </div>
      <div className={`bg-[#27272a] border-t border-[#3f3f46] flex ${isSplit ? 'p-0.5 gap-0.5' : 'p-1.5 gap-1.5'}`}>
        {steps.map((step, index) => {
          const status = stepStatuses[index] || 'idle';
          let ss = 'border-[#52525b] bg-zinc-800/30 text-zinc-600 opacity-50';
          let tc = 'text-zinc-600';
          if (status === 'processing') { ss = 'border-[#f59e0b] bg-[#f59e0b]/15 animate-pulse'; tc = 'text-[#f59e0b]'; }
          else if (status === 'success') { ss = 'border-[#22c55e] bg-[#22c55e]/15'; tc = 'text-[#22c55e]'; }
          else if (status === 'error') { ss = 'border-[#ef4444] bg-[#ef4444]/15'; tc = 'text-[#ef4444]'; }
          return (
            <div key={step.id} className={`flex-1 border flex items-center justify-between px-1.5 py-1 relative transition-all ${ss}`}>
              {(status === 'processing' || status === 'error') && <div className={`absolute top-0 left-0 w-full h-[2px] ${status === 'error' ? 'bg-[#ef4444]' : 'bg-[#f59e0b]'}`} />}
              <span className={`text-[9px] font-bold ${tc}`}>{step.title}</span>
              {status === 'success' ? <CheckCircle2 size={10} className={tc} /> : status === 'error' ? <XCircle size={10} className={tc} /> : status === 'processing' ? <step.icon size={10} className={tc} /> : null}
            </div>
          );
        })}
      </div>
    </section>
  );
}

// ─── 상세 팝업 모달 (CCTV 스타일) ────────────────────────────────────────────
function DetailModal({
  item, onClose, liveData, CAMERA_LIST, imageLogs,
}: {
  item: ViewItem;
  onClose: () => void;
  liveData: Record<string, CameraData>;
  CAMERA_LIST: { id: string; name: string; latestData: CameraData }[];
  imageLogs: ImageLog[];
}) {
  const { language, isAdminAuth, isDevAuth } = useAppStore();
  const t = locales[language];
  const [tab, setTab] = useState<'data' | 'edit'>('data');
  const [loading, setLoading] = useState(false);

  const isAuthorized = isAdminAuth || isDevAuth;

  const cam = item.type === 'camera' ? CAMERA_LIST.find(c => c.id === item.id) : null;
  const imgLog = item.type === 'image' ? imageLogs.find(i => i.id === item.id) : null;
  const rawData: CameraData | ImageLog | null = item.type === 'camera'
    ? (liveData[item.id] ?? cam?.latestData ?? null)
    : imgLog ?? null;

  const [editLogic, setEditLogic] = useState<Record<string, unknown>>({ ...(((rawData as CameraData)?.logic as Record<string, unknown>) || {}) });
  const [editDisplay, setEditDisplay] = useState<Record<string, string>>({ ...(((rawData as CameraData)?.display as Record<string, string>) || {}) });
  const [editLabel, setEditLabel] = useState<string>((rawData as CameraData)?.predicted_label || '');
  const [editUnknown, setEditUnknown] = useState<boolean>(Boolean((rawData as CameraData)?.is_unknown));
  const confidence = parseFloat(((editLogic.confidence as number) ?? (rawData as CameraData)?.confidence ?? 0).toFixed(2));

  const handleSaveOverride = async () => {
    if (item.type !== 'camera' || !isAuthorized) return;
    setLoading(true);
    try {
      await axios.post(`/api/override`, {
        cam_id: item.id,
        predicted_label: editLabel,
        confidence: confidence,
        is_unknown: editUnknown,
        logic: editLogic,
        display: editDisplay
      });
      alert(`[${item.id}] 데이터 변경 완료!`);
    } catch (err) {
      console.error(err);
      alert("Override 저장 실패");
    } finally {
      setLoading(false);
    }
  };

  const stepStatuses: StepStatus[] = (() => {
    const s: StepStatus[] = ['idle', 'idle', 'idle', 'idle'];
    const ld = rawData as CameraData;
    if (ld?.logic) {
      const logic = ld.logic as Record<string, unknown>;
      const completed = (logic.completed_labels as string[] | undefined) ?? [];
      const expected = (logic.expected_label as string | null | undefined) ?? null;
      const effective = (logic.effective_label as string | null | undefined) ?? null;
      const allowed = logic.allowed_transition as boolean | undefined;
      const targets = ['Target1', 'Target2', 'Target3', 'Target4'];
      for (let i = 0; i < 4; i++) {
        const t = targets[i];
        if (completed.includes(t)) {
          s[i] = 'success';
        } else if (t === expected) {
          if (effective === expected) s[i] = 'processing';
          else if (allowed === false && effective && effective !== 'unknown') s[i] = 'error';
          else s[i] = 'idle';
        }
      }
    }
    return s;
  })();

  return (
    <div className="fixed inset-0 bg-black/70 z-[150] flex items-center justify-center p-3">
      <div className="bg-[#18181b] border border-[#52525b] w-[94vw] h-[92vh] flex flex-col shadow-[0_0_40px_rgba(0,0,0,0.8)]">

        {/* 타이틀바 */}
        <div className="flex items-center justify-between px-4 py-2.5 bg-[#27272a] border-b-2 border-[#E50012] shrink-0">
          <div className="flex items-center gap-3">
            {item.type === 'camera' ? <Camera size={18} className="text-[#E50012]" /> : <ImageIcon size={18} className="text-[#E50012]" />}
            <span className="font-black text-zinc-100 tracking-widest text-base">
              {item.type === 'camera' ? `${item.id} — ${cam?.name}` : item.id}
            </span>
            <span className="px-2 py-0.5 border border-[#f59e0b] text-[#f59e0b] font-mono text-xs">
              {item.type === 'camera' ? 'LIVE STREAM' : `T${(imgLog as ImageLog)?.target_idx} · ${(imgLog as ImageLog)?.reason}`}
            </span>
          </div>
          <button onClick={onClose} className="text-zinc-400 hover:text-white p-2 hover:bg-zinc-700 border border-transparent hover:border-zinc-500 flex items-center gap-1.5 text-xs font-bold tracking-widest transition-colors">
            <X size={16} /> 닫기
          </button>
        </div>

        {/* 바디 */}
        <div className="flex-1 flex min-h-0">

          {/* 좌측: 대형 영상/이미지 */}
          <div className="flex-[7] relative bg-black flex items-center justify-center border-r border-[#3f3f46]">
            {item.type === 'camera' ? (
              <>
                <img
                  id={`video-stream-popup-${item.id}`}
                  className="w-full h-full object-contain opacity-90"
                  alt={item.id}
                />
                <div className="absolute inset-0 flex flex-col items-center justify-center opacity-20 pointer-events-none">
                  <Activity size={80} className="text-zinc-500" />
                </div>
              </>
            ) : (
              (imgLog as ImageLog)?.image_url ? (
                <img src={(imgLog as ImageLog).image_url} alt={item.id} className="w-full h-full object-contain" />
              ) : (
                <div className="flex flex-col items-center justify-center opacity-30">
                  <ImageIcon size={80} className="text-zinc-500 mb-4" />
                  <span className="font-mono text-zinc-500">NO IMAGE</span>
                </div>
              )
            )}
            <div className="absolute top-3 left-3 bg-[#18181b]/90 px-3 py-1 border border-[#3f3f46] font-mono text-zinc-300 text-xs z-10">{item.id}</div>
            {item.type === 'camera' && (rawData as CameraData)?.inference && (
              <div className="absolute top-3 right-3 bg-[#18181b]/90 px-3 py-1 border border-[#f59e0b] text-[#f59e0b] font-mono flex items-center gap-2 z-10 text-xs animate-pulse">
                <div className="bg-[#f59e0b] w-2 h-2" /> TARGET INFERENCE
              </div>
            )}
            {(rawData as CameraData)?.display && ((rawData as CameraData).display as Record<string,string>)?.system_message && (
              <div className="absolute bottom-4 inset-x-0 flex justify-center z-10 pointer-events-none">
                <div className={`px-8 py-2 border text-2xl font-black uppercase tracking-widest bg-[#18181b]/90 ${((rawData as CameraData).logic as Record<string,unknown>)?.allowed_transition === false ? 'text-[#ef4444] border-[#ef4444]' : 'text-[#22c55e] border-[#3f3f46]'}`}>
                  {((rawData as CameraData).display as Record<string,string>).system_message}
                </div>
              </div>
            )}
          </div>

          {/* 우측: 탭 패널 */}
          <div className="flex-[3] flex flex-col bg-[#18181b] min-w-0">
            <div className="flex border-b border-[#3f3f46] shrink-0">
              {(['data', 'edit'] as const).map(tabKey => (
                <button
                  key={tabKey}
                  onClick={() => setTab(tabKey)}
                  className={`flex-1 py-2.5 text-xs font-black tracking-widest uppercase border-b-2 transition-colors ${
                    tab === tabKey ? 'border-[#E50012] text-zinc-100 bg-[#27272a]' : 'border-transparent text-zinc-500 hover:text-zinc-300'
                  }`}
                >
                  {tabKey === 'data' ? '실시간 데이터' : '데이터 수정'}
                </button>
              ))}
            </div>

            {tab === 'data' ? (
              <div className="flex-1 overflow-y-auto p-4 custom-scrollbar">
                <div className="grid grid-cols-2 gap-2 mb-4">
                  {[
                    { label: 'Confidence', value: `${(((rawData as CameraData)?.confidence ?? 0) * 100).toFixed(1)}%`, color: '#f59e0b' },
                    { label: 'State', value: (rawData as CameraData)?.confirmed_state || (rawData as ImageLog)?.status || '-', color: '#22c55e' },
                    { label: 'Label', value: (rawData as CameraData)?.predicted_label || '-', color: '#a1a1aa' },
                    { label: 'Unknown', value: (rawData as CameraData)?.is_unknown ? 'TRUE' : 'FALSE', color: (rawData as CameraData)?.is_unknown ? '#f59e0b' : '#52525b' },
                  ].map(card => (
                    <div key={card.label} className="bg-[#27272a] border border-[#3f3f46] p-3">
                      <div className="text-[9px] text-zinc-500 uppercase tracking-widest mb-1">{card.label}</div>
                      <div className="font-mono text-xs font-bold truncate" style={{ color: card.color }}>{card.value}</div>
                    </div>
                  ))}
                </div>
                <div className="text-[9px] font-bold text-zinc-600 uppercase tracking-widest mb-1">Raw JSON</div>
                <pre className="text-[9px] font-mono text-zinc-400 bg-[#27272a] p-3 border border-[#3f3f46] overflow-auto max-h-[55vh] whitespace-pre-wrap break-all custom-scrollbar">
                  {JSON.stringify(rawData, null, 2)}
                </pre>
              </div>
            ) : (
              <div className="flex-1 overflow-y-auto p-4 space-y-4 custom-scrollbar">
                {item.id === 'CAM_00' ? (
                  <div className="space-y-6">
                    <div className="p-4 bg-[#020617] border border-[#1e293b] rounded-sm">
                      <h3 className="text-[#38bdf8] font-black text-sm tracking-tighter uppercase mb-4 flex items-center gap-2">
                        <Globe size={16} className="animate-pulse" /> {t.devPortal}
                      </h3>
                      
                      <div className="space-y-4">
                        <div>
                          <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-2 block">{t.contentType}</label>
                          <div className="grid grid-cols-2 gap-2">
                            {['youtube', 'news'].map(type => (
                              <button key={type}
                                onClick={() => setEditDisplay(p => ({ ...p, type }))}
                                className={`py-2 text-[10px] font-black uppercase border transition-all ${editDisplay.type === type ? 'border-[#38bdf8] bg-[#38bdf8]/20 text-[#38bdf8]' : 'border-[#1e293b] bg-[#0f172a] text-zinc-600 hover:text-zinc-400'}`}>
                                {type === 'youtube' ? t.youtubeFeed : t.newsFeed}
                              </button>
                            ))}
                          </div>
                        </div>

                        <div>
                          <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-1 block">{t.sourceUrl}</label>
                          <input type="text" value={editDisplay.url || ''}
                            onChange={e => setEditDisplay(p => ({ ...p, url: e.target.value }))}
                            className="w-full bg-[#0f172a] border border-[#1e293b] text-[#38bdf8] text-xs px-3 py-2 font-mono focus:outline-none focus:border-[#38bdf8]"
                            placeholder={editDisplay.type === 'youtube' ? "YouTube Embed URL..." : "News Source Name..."} />
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <>
                    {/* 타겟 단계 */}
                    <div>
                      <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-2 block">{t.analysisStep}</label>
                      <div className="grid grid-cols-4 gap-1">
                        {[1,2,3,4].map(n => (
                          <button key={n}
                            onClick={() => setEditLogic(p => ({ ...p, current_step_index: n }))}
                            className={`py-3 font-black text-sm border transition-all ${editLogic.current_step_index === n ? 'border-[#E50012] bg-[#E50012] text-white' : 'border-[#3f3f46] bg-[#27272a] text-zinc-400 hover:border-zinc-400 hover:text-white'}`}>
                            T{n}
                          </button>
                        ))}
                      </div>
                    </div>
                    {/* 신뢰도 */}
                    <div>
                      <div className="flex justify-between items-center mb-1">
                        <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">{t.confidence}</label>
                        <span className="text-[#f59e0b] font-mono text-xs font-bold">{confidence.toFixed(2)}</span>
                      </div>
                      <input type="range" min="0" max="1" step="0.01" value={confidence}
                        onChange={e => setEditLogic(p => ({ ...p, confidence: parseFloat(e.target.value) }))}
                        className="w-full accent-[#f59e0b] cursor-pointer" />
                    </div>
                    {/* 공정 전환 */}
                    <div className="flex items-center justify-between">
                      <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">{t.allowTransition}</label>
                      <button onClick={() => setEditLogic(p => ({ ...p, allowed_transition: !p.allowed_transition }))}
                        className={`px-3 py-2 border font-bold text-xs transition-all ${editLogic.allowed_transition ? 'border-[#22c55e] bg-[#22c55e]/20 text-[#22c55e]' : 'border-[#ef4444] bg-[#ef4444]/20 text-[#ef4444]'}`}>
                        {editLogic.allowed_transition ? t.allowed : t.blocked}
                      </button>
                    </div>
                    {/* 미인식 */}
                    <div className="flex items-center justify-between">
                      <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest">{t.unknownStatus}</label>
                      <button onClick={() => setEditUnknown(v => !v)}
                        className={`px-3 py-2 border font-bold text-xs transition-all ${editUnknown ? 'border-[#f59e0b] bg-[#f59e0b]/20 text-[#f59e0b]' : 'border-[#3f3f46] bg-[#27272a] text-zinc-400 hover:border-zinc-400'}`}>
                        {editUnknown ? 'TRUE' : 'FALSE'}
                      </button>
                    </div>
                    {/* 시스템 메시지 */}
                    <div>
                      <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-1 block">{t.systemMessage}</label>
                      <input type="text" value={editDisplay.system_message || ''}
                        onChange={e => setEditDisplay(p => ({ ...p, system_message: e.target.value }))}
                        className="w-full bg-[#27272a] border border-[#3f3f46] text-zinc-100 text-sm px-3 py-2 font-mono focus:outline-none focus:border-zinc-400"
                        placeholder="ANALYZING..." />
                    </div>
                    {/* 예측 레이블 */}
                    <div>
                      <label className="text-[10px] font-bold text-zinc-500 uppercase tracking-widest mb-1 block">{t.predictedLabel}</label>
                      <input type="text" value={editLabel}
                        onChange={e => setEditLabel(e.target.value)}
                        className="w-full bg-[#27272a] border border-[#3f3f46] text-zinc-100 text-sm px-3 py-2 font-mono focus:outline-none focus:border-zinc-400" />
                    </div>
                  </>
                )}
                <div className="pt-2">
                  <button
                    onClick={handleSaveOverride}
                    disabled={loading || item.type !== 'camera' || !isAuthorized}
                    className={`w-full flex items-center justify-center gap-2 py-3 font-black tracking-widest text-xs transition-all shadow-lg ${isAuthorized ? 'bg-[#E50012] hover:bg-[#ff3040] text-white' : 'bg-zinc-800 text-zinc-600 border border-zinc-700 cursor-not-allowed'}`}
                  >
                    <Save size={14} className={loading ? "animate-spin" : ""} />
                    {!isAuthorized ? t.authRequired : (loading ? "SAVING..." : t.saveOverride)}
                  </button>
                </div>
                {isAuthorized && <div className="text-[12px] text-[#22c55e] font-mono italic"> 주의: 저장 시 서버 설정이 즉시 변경됩니다.</div>}
                {!isAuthorized && <div className="text-[9px] text-zinc-600 font-mono italic">※ 데이터 수정을 위해 관리자/개발자 모드(로고 3회 클릭) 로그인이 필요합니다.</div>}
              </div>
            )}

            <div className="p-3 border-t border-[#3f3f46] shrink-0">
              <button onClick={onClose} className="w-full py-3 border border-[#52525b] text-zinc-400 hover:text-white text-sm font-bold transition-colors">닫기</button>
            </div>
          </div>
        </div>

        {/* 하단 Step 상태바 */}
        <div className="flex border-t border-[#3f3f46] shrink-0">
          {[1,2,3,4].map((n, i) => {
            const status = stepStatuses[i] || 'idle';
            let ss = 'bg-[#27272a] text-zinc-600';
            let tc = 'text-zinc-600';
            if (status === 'processing') { ss = 'bg-[#f59e0b]/10 border-t-2 border-[#f59e0b]'; tc = 'text-[#f59e0b]'; }
            else if (status === 'success') { ss = 'bg-[#22c55e]/10 border-t-2 border-[#22c55e]'; tc = 'text-[#22c55e]'; }
            else if (status === 'error') { ss = 'bg-[#ef4444]/10 border-t-2 border-[#ef4444]'; tc = 'text-[#ef4444]'; }
            return (
              <div key={n} className={`flex-1 flex items-center justify-between px-4 py-3 border-r border-[#3f3f46] last:border-r-0 ${ss}`}>
                <span className={`text-sm font-black tracking-wider ${tc}`}>{t.target} {n}</span>
                <div className="flex items-center gap-2">
                  {status === 'success' && <CheckCircle2 size={18} className={tc} />}
                  {status === 'error' && <XCircle size={18} className={tc} />}
                  {status === 'processing' && <Activity size={18} className={`${tc} animate-pulse`} />}
                  <span className={`text-xs font-mono uppercase ${tc}`}>{status}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ── 테스트 모드 모달 (DB 데이터 조회) ──────────────────────────────────────────
const VIDEO_EXTS = /\.(mp4|webm|mov|avi|mkv|m4v|wmv|flv|ts|ogv)$/i;

function TestModeModal() {
  const {
    isTestModeOpen, setTestModeOpen, dbLogs, setDbLogs, language,
    startVideoAnalysis, startImageAnalysis, applyImageInspections, setImageAnalysisError,
  } = useAppStore();
  const t = locales[language];
  const [loading, setLoading] = useState(false);
  const [offset, setOffset] = useState(0);
  const [hasMore, setHasMore] = useState(true);
  const LIMIT = 30;

  const fileInputRef = useRef<HTMLInputElement>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const observerRef = useRef<IntersectionObserver | null>(null);

  const fetchLogs = async (isNew = false) => {
    if (loading) return;
    setLoading(true);
    const currentOffset = isNew ? 0 : offset;
    try {
      const resp = await fetch(`/api/inspection-logs?offset=${currentOffset}&limit=${LIMIT}`);
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();

      if (Array.isArray(data)) {
        if (isNew) {
          setDbLogs(data);
          setOffset(data.length);
        } else {
          setDbLogs([...dbLogs, ...data]);
          setOffset(p => p + data.length);
        }
        if (data.length < LIMIT) setHasMore(false);
        else setHasMore(true);
      }
    } catch (err) {
      console.error("Failed to fetch logs:", err);
    } finally {
      setLoading(false);
    }
  };

  const startStreamingVideo = (videoFile: File) => {
    const jobId = (typeof crypto !== 'undefined' && 'randomUUID' in crypto)
      ? crypto.randomUUID().replace(/-/g, '')
      : `${Date.now().toString(16)}${Math.random().toString(16).slice(2, 10)}`;
    const cameraId = `VIDEO_${jobId.slice(0, 12)}`;
    const videoUrl = URL.createObjectURL(videoFile);
    startVideoAnalysis({ jobId, cameraId, filename: videoFile.name, videoUrl });
    setTestModeOpen(false);
  };

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    const files = Array.from(e.target.files);
    const videos = files.filter(f => VIDEO_EXTS.test(f.name) || f.type.startsWith('video/'));
    const images = files.filter(f => !videos.includes(f));

    try {
      // 영상은 한 번에 하나만 (분석 뷰가 단일 슬롯)
      if (videos.length > 0) {
        if (videos.length > 1) {
          alert('영상은 한 번에 한 개만 분석됩니다. 첫 번째 영상으로 진행합니다.');
        }
        startStreamingVideo(videos[0]);
      }

      // 이미지는 캐러셀 분석 뷰로 — store에 items 등록 후 batch 추론을 비동기로 실행
      if (images.length > 0) {
        const items = images.map((f) => ({
          filename: f.name,
          imageUrl: URL.createObjectURL(f),
        }));
        startImageAnalysis(items);
        setTestModeOpen(false);

        const fd = new FormData();
        images.forEach((f) => fd.append('files', f));
        fetch('/api/inspect-image', { method: 'POST', body: fd })
          .then(async (resp) => {
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            const insp = (data.inspections ?? []) as Array<Record<string, unknown>>;
            const results = images.map((_, i) => {
              const r = insp[i];
              if (!r) return { error: '결과 없음' };
              if (r.error) return { error: String(r.error) };
              const detail = (r.operational_detail ?? {}) as Record<string, unknown>;
              const decision = (detail.decision ?? {}) as Record<string, unknown>;
              return {
                predicted_label: String(detail.predicted_label ?? r.predicted_label ?? 'unknown'),
                final_label: String(detail.final_label ?? r.final_label ?? 'unknown'),
                confidence: Number(decision.top1_score ?? r.confidence ?? 0),
                decision_type: detail.decision_type as string | undefined,
                scores: detail.scores as Record<string, number> | undefined,
                thresholds: detail.thresholds as Record<string, number> | undefined,
                anomaly_flag: r.anomaly_flag as boolean | undefined,
                decision_reason: decision.reason as string | undefined,
                is_unknown: detail.unknown as boolean | undefined,
                ambiguous: detail.ambiguous as boolean | undefined,
                reinspect_needed: detail.reinspect_needed as boolean | undefined,
              };
            });
            applyImageInspections(results);
          })
          .catch((err) => {
            console.error(err);
            setImageAnalysisError(String(err?.message ?? err));
          });
      }
    } finally {
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const handleReinspect = async (logId: number) => {
    setLoading(true);
    let success = false;
    try {
      const resp = await fetch(`/api/reinspect-log/${logId}`, {
        method: "POST"
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      success = data.status === "success";
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
    if (success) {
      alert(`${t.reinspect} 완료!`);
      fetchLogs(true);
    }
  };

  useEffect(() => {
    if (isTestModeOpen) {
      fetchLogs(true);
    }
  }, [isTestModeOpen]);

  // 무한 스크롤 관찰자 설정
  useEffect(() => {
    if (!isTestModeOpen) return;
    
    const target = document.getElementById("scroll-trigger");
    if (!target) return;

    observerRef.current = new IntersectionObserver((entries) => {
      if (entries[0].isIntersecting && hasMore && !loading) {
        fetchLogs(false);
      }
    }, { threshold: 1.0 });

    observerRef.current.observe(target);
    return () => observerRef.current?.disconnect();
  }, [isTestModeOpen, hasMore, loading, offset]);

  if (!isTestModeOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-md z-[200] flex items-center justify-center p-4">
      <div className="bg-[#18181b] border-2 border-[#f59e0b] w-[95vw] h-[90vh] shadow-[0_0_50px_rgba(245,158,11,0.15)] flex flex-col">
        <div className="flex items-center justify-between p-5 border-b border-[#3f3f46] bg-[#27272a]">
          <div className="flex items-center gap-4">
            <div className="bg-[#f59e0b]/10 p-2 rounded-sm border border-[#f59e0b]/30">
              <Database size={28} className="text-[#f59e0b]" />
            </div>
            <div>
              <h2 className="text-2xl font-black text-zinc-100 tracking-tighter uppercase leading-none">{t.testMode}</h2>
              <p className="text-[10px] text-zinc-500 font-mono mt-1 uppercase tracking-widest">Total Cached: {dbLogs.length} Rows</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <input type="file" multiple accept="image/*,video/*" className="hidden" ref={fileInputRef} onChange={handleUpload} />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              className="flex items-center gap-2 px-6 py-3 bg-[#E50012] hover:bg-[#ff3040] text-white text-xs font-black transition-all disabled:opacity-50 shadow-lg border border-[#E50012] active:scale-95"
            >
              <ImageIcon size={16} />
              {t.localTest}
            </button>
            <button
              onClick={() => fetchLogs(true)}
              disabled={loading}
              className="flex items-center gap-2 px-6 py-3 bg-zinc-800 hover:bg-zinc-700 border border-zinc-600 text-zinc-300 text-xs font-black transition-all disabled:opacity-50"
            >
              <RefreshCw size={16} className={loading ? "animate-spin" : ""} />
              {t.refresh}
            </button>
            <button
              onClick={() => setTestModeOpen(false)}
              className="text-zinc-500 hover:text-white p-2 hover:bg-zinc-800 transition-colors"
            >
              <X size={32} />
            </button>
          </div>
        </div>

        <div className="flex-1 overflow-auto p-2 custom-scrollbar bg-black/60" ref={scrollRef}>
          <table className="w-full text-left border-collapse min-w-[1200px]">
            <thead className="sticky top-0 bg-[#27272a] shadow-xl z-20">
              <tr className="text-[#f59e0b] text-xs font-black uppercase tracking-widest border-b border-[#3f3f46]">
                <th className="p-4 pl-6">ID</th>
                <th className="p-4">Timestamp</th>
                <th className="p-4">Source</th>
                <th className="p-4">State</th>
                <th className="p-4">Label</th>
                <th className="p-4">Conf</th>
                <th className="p-4">Anomaly</th>
                <th className="p-4">File Path</th>
                <th className="p-4 pr-6 text-center">{t.action}</th>
              </tr>
            </thead>
            <tbody className="text-sm font-mono text-zinc-400 divide-y divide-zinc-800/50">
              {dbLogs.map((log) => (
                <tr key={log.id} className="hover:bg-white/5 transition-colors group">
                  <td className="p-5 pl-6 text-zinc-600 font-bold group-hover:text-zinc-400 transition-colors">#{log.id}</td>
                  <td className="p-5 whitespace-nowrap">{new Date(log.timestamp).toLocaleString()}</td>
                  <td className="p-5 text-blue-400/80 font-black uppercase text-[10px]">{log.source_type}</td>
                  <td className="p-5">
                    <span className={`px-3 py-1.5 rounded-sm text-[11px] font-black tracking-tighter ${log.confirmed_state === 'ERROR_SEQUENCE' ? 'bg-red-500/10 text-red-500 border border-red-500/30' : 'bg-green-500/10 text-green-500 border border-green-500/30'}`}>
                      {log.confirmed_state}
                    </span>
                  </td>
                  <td className="p-5 text-zinc-200 font-black">{log.predicted_label}</td>
                  <td className="p-5 text-[#f59e0b] font-black text-base italic">{(log.confidence * 100).toFixed(1)}%</td>
                  <td className="p-5">
                    {log.anomaly_flag ? 
                      <div className="flex items-center gap-2 text-red-500"><ShieldAlert size={14} className="animate-pulse" /><span className="font-black text-[10px] tracking-widest">ALARM</span></div> 
                      : <span className="text-zinc-800 font-black text-[10px]">NORMAL</span>
                    }
                  </td>
                  <td className="p-5 text-[10px] text-zinc-500 truncate max-w-[300px] hover:text-[#38bdf8] cursor-help transition-colors">{log.file_path}</td>
                  <td className="p-5 pr-6 text-center">
                    <button 
                      onClick={() => handleReinspect(log.id)}
                      disabled={loading}
                      className="px-5 py-3 font-normal bg-zinc-800 hover:bg-[#E50012] text-white text-[11px] font-black tracking-widest transition-all shadow-md active:scale-90 border border-zinc-700"
                    >
                      {t.reinspect}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          {dbLogs.length === 0 && !loading && (
            <div className="h-full flex flex-col items-center justify-center opacity-30 mt-20">
              <Database size={64} className="mb-4" />
              <p className="text-lg font-bold">NO DATA FOUND</p>
            </div>
          )}
        </div>

        <div className="p-3 bg-[#27272a] border-t border-[#3f3f46] flex justify-between items-center">
          <span className="text-[10px] font-mono text-zinc-500 tracking-widest">CONNECTED TO DB: factory_test.db</span>
          <button
            onClick={() => setTestModeOpen(false)}
            className="px-8 py-2 bg-[#f59e0b] hover:bg-[#ffb020] text-black font-black text-sm transition-all"
          >
            닫기
          </button>
        </div>
      </div>
    </div>
  );
}

// ─── 설정 모달 ────────────────────────────────────────────────────────────────
function SettingsModal() {
  const { language, setLanguage, isSettingsOpen, setSettingsOpen, setTestModeOpen } = useAppStore();
  const t = locales[language];
  if (!isSettingsOpen) return null;
  return (
    <div className="fixed inset-0 bg-black/70 backdrop-blur-sm z-[100] flex items-center justify-center p-4">
      <div className="bg-[#18181b] border border-[#3f3f46] w-full max-w-md shadow-2xl flex flex-col">
        <div className="flex items-center justify-between p-4 border-b border-[#27272a] bg-[#27272a]">
          <div className="flex items-center gap-3">
            <Settings size={20} className="text-zinc-300" />
            <h2 className="text-lg font-bold text-zinc-100">{t.settingsTitle}</h2>
          </div>
          <button onClick={() => setSettingsOpen(false)} className="text-zinc-400 hover:text-white p-1 hover:bg-zinc-600 transition-colors"><X size={20} /></button>
        </div>
        <div className="p-6 space-y-6">
          <div className="space-y-3">
            <label className="flex items-center gap-2 text-sm font-semibold text-zinc-300 uppercase tracking-widest"><Globe size={16} />{t.languageSelect}</label>
            <div className="grid grid-cols-2 gap-3 mt-2">
              {(['ko', 'en'] as const).map((lang) => (
                <button key={lang} onClick={() => setLanguage(lang)} className={`p-3 border font-bold transition-all flex items-center justify-center ${language === lang ? 'border-[#E50012] bg-[#E50012]/10 text-white' : 'border-[#52525b] bg-[#27272a] text-zinc-300 hover:border-zinc-400 hover:text-white hover:bg-[#3f3f46]'}`}>
                  {lang === 'ko' ? '한국어' : 'English'}
                </button>
              ))}
            </div>
          </div>

          <div className="pt-4 border-t border-zinc-800">
            <button
              onClick={() => { setSettingsOpen(false); setTestModeOpen(true); }}
              className="w-full flex items-center justify-center gap-3 p-4 bg-zinc-800 hover:bg-[#f59e0b] hover:text-black text-zinc-300 font-black tracking-widest transition-all group"
            >
              테스트 모드(Test  Mode)
            </button>
          </div>
        </div>
        <div className="p-4 border-t border-[#27272a] bg-[#18181b] flex justify-end">
          <button onClick={() => setSettingsOpen(false)} className="px-6 py-2 bg-[#3f3f46] hover:bg-white hover:text-black border border-[#52525b] text-white font-bold transition-colors">{t.close}</button>
        </div>
      </div>
    </div>
  );
}

// ─── 관리자/개발자 모달 ──────────────────────────────────────────────────────
function AdminModal() {
  const { isAdminOpen, setAdminOpen, setAdminAuth, isAdminAuth, setDevAuth, isDevAuth, language } = useAppStore();
  const t = locales[language];
  const [pw, setPw] = useState("");
  const [error, setError] = useState(false);

  if (!isAdminOpen) return null;

  const handleLogin = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (pw === import.meta.env.VITE_ADMIN_PW) {
      if (isAdminAuth) {
        if (window.confirm("관리자 모드를 비활성화 하시겠습니까?")) {
          setAdminAuth(false);
          setAdminOpen(false);
          setPw("");
          setError(false);
        }
      } else {
        if (window.confirm("관리자 모드를 활성화 하시겠습니까?")) {
          setAdminAuth(true);
          setAdminOpen(false);
          setPw("");
          setError(false);
        }
      }
    } else if (pw === import.meta.env.VITE_DEV_PW) {
      if (isDevAuth) {
        if (window.confirm("개발자 모드를 비활성화 하시겠습니까?")) {
          setDevAuth(false);
          setAdminOpen(false);
          setPw("");
          setError(false);
        }
      } else {
        if (window.confirm("개발자 모드를 활성화 하시겠습니까?")) {
          setDevAuth(true);
          setAdminOpen(false);
          setPw("");
          setError(false);
        }
      }
    } else {
      setError(true);
      setPw("");
    }
  };

  return (
    <div className="fixed inset-0 bg-black/80 backdrop-blur-sm z-[300] flex items-center justify-center p-4 animate-in fade-in duration-300">
      <form 
        onSubmit={handleLogin} 
        className="bg-[#27272a] border border-[#E50012] w-full max-w-sm shadow-2xl flex flex-col p-8 relative overflow-hidden group"
      >
        <div className="flex flex-col items-center mb-10 relative z-10">
          <div className="w-16 h-16 bg-[#18181b] border border-[#3f3f46] flex items-center justify-center rounded-2xl mb-4 shadow-sm">
            <Lock size={32} className={`${error ? 'text-[#ef4444]' : 'text-zinc-400'} transition-colors duration-500`} />
          </div>
          <h2 className="text-xl font-black text-zinc-100 tracking-[0.2em] uppercase">{t.systemAuth}</h2>
          <div className="h-0.5 w-12 bg-[#E50012] mt-3" />
        </div>
        
        <div className="space-y-8 relative z-10">
          <div className="space-y-4">
            <div className="flex justify-between items-center px-1">
              <label className="text-[10px] font-black text-zinc-400 uppercase tracking-widest">{t.accessCode}</label>
              {error && <span className="text-[#ef4444] text-[9px] font-bold animate-pulse">{t.invalidCode}</span>}
            </div>
            
            <div className="relative group flex justify-center">
              <input 
                type="password"
                autoFocus
                maxLength={4}
                value={pw}
                onChange={(e) => { setPw(e.target.value); setError(false); }}
                className={`w-full max-w-[310px] bg-[#18181b] border-b-2 ${error ? 'border-[#ef4444]' : 'border-[#3f3f46]'} focus:border-[#E50012] transition-all text-white text-center text-2xl tracking-[0.5em] py-2 outline-none font-mono shadow-inner`}
              />
            </div>
          </div>

          <div className="space-y-3">
            <button 
              type="submit"
              className="w-full py-4 bg-[#E50012] hover:bg-[#ff3040] text-white font-black tracking-[0.2em] uppercase transition-all shadow-md active:scale-95 flex items-center justify-center gap-2 group/btn"
            >
              <span>{t.login}</span>
              <ChevronRight size={18} className="group-hover/btn:translate-x-1 transition-transform" />
            </button>
            
            <button 
              type="button"
              onClick={() => { setAdminOpen(false); setPw(""); setError(false); }}
              className="w-full py-2 text-zinc-500 hover:text-zinc-300 text-[10px] font-bold tracking-[0.2em] uppercase transition-all"
            >
              {t.cancel}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}

// ─── 모바일 송출 모드 (Mobile Source View) ─────────────────────────────────────────
function MobileSourceView({ onExit }: { onExit: () => void }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<ReturnType<typeof window.setInterval> | null>(null);
  const sentCountRef = useRef(0);
  const updateLiveData = useAppStore((state) => state.updateLiveData);

  const [connectionState, setConnectionState] = useState<StreamConnectionState>('idle');
  const [cameraId, setCameraId] = useState('CAM_WS');
  const [modelMode, setModelMode] = useState<StreamModelMode>('real');
  const [scenario, setScenario] = useState('normal_target2_accept');
  const [targetOrder, setTargetOrder] = useState('Target1,Target2,Target3,Target4');
  const [sampleMs, setSampleMs] = useState(100);
  const [frameWidth, setFrameWidth] = useState(1280);
  const [jpegQuality, setJpegQuality] = useState(0.9);
  const [saveArtifacts, setSaveArtifacts] = useState(false);
  const [stats, setStats] = useState({
    sent: 0,
    received: 0,
    predicted: '-',
    finalLabel: '-',
    message: 'READY',
  });

  const streaming = connectionState === 'streaming' || connectionState === 'connecting';

  const stopStreaming = useCallback(() => {
    if (intervalRef.current) {
      window.clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    wsRef.current?.close();
    wsRef.current = null;
    setConnectionState('idle');
  }, []);

  useEffect(() => {
    async function startCamera() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: 'environment' },
            width: { ideal: 1920 },
            height: { ideal: 1080 },
            frameRate: { ideal: 30, max: 30 },
          },
          audio: false,
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error('Camera access failed:', err);
        alert('Camera permission is required.');
      }
    }
    startCamera();

    return () => {
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach(track => track.stop());
      }
      stopStreaming();
    };
  }, [stopStreaming]);

  const sendFrame = useCallback((ws: WebSocket) => {
    if (!videoRef.current || !canvasRef.current || ws.readyState !== WebSocket.OPEN) return;
    const video = videoRef.current;
    if (video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;

    const sourceWidth = video.videoWidth || 640;
    const sourceHeight = video.videoHeight || 480;
    const height = Math.max(1, Math.round(frameWidth * (sourceHeight / sourceWidth)));
    const canvas = canvasRef.current;
    canvas.width = frameWidth;
    canvas.height = height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, frameWidth, height);
    const frame = canvas.toDataURL('image/jpeg', jpegQuality);
    const nextSent = sentCountRef.current + 1;
    sentCountRef.current = nextSent;
    ws.send(JSON.stringify({
      cameraId,
      frame,
      model_mode: modelMode,
      scenario,
      target_order: parseTargetOrder(targetOrder),
      strict_sequence: true,
      save_artifacts: saveArtifacts,
      reset_state: nextSent === 1,
    }));
    setStats(prev => ({ ...prev, sent: nextSent }));
  }, [cameraId, frameWidth, jpegQuality, modelMode, saveArtifacts, scenario, targetOrder]);

  const startStreaming = () => {
    if (streaming) return;
    sentCountRef.current = 0;
    setStats({ sent: 0, received: 0, predicted: '-', finalLabel: '-', message: 'CONNECTING' });
    setConnectionState('connecting');

    const ws = new WebSocket(buildSourceWsUrl(cameraId, modelMode, scenario));
    wsRef.current = ws;

    ws.onopen = () => {
      setConnectionState('streaming');
      sendFrame(ws);
      intervalRef.current = window.setInterval(() => sendFrame(ws), Math.max(50, sampleMs));
    };

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data as string) as { type?: string; cameraId?: string; payload?: CameraData };
        if (message.type === 'inference' && message.cameraId && message.payload) {
          updateLiveData(message.cameraId, message.payload);
          setStats(prev => ({
            ...prev,
            received: prev.received + 1,
            predicted: message.payload?.predicted_label || '-',
            finalLabel: message.payload?.final_label || message.payload?.effective_label || '-',
            message: message.payload?.display?.system_message || message.payload?.confirmed_state || 'UPDATED',
          }));
        }
      } catch {
        setStats(prev => ({ ...prev, message: 'BAD MESSAGE' }));
      }
    };

    ws.onerror = () => {
      setConnectionState('error');
      setStats(prev => ({ ...prev, message: 'WS ERROR' }));
    };

    ws.onclose = () => {
      if (intervalRef.current) {
        window.clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      setConnectionState(prev => prev === 'error' ? 'error' : 'idle');
    };
  };

  const toggleStreaming = () => {
    if (streaming) {
      stopStreaming();
      return;
    }
    startStreaming();
  };

  return (
    <div className="fixed inset-0 bg-[#18181b] z-[500] grid grid-cols-1 lg:grid-cols-[minmax(280px,420px)_1fr] gap-4 p-4 overflow-y-auto">
      <div className="bg-[#27272a] border border-[#3f3f46] flex flex-col min-h-0">
        <div className="px-4 py-3 border-b border-[#3f3f46] flex items-center justify-between">
          <div>
            <h2 className="text-xl font-black text-[#E50012] tracking-tighter uppercase italic">Realtime Source</h2>
            <div className="text-[10px] text-zinc-500 font-mono uppercase tracking-widest">{connectionState.toUpperCase()}</div>
          </div>
          <button onClick={onExit} className="p-2 text-zinc-400 hover:text-white hover:bg-zinc-800 border border-transparent hover:border-zinc-600">
            <X size={20} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto custom-scrollbar p-4 space-y-4">
          <label className="block space-y-1">
            <span className="text-[10px] text-zinc-500 font-black uppercase tracking-widest">Camera ID</span>
            <input value={cameraId} disabled={streaming} onChange={(e) => setCameraId(e.target.value || 'CAM_WS')} className="w-full bg-[#18181b] border border-[#3f3f46] px-3 py-2 text-sm font-mono text-zinc-100 outline-none focus:border-[#E50012]" />
          </label>

          <div className="grid grid-cols-2 gap-2">
            {(['real', 'mock'] as const).map(mode => (
              <button
                key={mode}
                disabled={streaming}
                onClick={() => setModelMode(mode)}
                className={`py-3 border text-xs font-black uppercase tracking-widest ${modelMode === mode ? 'bg-[#E50012] border-[#E50012] text-white' : 'bg-[#18181b] border-[#3f3f46] text-zinc-400 hover:text-white'}`}
              >
                {mode}
              </button>
            ))}
          </div>

          <label className="block space-y-1">
            <span className="text-[10px] text-zinc-500 font-black uppercase tracking-widest">Scenario</span>
            <select disabled={streaming} value={scenario} onChange={(e) => setScenario(e.target.value)} className="w-full bg-[#18181b] border border-[#3f3f46] px-3 py-2 text-sm font-mono text-zinc-100 outline-none focus:border-[#E50012]">
              <option value="normal_target2_accept">normal_target2_accept</option>
              <option value="single_pass_accept">single_pass_accept</option>
              <option value="unknown_no_pass">unknown_no_pass</option>
              <option value="ambiguous_reinspect">ambiguous_reinspect</option>
              <option value="poor_detector_reinspect">poor_detector_reinspect</option>
              <option value="no_detection">no_detection</option>
            </select>
          </label>

          <label className="block space-y-1">
            <span className="text-[10px] text-zinc-500 font-black uppercase tracking-widest">Target Order</span>
            <input disabled={streaming} value={targetOrder} onChange={(e) => setTargetOrder(e.target.value)} className="w-full bg-[#18181b] border border-[#3f3f46] px-3 py-2 text-sm font-mono text-zinc-100 outline-none focus:border-[#E50012]" />
          </label>

          <div className="grid grid-cols-2 gap-3">
            <label className="block space-y-1">
              <span className="text-[10px] text-zinc-500 font-black uppercase tracking-widest">Sample ms</span>
              <input type="number" min={50} max={2000} step={50} disabled={streaming} value={sampleMs} onChange={(e) => setSampleMs(Number(e.target.value) || 100)} className="w-full bg-[#18181b] border border-[#3f3f46] px-3 py-2 text-sm font-mono text-zinc-100 outline-none focus:border-[#E50012]" />
            </label>
            <label className="block space-y-1">
              <span className="text-[10px] text-zinc-500 font-black uppercase tracking-widest">Width</span>
              <input type="number" min={224} max={1920} step={64} disabled={streaming} value={frameWidth} onChange={(e) => setFrameWidth(Number(e.target.value) || 1280)} className="w-full bg-[#18181b] border border-[#3f3f46] px-3 py-2 text-sm font-mono text-zinc-100 outline-none focus:border-[#E50012]" />
            </label>
          </div>

          <label className="block space-y-2">
            <div className="flex justify-between text-[10px] text-zinc-500 font-black uppercase tracking-widest">
              <span>JPEG Quality</span>
              <span>{Math.round(jpegQuality * 100)}%</span>
            </div>
            <input type="range" min={0.5} max={0.95} step={0.05} disabled={streaming} value={jpegQuality} onChange={(e) => setJpegQuality(Number(e.target.value))} className="w-full accent-[#E50012]" />
          </label>

          <label className="flex items-center justify-between bg-[#18181b] border border-[#3f3f46] px-3 py-3">
            <span className="text-xs text-zinc-300 font-black uppercase tracking-widest">Save Artifacts</span>
            <input type="checkbox" checked={saveArtifacts} disabled={streaming} onChange={(e) => setSaveArtifacts(e.target.checked)} className="h-4 w-4 accent-[#E50012]" />
          </label>
        </div>

        <div className="p-4 border-t border-[#3f3f46] space-y-3">
          <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
            <div className="bg-[#18181b] border border-[#3f3f46] p-2 text-zinc-400">SENT <span className="float-right text-zinc-100">{stats.sent}</span></div>
            <div className="bg-[#18181b] border border-[#3f3f46] p-2 text-zinc-400">RESULT <span className="float-right text-zinc-100">{stats.received}</span></div>
            <div className="bg-[#18181b] border border-[#3f3f46] p-2 text-zinc-400">PRED <span className="float-right text-zinc-100">{stats.predicted}</span></div>
            <div className="bg-[#18181b] border border-[#3f3f46] p-2 text-zinc-400">FINAL <span className="float-right text-zinc-100">{stats.finalLabel}</span></div>
          </div>
          <div className={`border px-3 py-2 text-center text-xs font-black uppercase tracking-widest ${connectionState === 'error' ? 'border-[#ef4444] text-[#ef4444]' : streaming ? 'border-[#22c55e] text-[#22c55e]' : 'border-[#3f3f46] text-zinc-500'}`}>
            {stats.message}
          </div>
          <button
            onClick={toggleStreaming}
            className={`w-full py-4 text-base font-black tracking-widest uppercase transition-all shadow-xl active:scale-95 ${streaming ? 'bg-zinc-800 text-red-500 border border-red-500/30' : 'bg-[#E50012] text-white hover:bg-white hover:text-black'}`}
          >
            {streaming ? 'STOP STREAM' : 'START STREAM'}
          </button>
        </div>
      </div>

      <div className="relative bg-black border border-[#3f3f46] overflow-hidden min-h-0 flex items-center justify-center">
        <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-contain" />
        <canvas ref={canvasRef} className="hidden" />

        {streaming && (
          <div className="absolute top-4 right-4 flex items-center gap-2 px-3 py-1 bg-red-600 animate-pulse text-[10px] font-black text-white uppercase italic">
            <Activity size={10} /> ON-AIR
          </div>
        )}

        <div className="absolute left-4 bottom-4 bg-[#18181b]/90 border border-[#3f3f46] px-3 py-2 font-mono text-[10px] text-zinc-400">
          <div>{buildSourceWsUrl(cameraId, modelMode, scenario)}</div>
          <div>{frameWidth}px / {sampleMs}ms / {modelMode}</div>
        </div>
      </div>
    </div>
  );
}


// ─── 영상 분석 뷰 ──────────────────────────────────────────────────────────
const VIDEO_TARGETS = ['Target1', 'Target2', 'Target3', 'Target4'];

function VideoAnalysisModal() {
  const { videoAnalysis, clearVideoAnalysis, appendVideoFrameForCamera, finishVideoAnalysis } = useAppStore();
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const sentRef = useRef(0);
  const receivedRef = useRef(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [sampleMs, setSampleMs] = useState(200);
  const [received, setReceived] = useState(0);

  const cameraId = videoAnalysis?.cameraId ?? '';

  // ws connection (source) — cameraId가 바뀔 때만 재연결.
  // dep에 videoAnalysis 객체를 넣으면 매 frame 추가(set 호출)마다 새 ref가 되어
  // ws가 close→reconnect를 반복하게 되고, 그때마다 backend가 sequence state를
  // reset해서 3-프레임 누적이 시작 단계에 갇힘 — 절대 dep에 넣으면 안 됨.
  useEffect(() => {
    if (!cameraId) return;
    const wsBase = (typeof window !== 'undefined'
      ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws`
      : 'ws://localhost:8080/ws');
    const url = new URL(`${wsBase}/source`);
    url.searchParams.set('cameraId', cameraId);
    url.searchParams.set('model_mode', 'real');
    url.searchParams.set('scenario', 'normal_target2_accept');
    url.searchParams.set('strict_sequence', 'true');

    const ws = new WebSocket(url.toString());
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data as string);
        if (msg.type === 'inference' && msg.cameraId === cameraId && msg.payload) {
          const tNow = videoRef.current?.currentTime ?? 0;
          const idx = receivedRef.current;
          receivedRef.current = idx + 1;
          appendVideoFrameForCamera(cameraId, {
            frameIndex: idx,
            timeSec: tNow,
            payload: msg.payload as CameraData,
          });
          setReceived(idx + 1);
        }
      } catch { /* ignore */ }
    };
    ws.onerror = () => finishVideoAnalysis('source ws error');
    ws.onclose = () => { wsRef.current = null; };

    return () => {
      ws.onmessage = null;
      ws.onerror = null;
      ws.onclose = null;
      ws.close();
      wsRef.current = null;
    };
  }, [cameraId, appendVideoFrameForCamera, finishVideoAnalysis]);

  // canvas sampler — sampleMs/cameraId가 바뀔 때만 interval 재설정.
  useEffect(() => {
    if (!cameraId) return;
    const tick = () => {
      const ws = wsRef.current;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN || !video || !canvas) return;
      if (video.paused || video.ended || video.readyState < 2) return;
      const w = video.videoWidth;
      const h = video.videoHeight;
      if (!w || !h) return;
      const targetW = Math.min(1280, w);
      const targetH = Math.round(targetW * (h / w));
      canvas.width = targetW;
      canvas.height = targetH;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.drawImage(video, 0, 0, targetW, targetH);
      const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
      sentRef.current += 1;
      ws.send(JSON.stringify({
        cameraId,
        frame: dataUrl,
        model_mode: 'real',
        scenario: 'normal_target2_accept',
        strict_sequence: true,
        target_order: ['Target1', 'Target2', 'Target3', 'Target4'],
        reset_state: sentRef.current === 1,
      }));
    };
    intervalRef.current = setInterval(tick, Math.max(50, sampleMs));
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      intervalRef.current = null;
    };
  }, [cameraId, sampleMs]);

  if (!videoAnalysis) return null;
  const { filename, videoUrl, frames, status, error } = videoAnalysis;

  // currentTime에 가장 가까운 frame 찾기 (frames는 도착 순이라 별도 sort 필요 없음).
  // tolerance는 sampling 간격을 덮을 정도로 넉넉히 — 그래야 video 재생과 ws 응답 도착의
  // 작은 latency로 인해 강조 칸이 한 칸 뒤로 밀리는 깜빡임이 안 생김.
  const tolerance = Math.max(0.3, sampleMs / 1000);
  let currentFrame: VideoFrameResult | null = null;
  for (const f of frames) {
    if (f.timeSec <= currentTime + tolerance) currentFrame = f;
  }

  const logic = (currentFrame?.payload?.logic ?? {}) as Record<string, unknown>;
  const completed = (logic.completed_labels as string[] | undefined) ?? [];
  const expected = (logic.expected_label as string | null | undefined) ?? null;
  const effective = (logic.effective_label as string | null | undefined) ?? null;
  const allowedAtCurrent = logic.allowed_transition as boolean | undefined;

  const stepStatuses: StepStatus[] = VIDEO_TARGETS.map(target => {
    if (completed.includes(target)) return 'success';
    if (target === expected) {
      if (effective === target) return 'processing';
      if (allowedAtCurrent === false && effective && effective !== 'unknown') return 'error';
    }
    return 'idle';
  });

  const handleSeek = (timeSec: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = timeSec;
      setCurrentTime(timeSec);
    }
  };

  const handlePlaybackRate = (rate: number) => {
    setPlaybackRate(rate);
    if (videoRef.current) videoRef.current.playbackRate = rate;
  };

  const statusLabel = (() => {
    if (status === 'streaming') return `스트리밍 중 (보낸 ${sentRef.current} / 받은 ${received})`;
    if (status === 'done') return `완료 (${frames.length} frames)`;
    if (status === 'error') return `에러: ${error ?? ''}`;
    return '';
  })();

  return (
    <div className="fixed inset-0 bg-black/85 z-[160] flex items-center justify-center p-3">
      <div className="bg-[#18181b] border-2 border-[#3b82f6] w-[96vw] h-[94vh] flex flex-col shadow-[0_0_50px_rgba(59,130,246,0.2)]">
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#3f3f46] bg-[#27272a]">
          <div className="flex items-center gap-3">
            <div className="bg-[#3b82f6]/10 p-2 border border-[#3b82f6]/30">
              <Video size={20} className="text-[#3b82f6]" />
            </div>
            <div>
              <h2 className="text-lg font-black text-zinc-100 tracking-tighter uppercase leading-none">VIDEO ANALYSIS</h2>
              <p className="text-[10px] text-zinc-500 font-mono uppercase tracking-widest mt-1 truncate max-w-[60vw]">{filename}</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="text-xs font-mono text-zinc-300">{statusLabel}</div>
            <button
              onClick={clearVideoAnalysis}
              className="p-2 bg-zinc-800 hover:bg-zinc-700 border border-zinc-600 text-zinc-300 hover:text-white"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        <div className="flex-1 flex overflow-hidden">
          <div className="flex-1 flex flex-col items-center justify-center bg-black p-3 min-w-0 gap-3">
            <video
              ref={videoRef}
              src={videoUrl}
              controls
              autoPlay
              onTimeUpdate={(e) => setCurrentTime(e.currentTarget.currentTime)}
              onLoadedMetadata={(e) => { e.currentTarget.playbackRate = playbackRate; }}
              className="max-w-full max-h-full"
            />
            <canvas ref={canvasRef} className="hidden" />
            <div className="flex items-center gap-2 text-[10px] font-mono text-zinc-400">
              <span>SPEED</span>
              {[1, 2, 4, 8].map((r) => (
                <button
                  key={r}
                  onClick={() => handlePlaybackRate(r)}
                  className={`px-2 py-1 border ${playbackRate === r ? 'bg-[#3b82f6] border-[#3b82f6] text-white' : 'border-zinc-700 text-zinc-400 hover:border-zinc-500'}`}
                >×{r}</button>
              ))}
              <span className="ml-3">SAMPLE</span>
              <input
                type="number" min={50} max={1000} step={50} value={sampleMs}
                onChange={(e) => setSampleMs(Number(e.target.value) || 200)}
                className="w-16 bg-[#18181b] border border-zinc-700 px-1 py-1 text-zinc-200"
              />
              <span>ms</span>
            </div>
          </div>

          <div className="w-[380px] flex-shrink-0 border-l border-[#3f3f46] flex flex-col bg-[#18181b]">
            <div className="p-4 border-b border-[#3f3f46]">
              <h3 className="text-[10px] font-black text-zinc-500 tracking-widest uppercase mb-3">SEQUENCE STATE</h3>
              <div className="grid grid-cols-4 gap-2">
                {VIDEO_TARGETS.map((target, i) => {
                  const st = stepStatuses[i];
                  const colors =
                    st === 'success' ? 'bg-[#22c55e] text-white border-[#22c55e]' :
                    st === 'processing' ? 'bg-[#3b82f6] text-white border-[#3b82f6] animate-pulse' :
                    st === 'error' ? 'bg-[#E50012] text-white border-[#E50012]' :
                    'bg-[#27272a] text-zinc-500 border-zinc-700';
                  return (
                    <div key={target} className={`px-1 py-3 text-center font-black border-2 ${colors}`}>
                      <div className="text-xs">{target.replace('Target', 'T')}</div>
                      <div className="text-[8px] font-mono mt-1 opacity-90">
                        {st === 'success' ? 'OK' : st === 'processing' ? 'PROC' : st === 'error' ? 'BLK' : '·'}
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="mt-3 grid grid-cols-2 gap-2 text-[10px] font-mono">
                <div><span className="text-zinc-500">PRED</span> <span className="text-zinc-300">{String(currentFrame?.payload?.predicted_label ?? '-')}</span></div>
                <div><span className="text-zinc-500">FINAL</span> <span className="text-zinc-300">{String(currentFrame?.payload?.final_label ?? '-')}</span></div>
                <div><span className="text-zinc-500">EXPECT</span> <span className="text-zinc-300">{String(expected ?? '-')}</span></div>
                <div><span className="text-zinc-500">CONF</span> <span className="text-zinc-300">{(Number(currentFrame?.payload?.confidence ?? 0) * 100).toFixed(0)}%</span></div>
              </div>
            </div>

            <div className="flex-1 overflow-auto">
              <div className="text-[10px] font-black text-zinc-500 tracking-widest uppercase px-4 py-2 sticky top-0 bg-[#18181b] border-b border-[#3f3f46]">
                ANALYZED FRAMES ({frames.length})
              </div>
              {frames.length === 0 && (
                <div className="p-4 text-xs text-zinc-500 text-center">분석 시작 직후입니다. 잠시 후 첫 프레임이 표시됩니다.</div>
              )}
              {[...frames].reverse().map((f, i) => {
                const isCurrent = currentFrame === f;
                const fLogic = (f.payload?.logic ?? {}) as Record<string, unknown>;
                const eventType = String(fLogic.state_event_type ?? '-');
                const finalLabel = String(f.payload?.final_label ?? '-');
                const allowed = fLogic.allowed_transition as boolean | undefined;
                const isAnomaly = allowed === false || finalLabel === 'sequence_blocked' || finalLabel === 'unknown';
                return (
                  <button
                    key={i}
                    onClick={() => handleSeek(f.timeSec)}
                    className={`w-full px-4 py-2 flex items-center gap-2 text-left border-b border-[#27272a] hover:bg-[#27272a] transition-colors ${isCurrent ? 'bg-[#3b82f6]/15 border-l-2 border-l-[#3b82f6]' : ''}`}
                  >
                    <span className="font-mono text-[10px] text-zinc-500 w-12 tabular-nums">{f.timeSec.toFixed(2)}s</span>
                    <span className={`font-mono text-xs flex-1 truncate ${isAnomaly ? 'text-amber-400' : 'text-zinc-300'}`}>{finalLabel}</span>
                    <span className="font-mono text-[9px] text-zinc-500 uppercase">{eventType.replace(/_/g, ' ')}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


// ─── 이미지 분석 캐러셀 뷰 ─────────────────────────────────────────────────
function ImageAnalysisModal() {
  const { imageAnalysis, setImageAnalysisIndex, clearImageAnalysis } = useAppStore();

  // 키보드 좌/우 화살표로 캐러셀 이동
  useEffect(() => {
    if (!imageAnalysis) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft') setImageAnalysisIndex(imageAnalysis.currentIndex - 1);
      else if (e.key === 'ArrowRight') setImageAnalysisIndex(imageAnalysis.currentIndex + 1);
      else if (e.key === 'Escape') clearImageAnalysis();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [imageAnalysis, setImageAnalysisIndex, clearImageAnalysis]);

  if (!imageAnalysis) return null;
  const { items, currentIndex, status, error } = imageAnalysis;
  const total = items.length;
  const item = items[currentIndex];
  const result: ImageInspection | null | undefined = item?.result;
  const itemError = item?.error;

  const goPrev = () => setImageAnalysisIndex(currentIndex - 1);
  const goNext = () => setImageAnalysisIndex(currentIndex + 1);

  const labelChip = (label: string) => {
    const isUnknown = label === 'unknown' || label === 'sequence_blocked';
    return (
      <span className={`font-mono text-sm ${isUnknown ? 'text-amber-400' : 'text-zinc-100'}`}>{label || '-'}</span>
    );
  };

  const targets = ['Target1', 'Target2', 'Target3', 'Target4'];

  const statusLabel = (() => {
    if (status === 'analyzing') {
      const completed = items.filter((it) => it.result || it.error).length;
      return `분석 중 (${completed}/${total})`;
    }
    if (status === 'done') return `완료 (${total} images)`;
    if (status === 'error') return `에러: ${error ?? ''}`;
    return '';
  })();

  return (
    <div className="fixed inset-0 bg-black/85 z-[160] flex items-center justify-center p-3">
      <div className="bg-[#18181b] border-2 border-[#a855f7] w-[96vw] h-[94vh] flex flex-col shadow-[0_0_50px_rgba(168,85,247,0.2)]">
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#3f3f46] bg-[#27272a]">
          <div className="flex items-center gap-3">
            <div className="bg-[#a855f7]/10 p-2 border border-[#a855f7]/30">
              <ImageIcon size={20} className="text-[#a855f7]" />
            </div>
            <div>
              <h2 className="text-lg font-black text-zinc-100 tracking-tighter uppercase leading-none">IMAGE ANALYSIS</h2>
              <p className="text-[10px] text-zinc-500 font-mono uppercase tracking-widest mt-1 truncate max-w-[60vw]">
                {item?.filename ?? '-'} · {currentIndex + 1} / {total}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            <div className="text-xs font-mono text-zinc-300">{statusLabel}</div>
            <button
              onClick={clearImageAnalysis}
              className="p-2 bg-zinc-800 hover:bg-zinc-700 border border-zinc-600 text-zinc-300 hover:text-white"
            >
              <X size={18} />
            </button>
          </div>
        </div>

        <div className="flex-1 flex overflow-hidden">
          <div className="flex-1 flex items-center justify-center bg-black p-3 min-w-0 relative">
            {/* prev */}
            <button
              onClick={goPrev}
              disabled={currentIndex === 0}
              className="absolute left-3 top-1/2 -translate-y-1/2 p-3 bg-[#18181b]/80 hover:bg-[#27272a] border border-zinc-700 text-zinc-300 hover:text-white disabled:opacity-20 disabled:cursor-not-allowed z-10"
              aria-label="이전 이미지"
            >
              <ChevronLeft size={24} />
            </button>
            {item?.imageUrl ? (
              <img
                src={item.imageUrl}
                alt={item.filename}
                className="max-w-full max-h-full object-contain"
              />
            ) : (
              <div className="text-zinc-500 text-sm">이미지 없음</div>
            )}
            <button
              onClick={goNext}
              disabled={currentIndex >= total - 1}
              className="absolute right-3 top-1/2 -translate-y-1/2 p-3 bg-[#18181b]/80 hover:bg-[#27272a] border border-zinc-700 text-zinc-300 hover:text-white disabled:opacity-20 disabled:cursor-not-allowed z-10"
              aria-label="다음 이미지"
            >
              <ChevronRight size={24} />
            </button>
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 flex items-center gap-1 px-3 py-1 bg-[#18181b]/80 border border-zinc-700">
              {items.map((_, i) => (
                <button
                  key={i}
                  onClick={() => setImageAnalysisIndex(i)}
                  className={`w-2 h-2 ${i === currentIndex ? 'bg-[#a855f7]' : 'bg-zinc-600 hover:bg-zinc-500'}`}
                  aria-label={`${i + 1}번 이미지`}
                />
              ))}
            </div>
          </div>

          <div className="w-[380px] flex-shrink-0 border-l border-[#3f3f46] flex flex-col bg-[#18181b] overflow-auto">
            <div className="p-4 border-b border-[#3f3f46]">
              <h3 className="text-[10px] font-black text-zinc-500 tracking-widest uppercase mb-3">PREDICTION</h3>
              {!result && !itemError && (
                <div className="text-xs text-zinc-500 font-mono">분석 결과 대기 중...</div>
              )}
              {itemError && (
                <div className="text-xs text-amber-400 font-mono">에러: {itemError}</div>
              )}
              {result && (
                <div className="space-y-2">
                  <div className="grid grid-cols-2 gap-2 text-[10px] font-mono">
                    <div className="flex flex-col">
                      <span className="text-zinc-500 uppercase tracking-widest">Predicted</span>
                      {labelChip(result.predicted_label)}
                    </div>
                    <div className="flex flex-col">
                      <span className="text-zinc-500 uppercase tracking-widest">Final</span>
                      {labelChip(result.final_label)}
                    </div>
                    <div className="flex flex-col">
                      <span className="text-zinc-500 uppercase tracking-widest">Confidence</span>
                      <span className="font-mono text-sm text-zinc-100 tabular-nums">{(result.confidence * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex flex-col">
                      <span className="text-zinc-500 uppercase tracking-widest">Decision</span>
                      <span className="font-mono text-xs text-zinc-300">{result.decision_type ?? '-'}</span>
                    </div>
                  </div>
                  {(result.is_unknown || result.ambiguous || result.reinspect_needed || result.anomaly_flag) && (
                    <div className="flex flex-wrap gap-1 pt-1">
                      {result.is_unknown && <span className="px-2 py-0.5 text-[9px] font-black bg-zinc-700 text-zinc-200 uppercase tracking-widest">UNKNOWN</span>}
                      {result.ambiguous && <span className="px-2 py-0.5 text-[9px] font-black bg-amber-700 text-amber-100 uppercase tracking-widest">AMBIGUOUS</span>}
                      {result.reinspect_needed && <span className="px-2 py-0.5 text-[9px] font-black bg-yellow-700 text-yellow-100 uppercase tracking-widest">REINSPECT</span>}
                      {result.anomaly_flag && <span className="px-2 py-0.5 text-[9px] font-black bg-[#E50012] text-white uppercase tracking-widest">ALARM</span>}
                    </div>
                  )}
                  {result.decision_reason && (
                    <div className="text-[10px] font-mono text-zinc-500 pt-1">사유: {result.decision_reason}</div>
                  )}
                </div>
              )}
            </div>

            {result?.scores && (
              <div className="p-4 border-b border-[#3f3f46]">
                <h3 className="text-[10px] font-black text-zinc-500 tracking-widest uppercase mb-3">CLASSIFIER SCORES</h3>
                <div className="space-y-2">
                  {targets.map((t) => {
                    const score = Number(result.scores?.[t] ?? 0);
                    const threshold = Number(result.thresholds?.[t] ?? 0);
                    const passed = threshold > 0 && score >= threshold;
                    const pct = Math.round(score * 100);
                    return (
                      <div key={t}>
                        <div className="flex items-center justify-between text-[10px] font-mono mb-1">
                          <span className="text-zinc-300">{t}</span>
                          <span className={`tabular-nums ${passed ? 'text-[#22c55e]' : 'text-zinc-500'}`}>
                            {(score).toFixed(4)}{threshold > 0 && ` / ${threshold.toFixed(2)}`}
                          </span>
                        </div>
                        <div className="w-full h-1.5 bg-zinc-800 overflow-hidden">
                          <div
                            className={`h-full transition-all ${passed ? 'bg-[#22c55e]' : 'bg-zinc-500'}`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            <div className="px-4 py-3 text-[10px] font-mono text-zinc-500">
              ←/→ 키로도 이미지 전환 가능
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}


export default function App() {
  const {
    language, setSettingsOpen, liveData, updateLiveData, imageLogs,
    addImageLog, setAdminOpen, isAdminAuth, isDevAuth,
  } = useAppStore();
  const t = locales[language];
  const [wsConnected, setWsConnected] = useState(false);
  const [activeItems, setActiveItems] = useState<ViewItem[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);
  const [detailItem, setDetailItem] = useState<ViewItem | null>(null);
  const [sourceMode, setSourceMode] = useState(false);
  const [onlineCameraIds, setOnlineCameraIds] = useState<string[]>([]);
  // 그리드 내 드래그-리오더용 상태
  const [dragSrcKey, setDragSrcKey] = useState<string | null>(null);
  const [dragOverKey, setDragOverKey] = useState<string | null>(null);

  const itemKey = (item: ViewItem) => `${item.type}:${item.id}`;

  const clickCountRef = useRef(0);
  const clickTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleLogoClick = () => {
    clickCountRef.current += 1;
    if (clickCountRef.current >= 3) {
      setAdminOpen(true); // 항상 비번 입력창을 띄움
      clickCountRef.current = 0;
    }
    if (clickTimerRef.current) clearTimeout(clickTimerRef.current);
    clickTimerRef.current = setTimeout(() => { clickCountRef.current = 0; }, 1000);
  };

  // 더블탭으로 추가/제거, 싱글탭으로 단일 전환
  const lastTapRef = useRef<Record<string, number>>({});
  const tapTimerRef = useRef<Record<string, ReturnType<typeof setTimeout>>>({});
  const handleItemTap = useCallback((item: ViewItem) => {
    const key = itemKey(item);
    const now = Date.now();
    const last = lastTapRef.current[key] || 0;
    lastTapRef.current[key] = now;
    if (now - last < 300) {
      if (tapTimerRef.current[key]) { clearTimeout(tapTimerRef.current[key]); delete tapTimerRef.current[key]; }
      const exists = activeItems.some(i => i.type === item.type && i.id === item.id);
      if (exists) { if (activeItems.length > 1) setActiveItems(activeItems.filter(i => !(i.type === item.type && i.id === item.id))); }
      else { setActiveItems(prev => [...prev, item]); }
    } else {
      if (tapTimerRef.current[key]) clearTimeout(tapTimerRef.current[key]);
      tapTimerRef.current[key] = setTimeout(() => { delete tapTimerRef.current[key]; setActiveItems([item]); }, 300);
    }
  }, [activeItems]);

  // ── 사이드바에서 중앙으로 드래그-드롭 (새 아이템 추가) ──
  const handleDragStart = (e: React.DragEvent, item: ViewItem) => {
    e.dataTransfer.setData("viewItem", JSON.stringify(item));
    e.dataTransfer.setData("source", "sidebar");
  };
  const handleDragOver = (e: React.DragEvent) => { e.preventDefault(); if (!dragSrcKey) setIsDragOver(true); };
  const handleDragLeave = (e: React.DragEvent) => { e.preventDefault(); setIsDragOver(false); };
  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault(); setIsDragOver(false);
    const source = e.dataTransfer.getData("source");
    if (source !== "sidebar") return; // 패널 리오더는 패널 자체 핸들러에서 처리
    try {
      const item: ViewItem = JSON.parse(e.dataTransfer.getData("viewItem"));
      if (!activeItems.some(i => i.type === item.type && i.id === item.id))
        setActiveItems(prev => [...prev, item]);
    } catch { /* ignore */ }
  };
  const removeSplitItem = (item: ViewItem) => {
    if (activeItems.length > 1) setActiveItems(activeItems.filter(i => !(i.type === item.type && i.id === item.id)));
  };

  // ── 그리드 내 패널 드래그-리오더 핸들러 ──
  const handlePanelDragStart = (e: React.DragEvent, item: ViewItem) => {
    e.dataTransfer.setData("viewItem", JSON.stringify(item));
    e.dataTransfer.setData("source", "grid");
    e.dataTransfer.effectAllowed = "move";
    setDragSrcKey(itemKey(item));
  };
  const handlePanelDragOver = (e: React.DragEvent, item: ViewItem) => {
    e.preventDefault();
    if (dragSrcKey) {
      e.dataTransfer.dropEffect = "move";
      setDragOverKey(itemKey(item));
    }
  };
  const handlePanelDragLeave = () => setDragOverKey(null);
  const handlePanelDrop = (e: React.DragEvent, targetItem: ViewItem) => {
    e.preventDefault();
    const source = e.dataTransfer.getData("source");
    if (source !== "grid") {
      setDragOverKey(null);
      return; // 이벤트 버블링을 허용하여 부모 컨테이너의 handleDrop이 실행되게 함
    }
    e.stopPropagation();
    setDragOverKey(null);
    setDragSrcKey(null);
    setIsDragOver(false);
    try {
      const srcItem: ViewItem = JSON.parse(e.dataTransfer.getData("viewItem"));
      const srcKey = itemKey(srcItem);
      const tgtKey = itemKey(targetItem);
      if (srcKey === tgtKey) return;
      setActiveItems(prev => {
        const arr = [...prev];
        const srcIdx = arr.findIndex(i => itemKey(i) === srcKey);
        const tgtIdx = arr.findIndex(i => itemKey(i) === tgtKey);
        if (srcIdx === -1 || tgtIdx === -1) return prev;
        arr.splice(srcIdx, 1);
        arr.splice(tgtIdx, 0, srcItem);
        return arr;
      });
    } catch { /* ignore */ }
  };
  const handlePanelDragEnd = () => { setDragSrcKey(null); setDragOverKey(null); setIsDragOver(false); };

  // WebSocket — popup img도 동시 업데이트 (백엔드 reload 등으로 끊기면 자동 재연결)
  useEffect(() => {
    let cancelled = false;
    let ws: WebSocket | null = null;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;

    const handleMessage = (event: MessageEvent) => {
      try {
        const batch = JSON.parse(event.data as string);
        if (batch.type === 'camera_list') {
          const cams = (batch.cameras as string[]).filter((id) => !id.startsWith('VIDEO_'));
          setOnlineCameraIds(cams);
          return;
        }
        if (batch.type === 'video_frame') {
          const camId = batch.cameraId as string;
          if (camId?.startsWith('VIDEO_')) return; // VIDEO_*는 운영 그리드에 그리지 않음
          const src = `data:image/jpeg;base64,${batch.frame as string}`;
          ['', 'popup-'].forEach((prefix) => {
            const el = document.getElementById(`video-stream-${prefix}${camId}`) as HTMLImageElement | null;
            if (el) { el.src = src; el.classList.remove('hidden'); }
          });
          return;
        }
        if (batch.type === 'inference' && batch.cameraId) {
          if ((batch.cameraId as string).startsWith('VIDEO_')) return; // modal이 직접 처리
          updateLiveData(batch.cameraId as string, batch.payload as CameraData);
          return;
        }
        if (Array.isArray(batch)) {
          (batch as Array<{ type?: string; payload?: unknown; cameraId?: string }>).forEach((item) => {
            if (item.type === 'image_log') addImageLog(item.payload as ImageLog);
            else if (item.cameraId && !item.cameraId.startsWith('VIDEO_')) updateLiveData(item.cameraId, item.payload as CameraData);
          });
        }
      } catch { /* ignore */ }
    };

    const connect = () => {
      if (cancelled) return;
      ws = new WebSocket(DASHBOARD_WS_URL);
      ws.onopen = () => setWsConnected(true);
      ws.onmessage = handleMessage;
      ws.onclose = () => {
        setWsConnected(false);
        if (cancelled) return;
        reconnectTimer = setTimeout(connect, 1500);
      };
      ws.onerror = () => { ws?.close(); };
    };

    connect();

    return () => {
      cancelled = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws) {
        ws.onclose = null;
        ws.onerror = null;
        ws.close();
      }
    };
  }, [updateLiveData, addImageLog]);

  // 서버에서 온라인으로 알림이 온 카메라만 사이드바에 표시
  const CAMERA_LIST = [
    ...(isDevAuth ? [{ id: 'CAM_00', name: 'Dev Portal', latestData: { predicted_label: 'easter_egg', confidence: 1.0, is_unknown: false } }] : []),
    ...onlineCameraIds.map(id => ({
      id,
      name: id.replace('_', ' '),
      latestData: liveData[id] ?? { predicted_label: 'connecting', confidence: 0, is_unknown: false }
    }))
  ];

  const calcSteps = (camData: CameraData | null): StepStatus[] => {
    const steps: StepStatus[] = ['idle', 'idle', 'idle', 'idle'];
    if (!camData?.logic) return steps;
    const logic = camData.logic as Record<string, unknown>;
    const completed = (logic.completed_labels as string[] | undefined) ?? [];
    const expected = (logic.expected_label as string | null | undefined) ?? null;
    const effective = (logic.effective_label as string | null | undefined) ?? null;
    const allowed = logic.allowed_transition as boolean | undefined;
    const targets = ['Target1', 'Target2', 'Target3', 'Target4'];
    for (let i = 0; i < 4; i++) {
      const t = targets[i];
      if (completed.includes(t)) {
        steps[i] = 'success';
      } else if (t === expected) {
        if (effective === expected) steps[i] = 'processing';
        else if (allowed === false && effective && effective !== 'unknown') steps[i] = 'error';
        else steps[i] = 'idle';
      }
    }
    return steps;
  };

  const renderPanel = (item: ViewItem, onRemove?: () => void, isSplit?: boolean) => {
    const openDetail = () => setDetailItem(item);
    if (item.type === 'camera') {
      const cam = CAMERA_LIST.find(c => c.id === item.id);
      if (!cam) return null;
      
      const camData = liveData[cam.id] ?? cam.latestData;
      return <CameraPanel key={`${item.type}-${item.id}`} cameraId={cam.id} title={cam.name} stepStatuses={calcSteps(camData)} latestData={camData} onRemove={onRemove} isSplit={isSplit} onOpenDetail={openDetail} />;
    } else {
      const img = imageLogs.find(i => i.id === item.id);
      if (!img) return null;
      return <ImagePanel key={`${item.type}-${item.id}`} img={img} onRemove={onRemove} isSplit={isSplit} onOpenDetail={openDetail} />;
    }
  };

  // 패널 수에 따라 열 수 자동 계산: 1→1열, 2~4→2열, 5~9→3열, 10+→4열
  const colCount = activeItems.length <= 1 ? 1 : activeItems.length <= 4 ? 2 : activeItems.length <= 9 ? 3 : 4;
  const gridColsClass = colCount === 1 ? 'grid-cols-1' : colCount === 2 ? 'grid-cols-2' : colCount === 3 ? 'grid-cols-3' : 'grid-cols-4';
  const isItemActive = (item: ViewItem) => activeItems.some(i => i.type === item.type && i.id === item.id);

  return (
    <div className="flex flex-col h-screen w-full bg-[#18181b] text-zinc-200">
      {sourceMode && <MobileSourceView onExit={() => setSourceMode(false)} />}
      <SettingsModal />
      <AdminModal />
      <TestModeModal />
      <VideoAnalysisModal />
      <ImageAnalysisModal />
      {detailItem && (
        <DetailModal
          item={detailItem}
          onClose={() => setDetailItem(null)}
          liveData={liveData}
          CAMERA_LIST={CAMERA_LIST}
          imageLogs={imageLogs}
        />
      )}

      {/* 헤더 */}
      <header className="flex items-center justify-between h-14 px-4 bg-[#27272a] border-b-2 border-[#E50012] shrink-0">
        <div className="flex items-center gap-4">
          <img src="/Canon_icon.svg" alt="Canon Logo" width={100} height={24}
            className="object-contain cursor-pointer hover:opacity-80 transition-opacity"
            onClick={handleLogoClick}
          />
          <span className="text-sm font-bold tracking-wider text-zinc-400 border-l border-zinc-600 pl-4">{t.title}</span>
        </div>
        <div className="flex items-center gap-6">
          {isAdminAuth && (
            <div className="flex items-center gap-2 px-3 py-1 bg-[#E50012] border border-[#ff3040] text-white font-black text-[10px] tracking-widest shadow-[0_0_10px_rgba(229,0,18,0.5)] animate-pulse">
              <ShieldAlert size={12} />
              ADMIN ACTIVE
            </div>
          )}
          <div className="flex items-center gap-3">
            {wsConnected ? (
              <>
                <span className="flex h-2.5 w-2.5 relative"><span className="animate-ping absolute inline-flex h-full w-full bg-[#22c55e] opacity-75"></span><span className="relative inline-flex h-2.5 w-2.5 bg-[#22c55e]"></span></span>
                <span className="text-xs font-medium text-zinc-300 hidden sm:block">LIVE</span>
                <LivenessIndicator />
              </>
            ) : (
              <><Server size={13} className="text-zinc-500" /><span className="text-xs text-zinc-500">OFFLINE</span></>
            )}
          </div>
          <button onClick={() => setSettingsOpen(true)} className="p-2 bg-[#18181b] hover:bg-[#3f3f46] border border-[#3f3f46] hover:border-zinc-400 text-zinc-300 hover:text-white transition-colors">
            <Settings size={18} />
          </button>
          <button 
            onClick={() => setSourceMode(true)} 
            className="p-2 bg-[#18181b] hover:bg-[#3f3f46] border border-[#3f3f46] hover:border-zinc-400 text-zinc-300 hover:text-white transition-colors flex items-center gap-2 group"
          >
            <Video size={18} className="group-hover:text-[#E50012]" />
            <span className="text-[10px] font-black hidden lg:block">STREAM SOURCE</span>
          </button>
        </div>
      </header>

      {/* 메인 */}
      <main className="flex-1 p-2 flex gap-2 overflow-hidden relative">

        {/* ── 좌측 사이드바 ── CCTV 스타일 */}
        <aside className="w-52 flex flex-col gap-2 shrink-0 h-full">

          {/* 카메라 목록 */}
          <div className="flex-none bg-[#27272a] border border-[#3f3f46] flex flex-col shadow-sm">
            <div className="flex items-center gap-2 px-3 py-2 border-b border-[#3f3f46] shrink-0">
              <Video size={14} className="text-zinc-400" />
              <span className="text-xs font-bold text-zinc-300 uppercase tracking-widest">{t.videoSources}</span>
            </div>
            <div className="overflow-y-auto custom-scrollbar">
              {CAMERA_LIST.map((cam) => {
                const item: ViewItem = { type: 'camera', id: cam.id };
                const active = isItemActive(item);
                return (
                  <div
                    key={cam.id}
                    onClick={() => {
                      const item: ViewItem = { type: 'camera', id: cam.id };
                      const exists = activeItems.some(i => i.type === item.type && i.id === item.id);
                      if (exists) {
                        if (activeItems.length > 1) setActiveItems(activeItems.filter(i => !(i.type === item.type && i.id === item.id)));
                      } else {
                        setActiveItems([item]); // 싱글 클릭으로 즉시 전환
                      }
                    }}
                    draggable onDragStart={(e) => handleDragStart(e, item)}
                    className={`px-3 py-2 border-b border-[#3f3f46] cursor-pointer transition-all flex items-center justify-between select-none ${active ? 'bg-[#E50012]/15 border-l-2 border-l-[#E50012]' : 'hover:bg-[#3f3f46]'}`}
                  >
                    <div className="flex items-center gap-2 min-w-0">
                      <div className={`w-1.5 h-1.5 shrink-0 ${active ? 'bg-[#E50012]' : 'bg-zinc-600'}`} />
                      <div className="min-w-0">
                        <div className="text-xs font-bold text-zinc-200">{cam.id}</div>
                        <div className="text-[10px] text-zinc-500 truncate">{cam.name.includes(' - ') ? cam.name.split(' - ')[1] : cam.name}</div>
                      </div>
                    </div>
                    {active && <Camera size={12} className="text-[#E50012] shrink-0" />}
                  </div>
                );
              })}
            </div>
          </div>

          {/* 이미지 로그 목록 */}
          <div className="flex-1 bg-[#27272a] border border-[#3f3f46] flex flex-col min-h-0 shadow-sm">
            <div className="flex items-center gap-2 px-3 py-2 border-b border-[#3f3f46] shrink-0">
              <ImageIcon size={14} className="text-zinc-400" />
              <span className="text-xs font-bold text-zinc-300 uppercase tracking-widest">{t.imageLogs}</span>
              <span className="ml-auto text-[10px] text-zinc-600 font-mono">{imageLogs.length}</span>
            </div>
            <div className="flex-1 overflow-y-auto custom-scrollbar">
              {imageLogs.length === 0 ? (
                <div className="px-3 py-4 text-center text-zinc-600 font-mono text-[10px]">NO LOGS YET</div>
              ) : (
                imageLogs.map((img) => {
                  const item: ViewItem = { type: 'image', id: img.id };
                  const active = isItemActive(item);
                  const sc = img.status === 'success' ? '#22c55e' : img.status === 'error' ? '#ef4444' : '#f59e0b';
                  return (
                    <div
                      key={img.id}
                      onClick={() => handleItemTap(item)}
                      draggable onDragStart={(e) => handleDragStart(e, item)}
                      className={`px-2 py-1.5 border-b border-[#3f3f46] flex items-center gap-2 cursor-pointer transition-colors select-none ${active ? 'bg-[#E50012]/15 border-l-2 border-l-[#E50012]' : 'hover:bg-[#3f3f46]'}`}
                    >
                      <div className="w-8 h-8 bg-zinc-900 border border-zinc-700 shrink-0 overflow-hidden">
                        {img.image_url
                          ? <img src={img.image_url} alt={img.id} className="w-full h-full object-cover" />
                          : <ImageIcon size={14} className="text-zinc-600 m-auto mt-1.5" />}
                      </div>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-1.5">
                          <span className="text-[10px] font-bold text-zinc-200 truncate">{img.id}</span>
                          <span className="text-[8px] font-bold px-1 border" style={{ color: sc, borderColor: sc }}>T{img.target_idx}</span>
                        </div>
                        <span className="text-[9px] text-zinc-500 font-mono">{img.cam} · {img.time}</span>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </aside>

        {/* ── 중앙 뷰어 ── */}
        <div
          className={`flex-1 flex flex-col min-h-0 relative transition-all duration-200 ${isDragOver ? 'ring-2 ring-[#22c55e] ring-offset-2 ring-offset-[#18181b]' : ''}`}
          onDrop={handleDrop} onDragOver={handleDragOver} onDragLeave={handleDragLeave}
        >
          {isDragOver && (
            <div className="absolute inset-0 z-50 bg-[#22c55e]/10 flex items-center justify-center pointer-events-none">
              <div className="bg-[#27272a] border border-[#22c55e] text-[#22c55e] px-8 py-4 text-lg font-bold tracking-widest flex items-center gap-3">
                <BoxSelect size={28} />{t.dropToSplit}
              </div>
            </div>
          )}
          <div className={`flex-1 grid ${gridColsClass} gap-1.5 min-h-0`} style={{ gridAutoRows: '1fr' }}>
            {activeItems.map((item) => {
              const key = itemKey(item);
              const isDraggingSrc = dragSrcKey === key;
              const isDraggingOver = dragOverKey === key && dragOverKey !== dragSrcKey;
              return (
                <div
                  key={key}
                  draggable
                  onDragStart={(e) => handlePanelDragStart(e, item)}
                  onDragOver={(e) => handlePanelDragOver(e, item)}
                  onDragLeave={handlePanelDragLeave}
                  onDrop={(e) => handlePanelDrop(e, item)}
                  onDragEnd={handlePanelDragEnd}
                  className={`min-h-0 transition-all duration-150 ${
                    isDraggingSrc ? 'opacity-40 scale-[0.97]' : ''
                  } ${
                    isDraggingOver ? 'ring-2 ring-[#22c55e] ring-inset' : ''
                  }`}
                  style={{ cursor: 'grab' }}
                >
                  {renderPanel(item, () => removeSplitItem(item), activeItems.length > 1)}
                </div>
              );
            })}
          </div>
        </div>
      </main>
    </div>
  );
}
