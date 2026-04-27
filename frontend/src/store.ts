import { create } from 'zustand';
import type { Language, CameraData, ImageLog } from './types';

export type { Language, CameraData, ImageLog } from './types';

export interface VideoFrameResult {
  frameIndex: number;
  timeSec: number;
  payload: CameraData;
}

export interface VideoAnalysis {
  jobId: string;
  cameraId: string;
  filename: string;
  videoUrl: string;
  frames: VideoFrameResult[];
  status: 'streaming' | 'done' | 'error';
  error?: string;
}

export interface ImageInspection {
  predicted_label: string;
  final_label: string;
  confidence: number;
  decision_type?: string;
  scores?: Record<string, number>;
  thresholds?: Record<string, number>;
  anomaly_flag?: boolean;
  decision_reason?: string;
  is_unknown?: boolean;
  ambiguous?: boolean;
  reinspect_needed?: boolean;
}

export interface ImageItem {
  filename: string;
  imageUrl: string;
  result?: ImageInspection | null;
  error?: string;
}

export interface ImageAnalysis {
  items: ImageItem[];
  currentIndex: number;
  status: 'analyzing' | 'done' | 'error';
  error?: string;
}

interface AppState {
  language: Language;
  setLanguage: (lang: Language) => void;
  isSettingsOpen: boolean;
  setSettingsOpen: (isOpen: boolean) => void;
  isAdminOpen: boolean;
  setAdminOpen: (isOpen: boolean) => void;
  isAdminAuth: boolean;
  setAdminAuth: (isAuth: boolean) => void;
  isDevAuth: boolean;
  setDevAuth: (isAuth: boolean) => void;
  isTestModeOpen: boolean;
  setTestModeOpen: (isOpen: boolean) => void;
  dbLogs: any[];
  setDbLogs: (logs: any[]) => void;
  liveData: Record<string, CameraData>;
  updateLiveData: (cameraId: string, data: CameraData) => void;
  imageLogs: ImageLog[];
  addImageLog: (log: ImageLog) => void;
  videoAnalysis: VideoAnalysis | null;
  startVideoAnalysis: (init: Omit<VideoAnalysis, 'frames' | 'status'>) => void;
  appendVideoFrameForCamera: (cameraId: string, frame: VideoFrameResult) => void;
  finishVideoAnalysis: (error?: string) => void;
  clearVideoAnalysis: () => void;
  imageAnalysis: ImageAnalysis | null;
  startImageAnalysis: (items: ImageItem[]) => void;
  setImageAnalysisIndex: (idx: number) => void;
  applyImageInspections: (results: (ImageInspection | { error: string })[]) => void;
  setImageAnalysisError: (error: string) => void;
  clearImageAnalysis: () => void;
}

export const useAppStore = create<AppState>((set) => ({
  language: 'ko',
  setLanguage: (lang) => set({ language: lang }),
  isSettingsOpen: false,
  setSettingsOpen: (isOpen) => set({ isSettingsOpen: isOpen }),
  isAdminOpen: false,
  setAdminOpen: (isOpen) => set({ isAdminOpen: isOpen }),
  isAdminAuth: false,
  setAdminAuth: (isAuth) => set({ isAdminAuth: isAuth }),
  isDevAuth: false,
  setDevAuth: (isAuth) => set({ isDevAuth: isAuth }),
  isTestModeOpen: false,
  setTestModeOpen: (isOpen) => set({ isTestModeOpen: isOpen }),
  dbLogs: [],
  setDbLogs: (logs) => set({ dbLogs: logs }),
  liveData: {},
  updateLiveData: (cameraId, data) => set((state) => ({
    liveData: { ...state.liveData, [cameraId]: data }
  })),
  imageLogs: [],
  addImageLog: (log) => set((state) => ({
    imageLogs: [log, ...state.imageLogs].slice(0, 50)
  })),
  videoAnalysis: null,
  startVideoAnalysis: (init) => set(() => ({
    videoAnalysis: { ...init, frames: [], status: 'streaming' },
  })),
  appendVideoFrameForCamera: (cameraId, frame) => set((state) => {
    if (!state.videoAnalysis || state.videoAnalysis.cameraId !== cameraId) return {};
    const current = state.videoAnalysis;
    return {
      videoAnalysis: {
        ...current,
        frames: [...current.frames, frame],
      },
    };
  }),
  finishVideoAnalysis: (error) => set((state) => {
    if (!state.videoAnalysis) return {};
    return {
      videoAnalysis: { ...state.videoAnalysis, status: error ? 'error' : 'done', error },
    };
  }),
  clearVideoAnalysis: () => set((state) => {
    if (state.videoAnalysis?.videoUrl) {
      try { URL.revokeObjectURL(state.videoAnalysis.videoUrl); } catch { /* ignore */ }
    }
    return { videoAnalysis: null };
  }),
  imageAnalysis: null,
  startImageAnalysis: (items) => set(() => ({
    imageAnalysis: { items, currentIndex: 0, status: 'analyzing' },
  })),
  setImageAnalysisIndex: (idx) => set((state) => {
    if (!state.imageAnalysis) return {};
    const max = state.imageAnalysis.items.length - 1;
    const clamped = Math.max(0, Math.min(max, idx));
    return { imageAnalysis: { ...state.imageAnalysis, currentIndex: clamped } };
  }),
  applyImageInspections: (results) => set((state) => {
    if (!state.imageAnalysis) return {};
    const items = state.imageAnalysis.items.map((item, i) => {
      const r = results[i];
      if (!r) return item;
      if ('error' in r) return { ...item, error: r.error, result: null };
      return { ...item, result: r };
    });
    return { imageAnalysis: { ...state.imageAnalysis, items, status: 'done' } };
  }),
  setImageAnalysisError: (error) => set((state) => {
    if (!state.imageAnalysis) return {};
    return { imageAnalysis: { ...state.imageAnalysis, status: 'error', error } };
  }),
  clearImageAnalysis: () => set((state) => {
    state.imageAnalysis?.items.forEach((item) => {
      if (item.imageUrl) { try { URL.revokeObjectURL(item.imageUrl); } catch { /* ignore */ } }
    });
    return { imageAnalysis: null };
  }),
}));
