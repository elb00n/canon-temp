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
}));
