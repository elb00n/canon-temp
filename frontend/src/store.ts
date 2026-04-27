import { create } from 'zustand';
import type { Language, CameraData, ImageLog } from './types';

export type { Language, CameraData, ImageLog } from './types';

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
  }))
}));
