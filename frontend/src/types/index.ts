export type Language = 'en' | 'ko';

export type StepStatus = 'idle' | 'processing' | 'success' | 'error';

export type ViewItem = 
  | { type: 'camera'; id: string } 
  | { type: 'image'; id: string };

export interface CameraData {
  timestamp?: string;
  predicted_label?: string;
  final_label?: string;
  confidence?: number;
  effective_label?: string;
  confirmed_state?: string;
  allowed_transition?: boolean;
  inference?: boolean;
  is_unknown?: boolean;
  ambiguous?: boolean;
  reinspect_needed?: boolean;
  logic?: Record<string, unknown>;
  display?: Record<string, string>;
}

export interface ImageLog {
  id: string;
  cam: string;
  time: string;
  image_url?: string;
  target_idx: number;
  status: 'success' | 'error' | 'processing';
  reason?: string;
  [key: string]: unknown;
}

export interface CameraListItem {
  id: string;
  name: string;
  latestData: CameraData;
}
