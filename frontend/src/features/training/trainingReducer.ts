import type { Hyperparameters } from "../../api/client";
import type { TrainingEvent } from "../../api/sse";

export const LOSS_KEYS = ["total_loss", "iou_loss", "conf_loss", "cls_loss", "l1_loss"] as const;
export type LossKey = (typeof LOSS_KEYS)[number];

export type LossPoint = { global_iter: number } & Record<LossKey, number | null>;
export interface LrPoint {
  global_iter: number;
  lr: number | null;
}
export interface ApPoint {
  epoch: number;
  ap50: number | null;
  ap50_95: number | null;
}

export interface TrainingState {
  lastSeq: number;
  state: string;
  datasetName: string;
  hyperparameters: Hyperparameters;
  maxEpoch: number;
  epoch: number;
  globalIter: number;
  totalIters: number;
  etaSeconds: number | null;
  lossSeries: LossPoint[];
  lrSeries: LrPoint[];
  evalSeries: ApPoint[];
  modelId: string;
  error: string;
}

export type TrainingAction = TrainingEvent | { type: "reset" };

export const initialTrainingState: TrainingState = {
  lastSeq: 0,
  state: "STARTING",
  datasetName: "",
  hyperparameters: {},
  maxEpoch: 0,
  epoch: 0,
  globalIter: 0,
  totalIters: 0,
  etaSeconds: null,
  lossSeries: [],
  lrSeries: [],
  evalSeries: [],
  modelId: "",
  error: "",
};

export const ACTIVE_STATES = ["STARTING", "RUNNING", "STOPPING"];

export function trainingReducer(state: TrainingState, action: TrainingAction): TrainingState {
  if (action.type === "reset") return initialTrainingState;
  if (action.seq <= state.lastSeq) return state;
  const next = { ...state, lastSeq: action.seq };

  switch (action.type) {
    case "started":
      return {
        ...next,
        state: "RUNNING",
        datasetName: action.dataset_name,
        hyperparameters: action.hyperparameters,
        maxEpoch: action.max_epoch,
        totalIters: action.max_epoch * action.iters_per_epoch,
      };
    case "iter": {
      const point = { global_iter: action.global_iter } as LossPoint;
      for (const key of LOSS_KEYS) point[key] = action.losses[key] ?? null;
      return {
        ...next,
        epoch: action.epoch,
        globalIter: action.global_iter,
        totalIters: action.total_iters,
        etaSeconds: action.eta_seconds,
        lossSeries: [...state.lossSeries, point],
        lrSeries: [...state.lrSeries, { global_iter: action.global_iter, lr: action.lr }],
      };
    }
    case "eval":
      return {
        ...next,
        evalSeries: [...state.evalSeries, { epoch: action.epoch, ap50: action.ap50, ap50_95: action.ap50_95 }],
      };
    case "finished":
      return { ...next, state: action.state, modelId: action.model_id, error: action.error, etaSeconds: null };
  }
}
