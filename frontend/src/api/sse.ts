import type { Hyperparameters } from "./client";

interface EventBase {
  seq: number;
  time_unix_ms: number;
}

export type TrainingEvent =
  | ({
      type: "started";
      dataset_name: string;
      hyperparameters: Hyperparameters;
      max_epoch: number;
      iters_per_epoch: number;
    } & EventBase)
  | ({
      type: "iter";
      epoch: number;
      iter: number;
      global_iter: number;
      total_iters: number;
      lr: number | null;
      losses: Record<string, number | null>;
      input_size: number;
      eta_seconds: number | null;
    } & EventBase)
  | ({
      type: "eval";
      epoch: number;
      ap50: number | null;
      ap50_95: number | null;
      best_ap50_95: number | null;
    } & EventBase)
  | ({
      type: "finished";
      state: string;
      model_id: string;
      error: string;
    } & EventBase);

const EVENT_TYPES = ["started", "iter", "eval", "finished"] as const;

/**
 * Streams training events starting after `afterSeq`. The browser resends
 * `Last-Event-ID` on automatic reconnects, so the stream resumes without gaps.
 * The stream closes after `finished`; the returned function closes it early.
 */
export function subscribeTrainingEvents(
  jobId: string,
  afterSeq: number,
  onEvent: (event: TrainingEvent) => void,
  onError: (error: Event) => void,
): () => void {
  const url = `/api/trainings/${encodeURIComponent(jobId)}/events?after_seq=${afterSeq}`;
  const source = new EventSource(url);

  for (const type of EVENT_TYPES) {
    source.addEventListener(type, (message) => {
      const data = JSON.parse((message as MessageEvent<string>).data) as Omit<TrainingEvent, "type">;
      if (type === "finished") source.close();
      onEvent({ type, ...data } as TrainingEvent);
    });
  }
  source.onerror = onError;

  return () => source.close();
}
