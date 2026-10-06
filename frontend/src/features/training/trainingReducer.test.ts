import { describe, expect, it } from "vitest";
import type { TrainingEvent } from "../../api/sse";
import { initialTrainingState, trainingReducer } from "./trainingReducer";

const base = { time_unix_ms: 0 };

const started: TrainingEvent = {
  type: "started",
  seq: 1,
  ...base,
  dataset_name: "cats",
  hyperparameters: { max_epoch: 2 },
  max_epoch: 2,
  iters_per_epoch: 2,
};

function iter(seq: number, globalIter: number, total: number | null, eta: number | null = 10): TrainingEvent {
  return {
    type: "iter",
    seq,
    ...base,
    epoch: 1,
    iter: globalIter,
    global_iter: globalIter,
    total_iters: 4,
    lr: 0.001 * globalIter,
    losses: { total_loss: total, iou_loss: 1, conf_loss: 2, cls_loss: 3, l1_loss: 0 },
    input_size: 416,
    eta_seconds: eta,
  };
}

const evalEvent: TrainingEvent = {
  type: "eval",
  seq: 4,
  ...base,
  epoch: 1,
  ap50: 0.5,
  ap50_95: 0.25,
  best_ap50_95: 0.25,
};

function run(events: TrainingEvent[]) {
  return events.reduce(trainingReducer, initialTrainingState);
}

describe("trainingReducer", () => {
  it("builds meta, series and progress from a scripted sequence", () => {
    const state = run([started, iter(2, 1, 5), iter(3, 2, null, 6), evalEvent]);

    expect(state.state).toBe("RUNNING");
    expect(state.datasetName).toBe("cats");
    expect(state.maxEpoch).toBe(2);
    expect(state.globalIter).toBe(2);
    expect(state.totalIters).toBe(4);
    expect(state.epoch).toBe(1);
    expect(state.etaSeconds).toBe(6);
    expect(state.lossSeries.map((p) => p.global_iter)).toEqual([1, 2]);
    expect(state.lossSeries.map((p) => p.total_loss)).toEqual([5, null]);
    expect(state.lossSeries[0]).toMatchObject({ iou_loss: 1, conf_loss: 2, cls_loss: 3, l1_loss: 0 });
    expect(state.lrSeries).toEqual([
      { global_iter: 1, lr: 0.001 },
      { global_iter: 2, lr: 0.002 },
    ]);
    expect(state.evalSeries).toEqual([{ epoch: 1, ap50: 0.5, ap50_95: 0.25 }]);
    expect(state.lastSeq).toBe(4);
  });

  it("ignores events with seq at or below lastSeq", () => {
    const once = run([started, iter(2, 1, 5)]);
    const again = trainingReducer(trainingReducer(once, iter(2, 1, 5)), started);

    expect(again).toBe(once);
  });

  it("records the final state, model id and error on finished", () => {
    const state = run([
      started,
      { type: "finished", seq: 2, ...base, state: "FAILED", model_id: "", error: "CUDA out of memory" },
    ]);

    expect(state.state).toBe("FAILED");
    expect(state.error).toBe("CUDA out of memory");
    expect(state.etaSeconds).toBeNull();

    const done = run([started, { type: "finished", seq: 2, ...base, state: "SUCCEEDED", model_id: "m1", error: "" }]);
    expect(done.modelId).toBe("m1");
  });
});
