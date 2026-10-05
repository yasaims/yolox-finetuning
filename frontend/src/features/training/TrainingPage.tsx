import { useQuery } from "@tanstack/react-query";
import { useEffect, useReducer, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../../api/client";
import { subscribeTrainingEvents } from "../../api/sse";
import { ApChart, LossChart, LrChart } from "./charts";
import TrainingForm from "./TrainingForm";
import { ACTIVE_STATES, initialTrainingState, trainingReducer, type TrainingState } from "./trainingReducer";

function formatEta(seconds: number | null): string {
  if (seconds === null) return "-";
  const total = Math.max(0, Math.round(seconds));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  return h > 0 ? `${h}h ${m}m` : m > 0 ? `${m}m ${s}s` : `${s}s`;
}

// Replays the job's events from seq 0; the reducer drops anything already applied.
function useTrainingStream(jobId: string | null) {
  const [state, dispatch] = useReducer(trainingReducer, initialTrainingState);
  const [disconnected, setDisconnected] = useState(false);

  useEffect(() => {
    dispatch({ type: "reset" });
    setDisconnected(false);
    if (!jobId) return;
    return subscribeTrainingEvents(
      jobId,
      0,
      (event) => {
        setDisconnected(false);
        dispatch(event);
      },
      () => setDisconnected(true),
    );
  }, [jobId]);

  return { state, disconnected };
}

interface StatusProps {
  jobId: string;
  state: TrainingState;
  serverState: string | null;
  disconnected: boolean;
}

// serverState is the state reported by /trainings/current; the event stream never reports STOPPING.
export function TrainingStatus({ jobId, state, serverState, disconnected }: StatusProps) {
  const [stopRequested, setStopRequested] = useState(false);
  const [stopError, setStopError] = useState("");
  const active = ACTIVE_STATES.includes(state.state);
  const stopping = active && (stopRequested || serverState === "STOPPING");
  const badge = stopping ? "STOPPING" : state.state;
  const percent = state.totalIters > 0 ? (state.globalIter / state.totalIters) * 100 : 0;

  async function stop() {
    setStopRequested(true);
    setStopError("");
    try {
      await api.stopTraining(jobId);
    } catch (error) {
      setStopRequested(false);
      setStopError(error instanceof Error ? error.message : String(error));
    }
  }

  return (
    <div className="card">
      <h3>
        {state.datasetName || "Training"} <span className={`badge badge-${badge.toLowerCase()}`}>{badge}</span>
      </h3>
      <progress value={state.globalIter} max={state.totalIters || 1} aria-label="Training progress" />
      <p>
        Epoch {state.epoch} / {state.maxEpoch} &middot; iteration {state.globalIter} / {state.totalIters} (
        {percent.toFixed(1)}%) &middot; ETA {formatEta(state.etaSeconds)}
      </p>
      {disconnected && active && <p className="muted">Connection lost, retrying...</p>}
      {active && (
        <button type="button" onClick={stop} disabled={stopping}>
          Stop
        </button>
      )}
      {stopError && <p className="error">{stopError}</p>}
      {state.error && <p className="error">{state.error}</p>}
      {state.modelId && (
        <p>
          Model saved: <Link to={`/models/${state.modelId}`}>{state.modelId}</Link>
        </p>
      )}
    </div>
  );
}

export default function TrainingPage() {
  const [jobId, setJobId] = useState<string | null>(null);
  const current = useQuery({ queryKey: ["trainings", "current"], queryFn: api.currentTraining });
  const { state, disconnected } = useTrainingStream(jobId);

  const currentJobId = current.data?.job?.job_id ?? null;
  const serverState = currentJobId === jobId ? (current.data?.job?.state ?? null) : null;
  useEffect(() => {
    if (jobId === null && currentJobId !== null) setJobId(currentJobId);
  }, [jobId, currentJobId]);

  const active = jobId !== null && ACTIVE_STATES.includes(state.state);

  return (
    <>
      <h2>Training</h2>
      {current.error && <p className="error">Failed to load the current job: {current.error.message}</p>}
      {!active && !current.isPending && <TrainingForm onStarted={setJobId} />}
      {jobId && (
        <>
          <TrainingStatus key={jobId} jobId={jobId} state={state} serverState={serverState} disconnected={disconnected} />
          <LossChart data={state.lossSeries} />
          <LrChart data={state.lrSeries} />
          <ApChart data={state.evalSeries} />
        </>
      )}
    </>
  );
}
