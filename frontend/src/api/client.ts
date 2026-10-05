import type { components } from "./schema";

type Schemas = components["schemas"];

export type Dataset = Schemas["Dataset"];
export type Hyperparameters = Schemas["Hyperparameters"];
export type StartTrainingRequest = Pick<Schemas["StartTrainingRequest"], "dataset_name"> & Partial<Schemas["StartTrainingRequest"]>;
export type StartTrainingResult = Schemas["StartTrainingResult"];
export type JobSummary = Schemas["JobSummary"];
export type Model = Schemas["Model"];
export type EpochEval = Schemas["EpochEval"];
export type EvaluateRequest = Partial<Schemas["EvaluateRequest"]>;
export type EvaluationResult = Schemas["EvaluationResult"];
export type PredictResponse = Schemas["PredictResponse"];
export type Detection = Schemas["Detection"];

const BASE = "/api";

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(`${status}: ${detail}`);
    this.name = "ApiError";
  }
}

function errorDetail(body: unknown, fallback: string): string {
  const detail = (body as { detail?: unknown } | null)?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d: { loc?: unknown[]; msg?: string }) => `${(d.loc ?? []).slice(1).join(".")}: ${d.msg ?? ""}`)
      .join("; ");
  }
  return fallback;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(BASE + path, init);
  if (!response.ok) {
    const body: unknown = await response.json().catch(() => null);
    throw new ApiError(response.status, errorDetail(body, response.statusText));
  }
  if (response.status === 204 || response.status === 202) return undefined as T;
  return (await response.json()) as T;
}

function json(method: string, body: unknown): RequestInit {
  return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

export const api = {
  listDatasets: () => request<Dataset[]>("/datasets"),
  defaultHyperparameters: () => request<Hyperparameters>("/hyperparameters/defaults"),

  startTraining: (body: StartTrainingRequest) => request<StartTrainingResult>("/trainings", json("POST", body)),
  currentTraining: () => request<{ job: JobSummary | null }>("/trainings/current"),
  stopTraining: (jobId: string) => request<void>(`/trainings/${encodeURIComponent(jobId)}/stop`, { method: "POST" }),

  listModels: () => request<Model[]>("/models"),
  getModel: (id: string) => request<Model>(`/models/${encodeURIComponent(id)}`),
  deleteModel: (id: string) => request<void>(`/models/${encodeURIComponent(id)}`, { method: "DELETE" }),
  evaluateModel: (id: string, body: EvaluateRequest = {}) =>
    request<EvaluationResult>(`/models/${encodeURIComponent(id)}/evaluate`, json("POST", body)),
  predict: (id: string, file: Blob, scoreThreshold: number) => {
    const form = new FormData();
    form.append("file", file);
    form.append("score_threshold", String(scoreThreshold));
    return request<PredictResponse>(`/models/${encodeURIComponent(id)}/predict`, { method: "POST", body: form });
  },
};
