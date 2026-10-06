import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState, type FormEvent } from "react";
import { ApiError, api, type Dataset, type Hyperparameters } from "../../api/client";

type NumericKey = Exclude<keyof Hyperparameters, "enable_mixup" | "fp16">;

const NUMERIC_FIELDS: { key: NumericKey; step: string }[] = [
  { key: "max_epoch", step: "1" },
  { key: "batch_size", step: "1" },
  { key: "learning_rate", step: "any" },
  { key: "input_size", step: "32" },
  { key: "warmup_epochs", step: "1" },
  { key: "no_aug_epochs", step: "1" },
  { key: "mosaic_prob", step: "0.1" },
  { key: "seed", step: "1" },
];

function startErrorMessage(error: Error): string {
  return error instanceof ApiError ? error.detail : error.message;
}

interface BodyProps {
  datasets: Dataset[];
  defaults: Hyperparameters;
  onStarted: (jobId: string) => void;
}

function FormBody({ datasets, defaults, onStarted }: BodyProps) {
  const queryClient = useQueryClient();
  const [datasetName, setDatasetName] = useState(datasets[0]?.name ?? "");
  const [displayName, setDisplayName] = useState("");
  const [numbers, setNumbers] = useState<Record<NumericKey, string>>(
    () =>
      Object.fromEntries(NUMERIC_FIELDS.map(({ key }) => [key, String(defaults[key] ?? "")])) as Record<
        NumericKey,
        string
      >,
  );
  const [enableMixup, setEnableMixup] = useState(defaults.enable_mixup ?? false);
  const [fp16, setFp16] = useState(defaults.fp16 ?? true);

  const start = useMutation({
    mutationFn: api.startTraining,
    onSuccess: async (result) => {
      // The page adopts the server's current job, so it must already be the new one.
      await queryClient.invalidateQueries({ queryKey: ["trainings", "current"] });
      onStarted(result.job_id);
    },
    onError: (error) => {
      if (error instanceof ApiError && error.status === 409) {
        void queryClient.invalidateQueries({ queryKey: ["trainings", "current"] });
      }
    },
  });

  function submit(event: FormEvent) {
    event.preventDefault();
    const hyperparameters: Hyperparameters = { enable_mixup: enableMixup, fp16 };
    for (const { key } of NUMERIC_FIELDS) {
      if (numbers[key].trim() !== "") hyperparameters[key] = Number(numbers[key]);
    }
    start.mutate({
      dataset_name: datasetName,
      ...(displayName.trim() ? { display_name: displayName.trim() } : {}),
      hyperparameters,
    });
  }

  return (
    <form className="card training-form" onSubmit={submit}>
      <h3>New training</h3>
      <label>
        Dataset
        <select value={datasetName} onChange={(e) => setDatasetName(e.target.value)}>
          {datasets.map((d) => (
            <option key={d.name} value={d.name}>
              {d.name} ({d.train_images} train / {d.val_images} val)
            </option>
          ))}
        </select>
      </label>
      <label>
        Display name
        <input value={displayName} onChange={(e) => setDisplayName(e.target.value)} />
      </label>
      <div className="form-grid">
        {NUMERIC_FIELDS.map(({ key, step }) => (
          <label key={key}>
            {key}
            <input
              type="number"
              step={step}
              value={numbers[key]}
              onChange={(e) => setNumbers((prev) => ({ ...prev, [key]: e.target.value }))}
            />
          </label>
        ))}
        <label>
          <input type="checkbox" checked={enableMixup} onChange={(e) => setEnableMixup(e.target.checked)} />
          enable_mixup
        </label>
        <label>
          <input type="checkbox" checked={fp16} onChange={(e) => setFp16(e.target.checked)} />
          fp16
        </label>
      </div>
      {start.error && <p className="error">{startErrorMessage(start.error)}</p>}
      <button type="submit" disabled={!datasetName || start.isPending}>
        Start training
      </button>
    </form>
  );
}

export default function TrainingForm({ onStarted }: { onStarted: (jobId: string) => void }) {
  const datasets = useQuery({ queryKey: ["datasets"], queryFn: api.listDatasets });
  const defaults = useQuery({ queryKey: ["hyperparameters", "defaults"], queryFn: api.defaultHyperparameters });

  const loadError = datasets.error ?? defaults.error;
  if (loadError) return <p className="error">Failed to load form data: {loadError.message}</p>;
  if (!datasets.data || !defaults.data) return <p className="muted">Loading...</p>;
  if (datasets.data.length === 0) return <p className="muted">No datasets found in the datasets directory.</p>;
  return <FormBody datasets={datasets.data} defaults={defaults.data} onStarted={onStarted} />;
}
