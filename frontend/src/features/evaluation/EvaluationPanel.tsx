import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { api, type EvaluationResult } from "../../api/client";
import { formatStat } from "./format";

const STAT_ROWS: { key: keyof EvaluationResult["stats"]; label: string }[] = [
  { key: "ap", label: "AP @[IoU=0.50:0.95 | area=all | maxDets=100]" },
  { key: "ap50", label: "AP @[IoU=0.50 | area=all | maxDets=100]" },
  { key: "ap75", label: "AP @[IoU=0.75 | area=all | maxDets=100]" },
  { key: "ap_small", label: "AP @[IoU=0.50:0.95 | area=small | maxDets=100]" },
  { key: "ap_medium", label: "AP @[IoU=0.50:0.95 | area=medium | maxDets=100]" },
  { key: "ap_large", label: "AP @[IoU=0.50:0.95 | area=large | maxDets=100]" },
  { key: "ar1", label: "AR @[IoU=0.50:0.95 | area=all | maxDets=1]" },
  { key: "ar10", label: "AR @[IoU=0.50:0.95 | area=all | maxDets=10]" },
  { key: "ar100", label: "AR @[IoU=0.50:0.95 | area=all | maxDets=100]" },
  { key: "ar_small", label: "AR @[IoU=0.50:0.95 | area=small | maxDets=100]" },
  { key: "ar_medium", label: "AR @[IoU=0.50:0.95 | area=medium | maxDets=100]" },
  { key: "ar_large", label: "AR @[IoU=0.50:0.95 | area=large | maxDets=100]" },
];

type Split = "val" | "train";

export default function EvaluationPanel({ modelId }: { modelId: string }) {
  const [split, setSplit] = useState<Split>("val");
  const evaluation = useMutation({
    mutationFn: (selected: Split) => api.evaluateModel(modelId, { split: selected }),
  });
  const result = evaluation.data;

  return (
    <div className="card">
      <h3>Evaluation</h3>
      <p className="muted">
        The val split has few images, so these numbers are noisy. The train split is optimistic because the model has
        seen it.
      </p>
      <div className="toolbar">
        <label>
          Split{" "}
          <select
            value={split}
            onChange={(event) => {
              setSplit(event.target.value as Split);
              evaluation.reset();
            }}
            disabled={evaluation.isPending}
          >
            <option value="val">val</option>
            <option value="train">train</option>
          </select>
        </label>
        <button type="button" onClick={() => evaluation.mutate(split)} disabled={evaluation.isPending}>
          {evaluation.isPending ? "Evaluating..." : "Run evaluation"}
        </button>
      </div>
      {evaluation.error && <p className="error">{evaluation.error.message}</p>}
      {result && (
        <>
          <p>
            {result.num_images} images evaluated on <strong>{split}</strong>
          </p>
          <table aria-label="COCO statistics">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {STAT_ROWS.map(({ key, label }) => (
                <tr key={key}>
                  <td>{label}</td>
                  <td>{formatStat(result.stats[key])}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <h4>Per class</h4>
          <table aria-label="Per-class AP">
            <thead>
              <tr>
                <th>Class</th>
                <th>AP</th>
                <th>AP50</th>
              </tr>
            </thead>
            <tbody>
              {result.per_class.map((row) => (
                <tr key={row.class_name}>
                  <td>{row.class_name}</td>
                  <td>{formatStat(row.ap)}</td>
                  <td>{formatStat(row.ap50)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
