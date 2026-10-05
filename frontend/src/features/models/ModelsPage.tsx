import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "../../api/client";
import { formatStat } from "../evaluation/format";
import DeleteModelButton from "./DeleteModelButton";

export default function ModelsPage() {
  const models = useQuery({ queryKey: ["models"], queryFn: api.listModels });

  return (
    <>
      <h2>Models</h2>
      {models.isPending && <p className="muted">Loading...</p>}
      {models.error && <p className="error">Failed to load models: {models.error.message}</p>}
      {models.data?.length === 0 && <p className="muted">No models yet. Finish a training run to create one.</p>}
      {models.data && models.data.length > 0 && (
        <div className="card">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>ID</th>
                <th>Dataset</th>
                <th>Status</th>
                <th>Epochs</th>
                <th>Best AP50</th>
                <th>Best AP50-95</th>
                <th>Created</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {models.data.map((model) => (
                <tr key={model.id}>
                  <td>
                    <Link to={`/models/${model.id}`}>{model.display_name || model.id}</Link>
                  </td>
                  <td>
                    <code>{model.id}</code>
                  </td>
                  <td>{model.dataset_name}</td>
                  <td>
                    <span className={`badge badge-${model.status === "completed" ? "succeeded" : "stopped"}`}>
                      {model.status}
                    </span>
                  </td>
                  <td>{model.epochs_trained}</td>
                  <td>{formatStat(model.best_ap50)}</td>
                  <td>{formatStat(model.best_ap50_95)}</td>
                  <td>{new Date(model.created_unix_ms).toLocaleString()}</td>
                  <td>
                    <DeleteModelButton modelId={model.id} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}
