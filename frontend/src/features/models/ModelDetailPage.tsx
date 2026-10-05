import { useQuery } from "@tanstack/react-query";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, type Model } from "../../api/client";
import EvaluationPanel from "../evaluation/EvaluationPanel";
import { formatStat } from "../evaluation/format";
import { ApChart } from "../training/charts";
import DeleteModelButton from "./DeleteModelButton";

function MetadataCard({ model }: { model: Model }) {
  const rows: [string, React.ReactNode][] = [
    ["ID", <code key="id">{model.id}</code>],
    ["Dataset", model.dataset_name],
    ["Classes", model.class_names.join(", ")],
    ["Epochs trained", model.epochs_trained],
    ["Best epoch", model.best_epoch],
    ["Best AP50", formatStat(model.best_ap50)],
    ["Best AP50-95", formatStat(model.best_ap50_95)],
    ["Created", new Date(model.created_unix_ms).toLocaleString()],
  ];
  return (
    <div className="card">
      <h3>
        {model.display_name || model.id}{" "}
        <span className={`badge badge-${model.status === "completed" ? "succeeded" : "stopped"}`}>{model.status}</span>
      </h3>
      <table>
        <tbody>
          {rows.map(([label, value]) => (
            <tr key={label}>
              <th>{label}</th>
              <td>{value}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function HyperparametersCard({ model }: { model: Model }) {
  return (
    <div className="card">
      <h3>Hyperparameters</h3>
      <table aria-label="Hyperparameters">
        <tbody>
          {Object.entries(model.hyperparameters).map(([name, value]) => (
            <tr key={name}>
              <th>{name}</th>
              <td>{value === null || value === undefined ? "—" : String(value)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function ModelDetailPage() {
  const { id = "" } = useParams();
  const navigate = useNavigate();
  const model = useQuery({ queryKey: ["models", id], queryFn: () => api.getModel(id) });

  const heading = <h2>Model {id}</h2>;
  if (model.isPending) {
    return (
      <>
        {heading}
        <p className="muted">Loading...</p>
      </>
    );
  }
  if (model.error) {
    return (
      <>
        {heading}
        <p className="error">Failed to load model: {model.error.message}</p>
        <Link to="/models">Back to models</Link>
      </>
    );
  }

  return (
    <>
      {heading}
      <p>
        <Link to="/models">&larr; Models</Link>
      </p>
      <div className="toolbar">
        <Link to={`/inference?model=${encodeURIComponent(id)}`}>Run inference</Link>
        <DeleteModelButton modelId={id} onDeleted={() => navigate("/models")} />
      </div>
      <MetadataCard model={model.data} />
      <HyperparametersCard model={model.data} />
      <ApChart data={model.data.eval_history} />
      <EvaluationPanel modelId={id} />
    </>
  );
}
