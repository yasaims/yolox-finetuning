import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { LOSS_KEYS, type ApPoint, type LossKey, type LossPoint, type LrPoint } from "./trainingReducer";

const LOSS_COLORS: Record<LossKey, string> = {
  total_loss: "#1f2328",
  iou_loss: "#0969da",
  conf_loss: "#bf8700",
  cls_loss: "#1a7f37",
  l1_loss: "#8250df",
};

const formatTick = (value: number) => value.toPrecision(3);

interface ChartProps {
  title: string;
  data: object[];
  xKey: string;
  xLabel: string;
  lines: { key: string; color: string; width?: number }[];
  tickFormatter?: (value: number) => string;
}

// Null values (non-finite metrics) leave gaps instead of breaking the chart.
function SeriesChart({ title, data, xKey, xLabel, lines, tickFormatter }: ChartProps) {
  return (
    <div className="card">
      <h3>{title}</h3>
      <ResponsiveContainer width="100%" height={240}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey={xKey}
            type="number"
            domain={["dataMin", "dataMax"]}
            label={{ value: xLabel, position: "insideBottom", offset: -4 }}
            height={40}
          />
          <YAxis tickFormatter={tickFormatter ?? formatTick} width={64} />
          <Tooltip formatter={(value) => (typeof value === "number" ? formatTick(value) : "n/a")} />
          <Legend />
          {lines.map((line) => (
            <Line
              key={line.key}
              dataKey={line.key}
              stroke={line.color}
              strokeWidth={line.width ?? 1.5}
              dot={false}
              connectNulls={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function LossChart({ data }: { data: LossPoint[] }) {
  return (
    <SeriesChart
      title="Loss"
      data={data}
      xKey="global_iter"
      xLabel="iteration"
      lines={LOSS_KEYS.map((key) => ({ key, color: LOSS_COLORS[key], width: key === "total_loss" ? 2.5 : 1.5 }))}
    />
  );
}

export function LrChart({ data }: { data: LrPoint[] }) {
  return (
    <SeriesChart
      title="Learning rate"
      data={data}
      xKey="global_iter"
      xLabel="iteration"
      lines={[{ key: "lr", color: "#0969da" }]}
      tickFormatter={(value) => value.toExponential(1)}
    />
  );
}

export function ApChart({ data }: { data: ApPoint[] }) {
  return (
    <SeriesChart
      title="Validation AP"
      data={data}
      xKey="epoch"
      xLabel="epoch"
      lines={[
        { key: "ap50", color: "#1a7f37" },
        { key: "ap50_95", color: "#0969da" },
      ]}
    />
  );
}
