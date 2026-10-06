import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { api, type PredictResponse } from "../../api/client";
import { toCanvasBoxes } from "./boxes";

const FETCH_THRESHOLD = 0.05;
const MAX_CANVAS_WIDTH = 900;
const LABEL_FONT = "14px sans-serif";

export default function InferencePage() {
  const [params] = useSearchParams();
  const models = useQuery({ queryKey: ["models"], queryFn: api.listModels });
  const [modelId, setModelId] = useState(params.get("model") ?? "");
  const [file, setFile] = useState<File | null>(null);
  const [threshold, setThreshold] = useState(0.3);
  const [image, setImage] = useState<HTMLImageElement | null>(null);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [imageError, setImageError] = useState("");
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const requested = models.data?.some((m) => m.id === modelId) ? modelId : "";
  const selectedModel = requested || models.data?.[0]?.id || "";

  const predict = useMutation({
    mutationFn: (f: File) => api.predict(selectedModel, f, FETCH_THRESHOLD),
  });

  useEffect(() => {
    setResult(null);
    setImage(null);
    setImageError("");
    predict.reset();
    if (!file) return;
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => setImage(img);
    img.onerror = () => setImageError("Could not read the selected file as an image.");
    img.src = url;
    return () => {
      img.onload = null;
      img.onerror = null;
      URL.revokeObjectURL(url);
    };
  }, [file]);

  const canvasSize = useMemo(() => {
    if (!image) return null;
    const scale = Math.min(1, MAX_CANVAS_WIDTH / image.naturalWidth);
    return { width: Math.round(image.naturalWidth * scale), height: Math.round(image.naturalHeight * scale) };
  }, [image]);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !image || !canvasSize) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(image, 0, 0, canvasSize.width, canvasSize.height);
    if (!result) return;
    const boxes = toCanvasBoxes(
      result.detections,
      result.image_width,
      result.image_height,
      canvasSize.width,
      canvasSize.height,
      threshold,
    );
    ctx.lineWidth = 2;
    ctx.font = LABEL_FONT;
    ctx.textBaseline = "top";
    for (const box of boxes) {
      ctx.strokeStyle = "#e5484d";
      ctx.strokeRect(box.x, box.y, box.width, box.height);
      const textWidth = ctx.measureText(box.label).width + 6;
      const labelY = box.y >= 18 ? box.y - 18 : box.y;
      ctx.fillStyle = "#e5484d";
      ctx.fillRect(box.x, labelY, textWidth, 18);
      ctx.fillStyle = "#fff";
      ctx.fillText(box.label, box.x + 3, labelY + 2);
    }
  }, [image, canvasSize, result, threshold]);

  const shown = result && canvasSize
    ? toCanvasBoxes(result.detections, result.image_width, result.image_height, canvasSize.width, canvasSize.height, threshold).length
    : 0;

  return (
    <>
      <h2>Inference</h2>
      <div className="card">
        <div className="toolbar">
          <label>
            Model{" "}
            <select value={selectedModel} onChange={(e) => {
                setModelId(e.target.value);
                setResult(null);
                predict.reset();
              }} disabled={!models.data?.length}>
              {models.data?.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.display_name || m.id}
                </option>
              ))}
            </select>
          </label>
          <input
            type="file"
            accept="image/*"
            aria-label="Image"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
          <button onClick={() => file && predict.mutate(file, { onSuccess: setResult })} disabled={!file || !image || !selectedModel || predict.isPending}>
            {predict.isPending ? "Running..." : "Run"}
          </button>
        </div>
        {models.error && <p className="error">Failed to load models: {models.error.message}</p>}
        {models.data?.length === 0 && <p className="muted">No models yet. Finish a training run to create one.</p>}
        {imageError && <p className="error">{imageError}</p>}
        {predict.error && <p className="error">Inference failed: {predict.error.message}</p>}
        {result && (
          <div className="toolbar">
            <label>
              Score threshold {threshold.toFixed(2)}{" "}
              <input
                type="range"
                min={FETCH_THRESHOLD}
                max={1}
                step={0.01}
                value={threshold}
                onChange={(e) => setThreshold(Number(e.target.value))}
              />
            </label>
            <span className="muted">
              {shown} / {result.detections.length} detections
            </span>
          </div>
        )}
        {canvasSize && (
          <canvas ref={canvasRef} width={canvasSize.width} height={canvasSize.height} style={{ maxWidth: "100%" }} />
        )}
      </div>
    </>
  );
}
