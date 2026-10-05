export interface RawDetection {
  x1: number | null;
  y1: number | null;
  x2: number | null;
  y2: number | null;
  score: number | null;
  class_id: number;
  class_name: string;
}

export interface CanvasBox {
  x: number;
  y: number;
  width: number;
  height: number;
  score: number;
  label: string;
}

/** Scales detections from image pixels to canvas pixels, dropping those below the threshold or with non-finite values. */
export function toCanvasBoxes(
  detections: readonly RawDetection[],
  imageW: number,
  imageH: number,
  canvasW: number,
  canvasH: number,
  threshold: number,
): CanvasBox[] {
  const sx = canvasW / imageW;
  const sy = canvasH / imageH;
  const boxes: CanvasBox[] = [];
  for (const d of detections) {
    const { x1, y1, x2, y2, score } = d;
    if (x1 == null || y1 == null || x2 == null || y2 == null || score == null) continue;
    if (![x1, y1, x2, y2, score].every(Number.isFinite) || score < threshold) continue;
    boxes.push({
      x: x1 * sx,
      y: y1 * sy,
      width: (x2 - x1) * sx,
      height: (y2 - y1) * sy,
      score,
      label: `${d.class_name} ${score.toFixed(2)}`,
    });
  }
  return boxes;
}
