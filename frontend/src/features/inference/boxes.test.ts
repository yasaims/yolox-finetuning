import { describe, expect, it } from "vitest";
import { toCanvasBoxes } from "./boxes";

const det = (over: Record<string, unknown> = {}) => ({
  x1: 100,
  y1: 50,
  x2: 300,
  y2: 250,
  score: 0.8,
  class_id: 1,
  class_name: "cat",
  ...over,
});

describe("toCanvasBoxes", () => {
  it("scales image pixel coordinates to the canvas and builds the label", () => {
    const [box] = toCanvasBoxes([det()], 400, 300, 200, 150, 0.3);
    expect(box).toMatchObject({ x: 50, y: 25, width: 100, height: 100, label: "cat 0.80" });
  });

  it("keeps boxes at or above the threshold and drops the rest", () => {
    const boxes = toCanvasBoxes(
      [det({ score: 0.29 }), det({ score: 0.3 }), det({ score: 0.9 })],
      400,
      300,
      400,
      300,
      0.3,
    );
    expect(boxes.map((b) => b.score)).toEqual([0.3, 0.9]);
  });

  it("skips boxes with non-finite (null) coordinates or score", () => {
    const boxes = toCanvasBoxes([det({ x2: null }), det({ score: null }), det()], 400, 300, 400, 300, 0);
    expect(boxes).toHaveLength(1);
  });
});
