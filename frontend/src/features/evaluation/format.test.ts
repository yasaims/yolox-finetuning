import { describe, expect, it } from "vitest";
import { formatStat } from "./format";

describe("formatStat", () => {
  it("shows a dash for the COCO undefined marker and for non-finite values", () => {
    expect(formatStat(-1)).toBe("—");
    expect(formatStat(null)).toBe("—");
    expect(formatStat(Number.NaN)).toBe("—");
  });

  it("keeps zero and formats other values", () => {
    expect(formatStat(0)).toBe("0.000");
    expect(formatStat(0.45678)).toBe("0.457");
  });
});
