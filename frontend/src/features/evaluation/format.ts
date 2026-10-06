// COCO reports -1 for undefined statistics; the gateway sends null for non-finite floats.
export function formatStat(value: number | null | undefined, digits = 3): string {
  if (value === null || value === undefined || !Number.isFinite(value) || value < 0) return "—";
  return value.toFixed(digits);
}
