import { afterEach, describe, expect, it, vi } from "vitest";
import { ApiError, api } from "./client";

function mockFetch(status: number, body: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue(new Response(body === undefined ? null : JSON.stringify(body), { status })),
  );
}

afterEach(() => vi.unstubAllGlobals());

describe("api client", () => {
  it("throws ApiError with the string detail", async () => {
    mockFetch(409, { detail: "a job is already running" });
    await expect(api.listDatasets()).rejects.toMatchObject({ status: 409, detail: "a job is already running" });
    await expect(api.listDatasets()).rejects.toBeInstanceOf(ApiError);
  });

  it("flattens FastAPI validation errors into one message", async () => {
    mockFetch(422, { detail: [{ loc: ["body", "hyperparameters", "max_epoch"], msg: "must be positive" }] });
    await expect(api.startTraining({ dataset_name: "x" })).rejects.toMatchObject({
      status: 422,
      detail: "hyperparameters.max_epoch: must be positive",
    });
  });

  it("resolves empty bodies for 204 responses", async () => {
    mockFetch(204, undefined);
    await expect(api.deleteModel("m1")).resolves.toBeUndefined();
  });
});
