import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, describe, expect, it, vi } from "vitest";
import TrainingPage, { TrainingStatus } from "./TrainingPage";
import { initialTrainingState } from "./trainingReducer";

const subscribe = vi.hoisted(() => vi.fn((..._args: unknown[]) => () => {}));
vi.mock("../../api/sse", () => ({ subscribeTrainingEvents: subscribe }));

describe("TrainingStatus", () => {
  it("shows STOPPING with Stop disabled when the server reports the job as stopping", () => {
    const running = { ...initialTrainingState, state: "RUNNING" };
    render(<TrainingStatus jobId="j1" state={running} serverState="STOPPING" disconnected={false} />);

    expect(screen.getByText("STOPPING")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Stop" })).toBeDisabled();
  });
});

describe("TrainingPage", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("follows the server's current job instead of a stale cached one", async () => {
    subscribe.mockClear();
    vi.stubGlobal(
      "fetch",
      vi.fn(async (path: string) =>
        path === "/api/trainings/current" ? Response.json({ job: { job_id: "job-b", state: "RUNNING" } }) : Response.json([]),
      ),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    client.setQueryData(["trainings", "current"], { job: { job_id: "job-a", state: "SUCCEEDED" } });

    render(
      <QueryClientProvider client={client}>
        <MemoryRouter>
          <TrainingPage />
        </MemoryRouter>
      </QueryClientProvider>,
    );

    await waitFor(() => expect(subscribe).toHaveBeenCalledWith("job-b", 0, expect.any(Function), expect.any(Function)));
    expect(subscribe.mock.calls.map((call) => call[0])).not.toContain("job-a");
  });

  it("does not bounce back to the previous finished job after starting a new one", async () => {
    subscribe.mockClear();
    subscribe.mockImplementation(((id: string, _after: number, onEvent: (e: unknown) => void) => {
      if (id === "job-a") onEvent({ type: "finished", seq: 1, state: "SUCCEEDED", model_id: "job-a", error: "" });
      return () => {};
    }) as never);
    let started = false;
    vi.stubGlobal(
      "fetch",
      vi.fn(async (path: string, init?: RequestInit) => {
        if (path === "/api/trainings/current") {
          if (!started) return Response.json({ job: { job_id: "job-a", state: "SUCCEEDED" } });
          await new Promise((resolve) => setTimeout(resolve, 20));
          return Response.json({ job: { job_id: "job-b", state: "RUNNING" } });
        }
        if (path === "/api/trainings" && init?.method === "POST") {
          started = true;
          return Response.json({ job_id: "job-b" });
        }
        if (path === "/api/datasets") return Response.json([{ name: "cats", class_names: ["cat"], train_images: 1, val_images: 1 }]);
        return Response.json({});
      }),
    );
    const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter>
          <TrainingPage />
        </MemoryRouter>
      </QueryClientProvider>,
    );

    fireEvent.click(await screen.findByRole("button", { name: "Start training" }));
    await waitFor(() => expect(subscribe.mock.calls.map((call) => call[0])).toContain("job-b"));
    await new Promise((resolve) => setTimeout(resolve, 100));

    const ids = subscribe.mock.calls.map((call) => call[0]);
        expect(ids.slice(ids.indexOf("job-b"))).toEqual(["job-b"]);
    subscribe.mockImplementation((..._args: unknown[]) => () => {});
  });
});
