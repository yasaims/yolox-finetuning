import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
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
});
