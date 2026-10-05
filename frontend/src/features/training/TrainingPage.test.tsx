import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { TrainingStatus } from "./TrainingPage";
import { initialTrainingState } from "./trainingReducer";

describe("TrainingStatus", () => {
  it("shows STOPPING with Stop disabled when the server reports the job as stopping", () => {
    const running = { ...initialTrainingState, state: "RUNNING" };
    render(<TrainingStatus jobId="j1" state={running} serverState="STOPPING" disconnected={false} />);

    expect(screen.getByText("STOPPING")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Stop" })).toBeDisabled();
  });
});
