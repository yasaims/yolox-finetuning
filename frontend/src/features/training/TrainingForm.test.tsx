import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import TrainingForm from "./TrainingForm";

const DEFAULTS = {
  max_epoch: 50,
  batch_size: 8,
  learning_rate: 0.00125,
  input_size: 416,
  warmup_epochs: 3,
  no_aug_epochs: 10,
  mosaic_prob: 1,
  enable_mixup: false,
  fp16: true,
  seed: 42,
};

function mockApi(startResponse: Response) {
  const fetchMock = vi.fn(async (path: string) => {
    if (path === "/api/datasets") {
      return Response.json([
        { name: "cats", class_names: ["cat"], train_images: 10, val_images: 2 },
        { name: "dogs", class_names: ["dog"], train_images: 10, val_images: 2 },
      ]);
    }
    if (path === "/api/hyperparameters/defaults") return Response.json(DEFAULTS);
    return startResponse;
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

function renderForm(onStarted = vi.fn()) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <TrainingForm onStarted={onStarted} />
    </QueryClientProvider>,
  );
  return onStarted;
}

afterEach(() => vi.unstubAllGlobals());

describe("TrainingForm", () => {
  it("posts the selected dataset and edited hyperparameters", async () => {
    const fetchMock = mockApi(Response.json({ job_id: "j1", hyperparameters: DEFAULTS }, { status: 201 }));
    const onStarted = renderForm();

    fireEvent.change(await screen.findByLabelText("Dataset"), { target: { value: "dogs" } });
    expect(screen.getByLabelText("max_epoch")).toHaveValue(50);
    fireEvent.change(screen.getByLabelText("max_epoch"), { target: { value: "5" } });
    fireEvent.click(screen.getByLabelText("enable_mixup"));
    fireEvent.click(screen.getByRole("button", { name: "Start training" }));

    await waitFor(() => expect(onStarted).toHaveBeenCalledWith("j1"));
    const [path, init] = fetchMock.mock.calls.find(([p]) => p === "/api/trainings")! as unknown as [string, RequestInit];
    expect(path).toBe("/api/trainings");
    expect(init.method).toBe("POST");
    expect(JSON.parse(init.body as string)).toEqual({
      dataset_name: "dogs",
      hyperparameters: { ...DEFAULTS, max_epoch: 5, enable_mixup: true },
    });
  });

  it("shows the server's reason for a 409 conflict", async () => {
    mockApi(Response.json({ detail: "pretrained weights not found" }, { status: 409 }));
    renderForm();

    fireEvent.click(await screen.findByRole("button", { name: "Start training" }));

    expect(await screen.findByText("pretrained weights not found")).toBeInTheDocument();
  });

  it("shows the server's validation message", async () => {
    mockApi(Response.json({ detail: "warmup_epochs must be below max_epoch" }, { status: 422 }));
    renderForm();

    fireEvent.click(await screen.findByRole("button", { name: "Start training" }));

    expect(await screen.findByText("warmup_epochs must be below max_epoch")).toBeInTheDocument();
  });
});
