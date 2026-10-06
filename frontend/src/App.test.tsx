import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";

function renderAt(path: string) {
  return render(
    <QueryClientProvider client={new QueryClient()}>
      <MemoryRouter initialEntries={[path]}>
        <App />
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("App routes", () => {
  beforeEach(() => vi.stubGlobal("fetch", vi.fn(() => new Promise(() => {}))));
  afterEach(() => vi.unstubAllGlobals());

  it.each([
    ["/training", "Training"],
    ["/models", "Models"],
    ["/models/abc", "Model abc"],
    ["/inference", "Inference"],
  ])("%s renders its page", (path, heading) => {
    renderAt(path);
    expect(screen.getByRole("heading", { level: 2, name: heading })).toBeInTheDocument();
  });

  it("redirects / to training and shows the nav", () => {
    renderAt("/");
    expect(screen.getByRole("heading", { level: 2, name: "Training" })).toBeInTheDocument();
    for (const name of ["Training", "Models", "Inference"]) {
      expect(screen.getByRole("link", { name })).toBeInTheDocument();
    }
  });
});
