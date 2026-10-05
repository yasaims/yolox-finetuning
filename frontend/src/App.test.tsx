import { render, screen } from "@testing-library/react";
import { MemoryRouter } from "react-router-dom";
import { describe, expect, it } from "vitest";
import App from "./App";

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  );
}

describe("App routes", () => {
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
