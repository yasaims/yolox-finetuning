import { afterEach, describe, expect, it, vi } from "vitest";
import { subscribeTrainingEvents } from "./sse";

class FakeEventSource {
  static last: FakeEventSource;
  closed = false;
  onerror: ((e: Event) => void) | null = null;
  private listeners = new Map<string, (m: MessageEvent<string>) => void>();

  constructor(readonly url: string) {
    FakeEventSource.last = this;
  }

  addEventListener(type: string, listener: (m: MessageEvent<string>) => void) {
    this.listeners.set(type, listener);
  }

  close() {
    this.closed = true;
  }

  emit(type: string, data: object) {
    this.listeners.get(type)!({ data: JSON.stringify(data) } as MessageEvent<string>);
  }
}

afterEach(() => vi.unstubAllGlobals());

describe("subscribeTrainingEvents", () => {
  it("closes the stream on finished even when the handler throws", () => {
    vi.stubGlobal("EventSource", FakeEventSource);
    const onEvent = vi.fn(() => {
      throw new Error("handler failed");
    });
    subscribeTrainingEvents("job1", 0, onEvent, () => {});

    expect(FakeEventSource.last.url).toBe("/api/trainings/job1/events?after_seq=0");
    expect(() => FakeEventSource.last.emit("finished", { seq: 3, state: "SUCCEEDED" })).toThrow();
    expect(FakeEventSource.last.closed).toBe(true);
    expect(onEvent).toHaveBeenCalledWith(expect.objectContaining({ type: "finished", seq: 3 }));
  });
});
