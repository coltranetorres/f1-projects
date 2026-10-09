import { describe, expect, it } from "vitest";
import { parseSSE } from "./sse.js";

describe("parseSSE", () => {
  it("parses complete events and keeps the partial remainder", () => {
    const buf = 'data: {"type":"a"}\n\ndata: {"type":"b","n":1}\n\ndata: {"type":"c"';
    const { events, rest } = parseSSE(buf);
    expect(events).toEqual([{ type: "a" }, { type: "b", n: 1 }]);
    expect(rest).toBe('data: {"type":"c"');
  });

  it("returns nothing for an empty buffer", () => {
    expect(parseSSE("")).toEqual({ events: [], rest: "" });
  });

  it("ignores blocks without a data line", () => {
    expect(parseSSE(": keepalive\n\n").events).toEqual([]);
  });
});
