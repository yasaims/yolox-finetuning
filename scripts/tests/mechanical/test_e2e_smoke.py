from dataset_tools.e2e_smoke import SseFrame, parse_sse


def test_parse_sse_frames() -> None:
    lines = [
        ": ping",
        "id: 3",
        "event: iter",
        'data: {"epoch": 1}',
        "",
        "event: finished",
        'data: {"state":',
        'data: "SUCCEEDED"}',
        "",
        "event: eval",
        "",
    ]
    assert list(parse_sse(lines)) == [
        SseFrame("iter", "3", {"epoch": 1}),
        SseFrame("finished", "", {"state": "SUCCEEDED"}),
    ]
