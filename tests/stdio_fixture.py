# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Exercise the real CLI and MCP transport without loading model checkpoints."""

from fakehuman import research_mcp


class AcceptingGuard:
    def check_pairs(self, pairs):
        return [{"passed": True, "reason": "test fixture"} for _ in pairs]


class FixtureDetector:
    def __init__(self, score):
        self.value = score
        self.last_input_report = []

    def score(self, texts):
        self.last_input_report = [
            {
                "input_index": index,
                "input_characters": len(text),
                "prepared_characters": len(text),
                "preprocessing_changed": False,
                "tokens": len(text.split()),
                "max_tokens": 512,
                "truncated": False,
            }
            for index, text in enumerate(texts)
        ]
        return [self.value for _ in texts]

    def metadata(self):
        return {
            "model_id": "fixture/model",
            "revision": "fixture-revision",
            "preprocessing": "none",
        }


original_make_server = research_mcp.make_server


def fixture_make_server(manifest):
    return original_make_server(
        manifest,
        guard=AcceptingGuard(),
        detectors={
            "first": FixtureDetector(0.25),
            "second": FixtureDetector(0.75),
        },
    )


research_mcp.make_server = fixture_make_server

# Import after patching so either module- or function-based CLI imports work.
from fakehuman.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
