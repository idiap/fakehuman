<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
SPDX-FileContributor: William Droz <william.droz@idiap.ch>
SPDX-License-Identifier: GPL-3.0-only
-->

# fakehuman

A standalone MCP server and browser workbench for text post-processing and
AI-text detection. Its default workflow applies deterministic Python edits with local
meaning checks and detectors. Optional LLM workflows propose and review rewrites
through a configured OpenAI-compatible provider.

## Install and run

Requires Python 3.13 or newer, [uv](https://docs.astral.sh/uv/), and
[Git LFS](https://git-lfs.com/) for the dependency lock.
Run `git lfs install` before cloning the repository using its published clone
URL. In an existing checkout, run `git lfs pull` to fetch the lock, then run:

```sh
cd fakehuman
uv sync --locked --extra cpu
uv run --locked --extra cpu fakehuman
```

The last command runs an MCP **stdio** server and waits for a client on standard
input. Connect it to an MCP client to call its tools.
`uv run --locked --extra cpu python -m fakehuman` is equivalent.
Use `uv run --locked --extra cpu fakehuman --help` to inspect options without
loading models.

The `cpu` extra selects PyTorch's CPU-only wheels on Linux and Windows, avoiding
CUDA dependencies. On macOS it uses the PyPI wheels. Keep `--extra cpu` in
subsequent `uv run` commands and MCP configurations to retain that selection.

On first use, the relevant pinned Hugging Face model checkpoints download to
the normal local cache. The two detectors together require roughly 2.3 GB of
weights; the meaning screen downloads an additional checkpoint. Subsequent
calls reuse the cache. CPU is the default, with four PyTorch threads.

### Prefetch models and run offline

Download all three pinned checkpoints before starting a client:

```sh
uv run --locked --extra cpu fakehuman download-models
uv run --locked --extra cpu fakehuman download-models --offline
uv run --locked --extra cpu fakehuman --offline
```

The download command fetches only the required configuration, tokenizer and
safetensors files; it does not load models into memory. It prints the pinned
revisions and cache locations as JSON. `download-models --offline` checks every
required file, including weights, and fails with instructions if anything is
missing. `--offline` on the server prevents Hugging Face network lookups and
disables MCP update checks and remote LLM rewriting. It must be used after prefetching for calls that
need models. `HF_HOME` and `HF_HUB_CACHE` can override the normal cache location.

### Optional browser interface

Install the `ui` extra to try transformations and detector scores with NiceGUI:

```sh
uv sync --locked --extra cpu --extra ui
uv run --locked --extra cpu --extra ui fakehuman ui
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080). Paste or type a passage,
inspect the accepted output and attempted changes, and copy the input, accepted
text, or candidate with their buttons. In the inline diff, removed words are
struck through and added words are underlined. Clipboard access depends on
browser permission and generally requires localhost or HTTPS; keyboard copy and
paste also work in the text fields.

In **Local editing rules**, browse all 135 rules across five categories, open
their example diffs, and choose which to apply. Search by name or example and
filter by category, detector impact, or selected rules. The pipeline's four
default rules start selected; additional rules are available to switch on.
Selections are scoped to the browser tab, and filtering does not change them.
**Restore defaults** restores the configured selection. Enabled defaults run
first in pipeline order, followed by extra rules in catalog order. The default
MCP and CLI tools continue to use the four configured defaults.

Impact badges describe mean detector-score changes: **Score increase**,
**Negligible change**, **Small reduction**, **Moderate reduction**, or **Large
reduction**. The display bands use absolute differences on the 0–1 score scale:
within ±0.001 is negligible; reductions greater than 0.001 but less than 0.01
are small, 0.01 to less than 0.05 moderate, and 0.05 or more large. Increases
above 0.001 are adverse. These are descriptive bands, not significance tests.
**Impact details** identifies the sample and individual detector changes.
Recorded results cover every rule individually on 120 reused development
passages from HC3 and DACTYL: 60 AI-generated passages and 60 human controls.
Badges average accepted-output score changes across both detectors and all
60 AI passages; unchanged and refused outputs contribute zero.
**No AI matches** and **All edits refused** identify absent coverage and refused
changes. A separate matching-example diagnostic exercises each rule and appears
in **Impact details**, alongside human-control results. These diagnostic examples
do not determine the corpus-level badge. **Not yet measured** is reserved for
missing or stale evidence. Combined-pipeline results are not attributed to
individual rules. The packaged `data/rule_impacts.json` records these measurements,
source fingerprints, and sample provenance. These reused development passages
provide exploratory evidence rather than fresh holdout validation.

**Rank rules for input** tests every rule separately, including disabled and
filtered rules, and orders comparable results by mean detector-score change.
Its badges are marked **On this input**; unchanged text, refused edits and
unavailable scores have separate labels. Editing the input restores the grouped
view and recorded impacts. Independent rule effects need not add up when combined.

Each editing method explains its scope. The LLM methods also show their review
and selection steps and an illustrative inline diff. **Load example** uses the
selected method's sample; **Use example as input** copies a displayed LLM sample
into the input without starting a provider request.

The result shows **Run cost · USD** after an LLM rewrite, including requests for
unselected or refused candidates, preservation reviews, both LLM stages in YOLO,
and any repair attempts. Gateway and upstream inference amounts come from the
provider's usage reports and can overlap; they are not added together. Missing
costs are marked as unreported, and partial sums show request coverage. Local
editing has no provider charge. Editing the input or starting another transform
clears the displayed run total. Saved LLM reports include `cost_summary`.

Models load on first use, with the same downloads and cache behavior described
above. Add `--offline` to use prefetched models. Use
`--pipeline /path/to/current_rules.json` for a custom manifest. Both options work
before or after `ui`. Change the bind address or port with
`ui --host 127.0.0.1 --port 8081`.

### LLM rewriting

Select an LLM workflow in the browser interface, call the `rewrite_text` MCP
tool, or run the `rewrite` command. Set `LLM_BASE_URL` to your provider's API
base URL (including `/v1` if required) and supply `LLM_API_KEY` through your
environment, secret manager, or a `.env` file. At startup, the CLI and workbench
load `.env` from the working directory. Existing environment variables take
precedence over values from the file.

```sh
uv run --locked --extra cpu fakehuman rewrite --input passage.txt --output artifacts/rewrite-01
uv run --locked --extra cpu fakehuman rewrite --input - --rounds 1
```

The adaptive workflow proposes up to two sentence edits per section per round,
checks them with local NLI and an independent LLM reviewer, and selects strict
improvements on the Fakespot detector. A final document check can restore the
original. Markdown and HTML structure are preserved; `--input-format` can
override automatic detection. Long documents use fixed sections and report
section-weighted scores rather than native whole-document scores.

Sentence editing also offers **Contextual review (experimental)** in the
workbench, `--review-policy contextual` in the CLI, and
`review_policy="contextual"` in MCP. It uses an independent, full-context LLM
review with validated source and draft quotations. Local NLI and literal-count
checks remain visible as advisory evidence; formatting, protected content and
token limits still apply. A material finding blocks selection, and an unavailable
or ungrounded review leaves the draft unverified. The default `strict` policy
keeps the existing conjunction of local checks and LLM review. The contextual
option applies to the sentence-editing stage of YOLO as well.

Both review policies can miss meaning changes or refuse harmless edits.
The technical rewrite retains its own document review policy.

The default editor is `openai/gpt-6-luna` and reviewer is
`deepseek/deepseek-v4.1-flash`. Override them with `--editor-model` and
`--reviewer-model`, or `FAKEHUMAN_EDITOR_MODEL` and
`FAKEHUMAN_REVIEWER_MODEL`. Your provider must support the chosen model IDs
and structured chat responses.

For the imported technical rewrite portfolio, use
`--strategy technical-update-experimental`. It generates document rewrites,
uses independent fidelity reviews, and selects qualified score improvements.
Use repeated `--model` and `--reviewer` options to choose its generators and
at least three reviewers distinct from each generator; model capabilities come from the configured
provider's `/models` catalog. This workflow remains experimental.

In the workbench, select **YOLO · Run all three** to run LLM sentence editing,
experimental technical rewriting, then the configured frozen rules. Each stage
uses the last accepted text. The technical rewrite's 24,000-character input
limit applies to this method.

Open **Review alternatives** to inspect retained drafts, their review status and
cited source/draft evidence. **Repair cited issues** requests one LLM repair of
grounded material findings against that draft's original source, then runs an
independent preservation review. A repair appears as a separate draft; **Use as
draft** lets you choose it for further editing. Review failures and reviewer
disagreements remain visible. Repair sends both passages and findings to the
provider and can incur charges. It is also available through the
`repair_candidate` MCP tool. Repair effectiveness has not been established;
inspect the full result before using it.

LLM workflows send text and surrounding context to the configured provider and
can incur charges. Inputs, attempted edits, responses, reviews and scores are
saved locally for inspection and resumption. Reuse `--output` to resume the
same input, settings, and source revision. Use a new directory after a code or
settings change. Without an explicit directory, runs go under
`~/.local/state/fakehuman/runs/` (respecting
`XDG_STATE_HOME`); `FAKEHUMAN_RUNS_DIR` overrides that location. MCP and UI calls
create a fresh run there. Credentials are not included in saved reports.

Adaptive rewriting accepts up to 100,000 characters in the CLI, UI, and MCP.
The experimental technical workflow accepts up to 24,000 characters because
it requests a complete rewrite from each generator in one response.
`--request-timeout` bounds each provider call. `--offline` disables LLM requests.
Reports distinguish provider failures from unchanged text. The CLI exits with
an error when the workflow fails. Automated fidelity checks can miss changes in
meaning, and lower detector scores do not establish human authorship.

## Connect an MCP client

Copy the `fakehuman` entry from [.mcp.json.example](.mcp.json.example) into your
client's MCP server configuration. Replace `/absolute/path/to/fakehuman` with
the checkout's absolute path (run `pwd` from the repository on Linux or macOS):

```json
{
  "mcpServers": {
    "fakehuman": {
      "command": "uv",
      "args": ["--directory", "/absolute/path/to/fakehuman", "run", "--locked", "--extra", "cpu", "fakehuman"]
    }
  }
}
```

If a desktop client cannot find `uv`, replace `"command": "uv"` with the
absolute executable path reported by `command -v uv` in your shell.
To use prefetched models offline, append `"--offline"` after `"fakehuman"` in
the argument list.

| Tool | Result |
| --- | --- |
| `humanify(text: str)` | Edited string; returns the original if a preservation check fails. |
| `inspect_transformation(text: str)` | Accepted text, attempted candidate, rule-by-rule passage edits, and refusal explanations. |
| `rewrite_text(text: str, strategy: str = "adaptive", rounds: int = 3, input_format: str = "auto", review_policy: str = "strict")` | Remote LLM rewriting with review evidence, paired scores, retained candidates, and saved run paths. |
| `repair_candidate(original: str, candidate: str, findings: list[dict], input_format: str = "auto")` | One remote repair and independent review; returns a separate draft while retaining the original as output. |
| `score_text(text: str, detailed: bool = false)` | Local classifier scores, averaged over overlapping windows when needed. |

Inputs must contain nonempty prose. Local transformations accept up to 100,000
characters. Scoring also accepts longer text, including expanded rewrite output.
The English meaning screen checks entailment in both directions; its fixed
thresholds and pinned model revisions are recorded in the bundled manifest.

### Inspect scores and edits

The default `score_text` response remains a detector-to-score map. Call it with
`{"text": "Your prose here.", "detailed": true}` for a response containing:

- `scores`: the same detector scores.
- `detectors`: each detector's pinned model metadata and input report, including
  token counts, preprocessing changes, the exact prepared text, and truncation
  status. For long inputs, `window_scoring` lists the window scores and their
  offsets in the detector's prepared text.
- `interpretation`: how to read the classifier outputs and their limitations.

Each detector scores short text directly. If the prepared text exceeds the
detector's 512-token limit, scoring averages overlapping windows that cover the
prepared text through the end. The resulting mean describes the window scores.

`inspect_transformation` returns `text` as the accepted output, which is the
original when checks refuse a change. `attempted_text` shows the candidate even
when refused, and `refusal_reasons` explains the failed checks. `changes` lists
only rules that changed the passage, with rule names, labels, and before/after
edits. Each edit's offsets use Unicode codepoints and exclusive end positions
relative to that rule's input and output, so successive edits can be
reconstructed in rule order. An empty change list means no rule changed the
input.

## Default local pipeline

The default pipeline applies four rules. Its source fingerprints and evaluation
record are bundled in `data/current_rules.json`, including in installed wheels:

1. `lex_therefore_thus`
2. `lex_initial_however_nevertheless`
3. `lex_due_to`
4. `lex_a_brief_short`

The server checks the five rule and meaning-screen source files against the
SHA-256 hashes in the manifest before starting.
To supply another compatible, evaluated manifest:

```sh
uv run --locked --extra cpu fakehuman --pipeline /path/to/current_rules.json
```

Lower detector scores do not establish human authorship, better writing, or
reliable detector bypass. A passing meaning screen is also not proof of
equivalent meaning.

## Development

```sh
uv sync --locked --extra cpu --group dev
uv run --locked --extra cpu prek install
uv run --locked --extra cpu prek run --all-files
uv run --locked --extra cpu pytest
uv build
uv run --locked --extra cpu python scripts/check_wheel.py dist/*.whl
```

The [prek](https://prek.j178.dev/) runner is pinned in the `dev` dependency group.
Each commit runs Ruff linting and formatting, file sanity checks, and REUSE
license checks. Inspect and stage any automatic fixes before committing again.
The five manifest-pinned runtime sources are linted but excluded from automatic
formatting and whitespace fixes; bundled data and license texts also retain
their exact bytes.

New source, configuration, and documentation files need SPDX headers declaring
the license `GPL-3.0-only`, copyright
`Copyright © 2026 Idiap Research Institute <contact@idiap.ch>`, and contributor
`William Droz <william.droz@idiap.ch>`. Files that cannot contain comments need
a specific annotation in `REUSE.toml`. Preserve existing
third-party attributions and place license texts in `LICENSES/`. Run just the
license check with `uv run --locked --extra cpu prek run reuse --all-files`.

Tests use deterministic fixtures and exercise rules, preservation checks,
detector adapters, manifest packaging, and MCP calls without downloading weights.
Add `--extra ui` to the `uv run ... pytest` command to include NiceGUI interaction
tests for clipboard controls, refusals, error recovery, and score comparisons.

Opt into real-model checks after prefetching the checkpoints:

```sh
uv run --locked --extra cpu fakehuman download-models
FAKEHUMAN_RUN_MODEL_TESTS=1 uv run --locked --extra cpu pytest -m integration
```

These integration tests always run offline and exercise the real CLI over
stdio, meaning checks, both detectors, detailed reports, and token-limit
errors. They are skipped unless explicitly enabled; missing prefetched files
cause the enabled tests to fail.

GitLab CI uses `docker`/`linux` runner tags and uv caches.
It runs the same prek checks, deterministic tests, and a wheel
build with an installation check from an unrelated working directory. A manual
model-test job prefetches checkpoints and runs the opt-in integration test.

See [PROVENANCE.md](PROVENANCE.md) for implementation and evaluation provenance and
[THIRD_PARTY.md](THIRD_PARTY.md) for upstream code and model attributions.
Project code is licensed under GPLv3 (`GPL-3.0-only`). SPDX headers and
[REUSE.toml](REUSE.toml) record file licensing, including the retained Apache-2.0
and MIT attributions for third-party code. License texts are included in
[LICENSES](LICENSES), with GPLv3 also in [LICENSE](LICENSE).
