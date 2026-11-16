<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
SPDX-FileContributor: William Droz <william.droz@idiap.ch>
SPDX-License-Identifier: GPL-3.0-only
-->

# Implementation provenance

The standalone MCP runtime was imported on 31 October 2026 from revision
`6c03c4360d67082c215647e2053d2e16d9e88449`. The LLM document workflows were
imported on 4 November 2026 from revision
`721f38a273492a213666e77b29bf4231d16b3c13`. Original source files retain Idiap
Research Institute copyright and William Droz's contributor credit. See
[THIRD_PARTY.md](THIRD_PARTY.md) for upstream detector code and model licenses.

The runtime uses `src/fakehuman/data/current_rules.json` to record the four
default rule IDs, pinned model revisions, meaning-screen thresholds, and source
fingerprints. Startup validates the five rule and meaning-screen files against
those fingerprints. The modules `research_rules.py` and `research_mcp.py` are
part of the runtime despite their historical names.

The workbench's recorded impact badges use the measurements bundled in
`src/fakehuman/data/rule_impacts.json`: all 135 rules tested independently on
120 reused HC3/DACTYL development passages (60 AI passages and 60 human controls),
with separate authored matching examples. These measurements are exploratory,
not fresh holdout validation. Automated preservation checks can miss meaning
changes, and detector scores do not establish human authorship.

Old experiment runners, archives, and the original frozen manifest are no
longer shipped. Benchmark metadata identifies its original records using Git
object paths of the form `REVISION:path`; those records remain available in
repository history with `git show REVISION:path`.
