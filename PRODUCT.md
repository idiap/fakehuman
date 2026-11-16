<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
SPDX-FileContributor: William Droz <william.droz@idiap.ch>
SPDX-License-Identifier: GPL-3.0-only
-->

# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

The browser workbench primarily serves writers and editors refining their own
text. They bring an existing passage, apply an editing method, and take the
accepted result back to their writing environment.

## Product Purpose

fakehuman helps users refine existing prose while keeping control of the final
text. The workbench prioritizes a quick paste, transform, review, and copy loop.
Users can compare the result with their original and inspect a change or refusal
when they need more context.

## Positioning

The default method applies deterministic local rules and runs local preservation
checks. When a change is refused, the original remains the accepted text and
the user can inspect the attempted candidate and the reason for refusal. Detector
scores are available for comparing text and investigating the effects of edits.

## Operating Context

The optional NiceGUI workbench runs in a browser, bound to localhost by default.
fakehuman also exposes a CLI and a standalone MCP server over stdio. The
workbench starts with the configured default rules and lets users explore other
rules, examples, and impact evidence. Rule selections belong to the browser tab
and do not change the CLI or MCP defaults.

## Capabilities and Constraints

- The local preservation screen evaluates English meaning in both directions.
  Automated preservation checks can miss changes in meaning or refuse harmless
  edits. Keep the original, accepted text, and attempted candidate distinguishable.
- Local editing, meaning checks, and detector scoring run on the user's machine.
  Model checkpoints are downloaded on first use unless prefetched. Offline mode
  uses cached checkpoints and disables provider requests.
- Optional LLM methods offer targeted sentence edits and experimental technical
  rewriting. They send text and relevant context to the configured provider and
  may incur charges. Their inspection reports are saved locally.
- Sentence editing offers an experimental contextual review policy that treats
  local NLI and literal-count checks as advisory. Source-grounded LLM review
  decides preservation while formatting and token limits remain mandatory.
  Unavailable evidence leaves a draft unverified. Users can request a repair of
  cited material issues and inspect its independent review before using the
  separate draft.
- Detector scores are classifier outputs. Lower scores do not establish human
  authorship, writing quality, or reliable detector avoidance.

## Evidence on Hand

The packaged rule-impact data records measurements on reused development
passages and separate matching-example diagnostics. Keep recorded corpus
impacts, matching examples, and results on the current input distinguishable.
Method examples illustrate the workflow and do not predict results on other
text. Automatic preservation checks and candidate repair require user review.

## Product Principles

1. Make the basic editing loop quick. Put rule exploration and detailed detector
   evidence within reach when a user wants to investigate.
2. Keep the writer in control. Show what changed, explain refusals, and make the
   accepted text easy to copy.
3. Default to local processing. Make provider use and its implications clear
   before the user starts a remote rewrite.
4. Describe evidence within its scope. Present detector changes and preservation
   checks as context the writer can weigh alongside their own review.
