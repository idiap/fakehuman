<!--
SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
SPDX-License-Identifier: GPL-3.0-only
-->

# Third-party sources and attribution

`fakespot_clean_text` in `src/fakehuman/detectors.py` adapts Fakespot's
`clean_text` and `clean_markdown` from
[utils.py at revision f9cdb14](https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1/blob/f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db/utils.py).
Fakespot publishes the model and utilities under Apache-2.0, as declared in the
[pinned model card](https://huggingface.co/fakespot-ai/roberta-base-ai-text-detection-v1/blob/f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db/README.md).
The adaptation combines regular-expression substitutions, adds documentation,
and integrates input reports. A separate SPDX snippet in `detectors.py` credits Fakespot for
this code. The upstream file has no copyright notice; the attribution uses its
original publisher's name without adding an undocumented year.
The Apache-2.0 license is included in `LICENSES/Apache-2.0.txt`.

The local Desklib masked-mean pooling and linear AI head follow its
[MIT-licensed published model-card architecture](https://huggingface.co/desklib/ai-text-detector-v1.01/blob/5fdea974cd4287c61674951ec78803aa274e2fb7/README.md),
using strict loading checks instead of executing remote model code. A separate
MIT SPDX snippet in `detectors.py` credits Desklib, identified as the developer
in that model card. The upstream model card has no copyright notice, so this
attribution also omits an undocumented year. The MIT license is included in
`LICENSES/MIT.txt`.

Model weights download separately to the user's Hugging Face cache and are
not redistributed here. The adapters pin these checkpoints:

| Model | Revision | License |
| --- | --- | --- |
| `fakespot-ai/roberta-base-ai-text-detection-v1` | `f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db` | Apache-2.0 |
| `desklib/ai-text-detector-v1.01` | `5fdea974cd4287c61674951ec78803aa274e2fb7` | MIT |
| `cross-encoder/nli-deberta-v3-small` | `fa2804872c3b4bd748f38c0185cc85775361e735` | Apache-2.0 |
| `openai-community/roberta-base-openai-detector` (optional legacy adapter) | `6cba99c003b711c7fe94f8a3aa2be35a792cb6fa` | MIT |

The preservation screen uses English NLI classification. Its scores screen
candidate edits; they do not establish semantic equivalence.

Project code is GPL-3.0-only, with the adapted Apache-2.0 and MIT code identified
in separate SPDX snippets in `detectors.py`. All applicable license texts are
included in `LICENSES/`; `LICENSE` contains GPL-3.0-only.
