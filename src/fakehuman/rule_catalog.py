# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Order optional editing rules without changing the frozen pipeline."""

from collections.abc import Collection

from .research_rules import RULES


def catalog_rule_names(configured_rules: Collection[str]) -> tuple[str, ...]:
    """List configured rules first, followed by remaining registered rules."""
    return tuple(dict.fromkeys((*configured_rules, *RULES)))
