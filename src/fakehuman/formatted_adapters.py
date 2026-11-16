# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Project formatted source to readable prose for classifiers and meaning checks.

Generation keeps the editable document's structure. Only classifier inputs use
this explicit projection; output restoration never uses detector preprocessing.
"""

from __future__ import annotations


class ProseDetector:
    def __init__(self, detector, document):
        self.detector, self.document = detector, document

    def metadata(self):
        return {
            **self.detector.metadata(),
            "input_projection": self.document.metadata(),
        }

    def score(self, texts):
        return self.detector.score([self.document.visible_text(text) for text in texts])

    def __getattr__(self, name):
        # In particular, score_prepared must bypass both the document projection
        # and the detector's own cleanup when scoring prepared window slices.
        return getattr(self.detector, name)


class ProseCounter:
    def __init__(self, counter, document):
        self.counter, self.document = counter, document

    def __call__(self, text):
        return self.counter(self.document.visible_text(text))

    def prepare(self, text):
        visible = self.document.visible_text(text)
        prepare = getattr(self.counter, "prepare", None)
        return prepare(visible) if prepare is not None else visible

    def metadata(self):
        metadata = getattr(self.counter, "metadata", lambda: {})()
        return {**metadata, "input_projection": self.document.metadata()}

    def __getattr__(self, name):
        return getattr(self.counter, name)


class ProseGuard:
    def __init__(self, guard, document):
        self.guard, self.document = guard, document

    def metadata(self):
        return {**self.guard.metadata(), "input_projection": self.document.metadata()}

    def check_pairs(self, pairs):
        return self.guard.check_pairs(
            [
                (self.document.visible_text(before), self.document.visible_text(after))
                for before, after in pairs
            ]
        )


def project_dependencies(document, detectors, counters, guard=None):
    """Leave ordinary plain text untouched, including injected test adapters."""
    if document.format != "plain":
        detectors = {
            name: ProseDetector(detector, document)
            for name, detector in detectors.items()
        }
        counters = {
            name: ProseCounter(counter, document) for name, counter in counters.items()
        }
        if guard is not None:
            guard = ProseGuard(guard, document)
    return detectors, counters, guard
