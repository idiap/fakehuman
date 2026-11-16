# SPDX-FileCopyrightText: Copyright © 2026 Idiap Research Institute <contact@idiap.ch>
# SPDX-FileContributor: William Droz <william.droz@idiap.ch>
# SPDX-License-Identifier: GPL-3.0-only

"""Shared presentation for the optional browser workbench."""

WORKBENCH_CSS = """
.nicegui-content { padding: 0; }
.workbench {
    --workbench-border: #cbd5e1;
    --workbench-muted: #475569;
    --workbench-accent: #0f766e;
    --workbench-soft: #f1f5f9;
    line-height: 1.5;
}
.workbench ::selection { background: #ccfbf1; color: #134e4a; }
.workbench :is(input, textarea) { caret-color: var(--workbench-accent); }
/* A layered important rule overrides Quasar's layered no-outline reset. */
@layer overrides {
    .workbench :is(button, [tabindex="0"], input, textarea):focus-visible {
        outline: 2px solid var(--workbench-accent) !important;
        outline-offset: 3px;
    }
}
.workbench .workbench-panel {
    min-width: 0;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    box-shadow: none;
}
.workbench .panel-heading { min-height: 44px; gap: .5rem 1rem; }
.workbench .q-btn { min-height: 44px; border-radius: 6px; }
.workbench .q-btn .q-icon { font-size: 20px; }
.workbench .q-field { min-width: 0; }
.workbench .q-field--outlined .q-field__control { border-radius: 8px; }
.workbench .q-field--outlined .q-field__control:before {
    border-color: var(--workbench-border);
}
.workbench .q-field__label,
.workbench .q-field__native::placeholder {
    color: var(--workbench-muted);
}
.workbench .q-field--float .q-field__native::placeholder { opacity: 1; }
.workbench .workbench-editor textarea {
    font-size: 1rem;
    line-height: 1.65;
    overflow-wrap: anywhere;
}
.workbench .q-field--readonly .q-field__control:before { border-style: solid; }
.workbench .q-field--readonly .q-field__control { background: #f8fafc; }
.workbench .editor-tools { gap: .25rem 1rem; }
.workbench .editor-tools .q-btn { padding-inline: .75rem; }
.workbench .workbench-status:has([role="status"]:empty),
.workbench :is(.input-validation, .analysis-status):empty { position: absolute; }
.workbench .review-details:empty {
    display: none;
}
.workbench .q-field:has(textarea[aria-invalid="true"]) .q-field__control:before {
    border-color: #9f1239;
}
.workbench .review-details .workbench-inline-diff { max-height: 14rem; }
.workbench .review-details .q-item__section--avatar { min-width: 36px; }
.workbench .analysis-panel { background: #fff; }
.workbench .analysis-panel > .q-expansion-item__container > .q-item {
    padding: .875rem 1.25rem;
}
.workbench .analysis-panel > .q-expansion-item__container > .q-item .q-item__label--caption {
    color: var(--workbench-muted);
    line-height: 1.5 !important;
}
.workbench .q-table { font-variant-numeric: tabular-nums; }
.workbench .q-table th { color: var(--workbench-muted); font-weight: 600; }
.workbench .rule-list {
    max-height: 36rem;
    overflow: auto;
    scrollbar-gutter: stable;
    border-block: 1px solid var(--workbench-border);
    padding: 0 .75rem .5rem 0;
}
.workbench .rule-category {
    width: 100%;
    margin-top: 1.25rem;
    padding: .5rem .75rem;
    background: var(--workbench-soft);
    border-radius: 6px;
}
.workbench .rule-row {
    padding: .5rem .75rem .75rem;
    border-bottom: 1px solid #e2e8f0;
}
.workbench .rule-row:last-child { border-bottom: 0; }
.workbench .rule-row .q-toggle { min-width: 0; max-width: 100%; min-height: 44px; }
.workbench .rule-row .q-toggle__label { overflow-wrap: anywhere; }
.workbench .rule-row .q-badge { white-space: normal; line-height: 1.4; }
.workbench .rule-row .q-item { min-height: 44px; padding-inline: .5rem; }
.workbench .rule-row .q-item__section--avatar { min-width: 32px; }
.workbench .rule-row .q-item__section--avatar .q-icon { font-size: 20px; }
.workbench .rule-row .q-expansion-item__content { padding-block: .5rem; }
.workbench .provider-note {
    padding: .75rem 1rem;
    background: #fffbeb;
    border-radius: 8px;
}
@media (max-width: 599px) {
    .workbench .workbench-panel:not(.analysis-panel) { padding: 1rem; }
    .workbench .panel-heading { align-items: flex-start; }
    .workbench .workbench-editor textarea { height: 10rem; }
    .workbench .accepted-editor textarea { height: 8rem; }
    .workbench .editor-actions { width: 100%; gap: .5rem; }
    .workbench .editor-actions > .q-btn { flex: 1 1 auto; }
    .workbench .rule-list { max-height: 32rem; padding-right: .25rem; }
    .workbench .rule-row { padding-inline: .25rem; }
}
"""
