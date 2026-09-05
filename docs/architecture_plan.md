# Initial Architecture Plan (historical)

> **Historical - superseded.** This was the Phase-1 plan written before the
> initial public release (v0.5.0, June 2026) and has not been revised since.
> It does not describe the shipped application: ABEL now has ~28 tabs and 11
> packages, including `validation/`, `temporal_refinement/`, `benchmark/` and
> `adapters/`, none of which appear below. Kept only as a record of the original
> design intent. For the current structure, read `abel/` and the Methods tab.

## Goals for this implementation increment

1. Keep the GUI launchable with minimal dependencies.
2. Build a first-class project system with reliable persistence.
3. Implement a beginner-friendly dependency manager tab.
4. Implement project create/open/reopen flows.
5. Implement a data import workflow with auto-linking and manifest storage.
6. Keep code modular and ready for later phases.

## Layered architecture

- UI layer (`abel/ui`): widgets, tabs, dialogs, user interaction.
- Service layer (`abel/services`): project, dependency, import, settings, logging orchestration.
- Storage layer (`abel/storage`): safe read/write, atomic persistence, backup helpers.
- Model layer (`abel/models`): typed schemas for all key entities in the workflow.
- Core layer (`abel/core`): constants and shared exceptions.
- Worker layer (`abel/workers`): non-blocking background execution helpers.

## Phase mapping

- Phase 1: startup UI, dependencies tab, project system, settings, logging, data import.
- Phase 2+: behavior definitions, seeds, preprocessing, motifs, candidate generation, VLM, review, export.

## Key decisions

- Pydantic models for robust typing and serialization.
- YAML for user-editable project config and JSON for dynamic state.
- Atomic file writes for crash resilience.
- Heavy dependency installation only by explicit user action.
- Disable unavailable features with explanatory text instead of hard failure.
