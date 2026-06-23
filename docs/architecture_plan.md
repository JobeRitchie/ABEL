# Initial Architecture Plan

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
