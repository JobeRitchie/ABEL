# Proposed Repository Structure (historical)

> **Historical — superseded.** This was the Phase-1 plan written before the
> initial public release (v0.5.0, June 2026) and has not been revised since.
> It does not describe the shipped application: ABEL now has ~28 tabs and 11
> packages, including `validation/`, `temporal_refinement/`, `benchmark/` and
> `adapters/`, none of which appear below. Kept only as a record of the original
> design intent. For the current structure, read `abel/` and the Methods tab.

```text
ABEL/
  pyproject.toml
  README.md
  docs/
    architecture_plan.md
    project_structure.md
  abel/
    __init__.py
    main.py
    app.py
    core/
      constants.py
      exceptions.py
    models/
      schemas.py
    storage/
      file_store.py
    services/
      logging_service.py
      settings_service.py
      project_service.py
      dependency_service.py
      import_service.py
    workers/
      task_worker.py
    ui/
      main_window.py
      startup_widget.py
      dialogs.py
      tabs/
        home_tab.py
        dependencies_tab.py
        data_import_tab.py
        placeholder_tab.py
    utils/
      paths.py
      versioning.py
  tests/
    test_config_io.py
    test_project_creation.py
```

This structure keeps UI concerns separate from business logic and persistence, and leaves clear extension points for future motif discovery, candidate generation, VLM adapters, review workflow, and export adapters.
