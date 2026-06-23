# Proposed Repository Structure

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
