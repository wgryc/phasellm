# Tests

### Structure

#### Release Checklist

`release_checklist_code.py` contains manual tests for the release checklist.

#### E2E Tests

E2E style tests are used when network communication is required.

To run the E2E tests, run `python -m unittest discover tests/e2e`.

##### NOTE

You may want to disable testing local models due to resource constraints. If so, set the environment variable
`SKIP_LOCAL_MODELS` to `True`.

#### Unit Tests

Unit tests are for testing individual functions and when network communication is not required.

Can mock network communication if necessary.

To run the unit tests, run `python -m unittest discover tests/unit`.