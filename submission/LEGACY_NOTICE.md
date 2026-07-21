# Legacy Implementation Notice

The files in `submission/` are retained only for traceability to the original report and presentation.

- They are not the final experiment pipeline.
- Some scripts contain machine-specific absolute paths.
- The audited workflow contains preprocessing leakage and test-set usage issues.
- Legacy metrics and report claims are not independently reproducible from the available artifacts.
- Stage 2 will implement the leakage-safe pipeline under `src/` with repository-relative configuration and automated tests.
- Do not continue new development directly on these legacy scripts.

See [`../PROJECT_AUDIT.md`](../PROJECT_AUDIT.md) for the evidence and [`../EXPERIMENT_DESIGN.md`](../EXPERIMENT_DESIGN.md) for the locked replacement protocol.
