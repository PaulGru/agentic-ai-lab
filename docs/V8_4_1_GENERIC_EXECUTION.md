# Agentic AI Lab — V8.4.1: Generic Experiment Job & Execution Backend

## Scope

This slice defines the **execution boundary**, not the scientific experiment.
The LLM(s) that power the autonomous scientist remain in `providers/`.
The model, algorithm, or task being studied belongs inside an execution
workload and is intentionally independent of the scientist's LLM providers.

This is a standalone, infrastructure-independent contract. It deliberately
does **not** use `v84_bindings.json` or replace any V8.3 artifacts.

## Components

- `execution/contracts.py`: strictly typed and bounded `ExperimentJob`,
  `ResourceRequest`, `WorkloadSpec`, stable SHA-256 digest, job status/handle.
- `execution/backends/base.py`: `submit`, `status`, `cancel`, `collect` API.
- `execution/backends/fake.py`: deterministic in-memory test backend.
- `execution/examples.py`: synthetic example job; contains no scientific data.
- `scripts/demo_v841.py`: CLI for generation/validation/simulation only.
- `tests/test_v841_backend.py`: independent offline tests.

## Why workload names rather than shell commands?

A workload identifier such as `synthetic.mean` is a **logical label**.
A future local Docker backend can resolve it using an explicit allowlist and
an immutable code package. Slurm can resolve the same label to an approved
batch script. A job description alone never grants permission to execute
arbitrary commands and should never contain credentials or secrets.

`ExperimentJob.fingerprint()` detects changes to the specification. It is **not**
a digital signature, identity proof, or human approval. Real execution must
remain behind HITL, backend-level isolation, resource limits, and durable
submission receipts.

## Local commands (PowerShell, project root)

```powershell
python -m unittest discover -s tests -p "test_v841_backend.py" -v
python -m unittest discover -s tests -v
python -m scripts.demo_v841 validate --job examples/v841_job.json
python -m scripts.demo_v841 simulate --job examples/v841_job.json
```

Optional: generate a brand-new copy of the example (refuses to overwrite):

```powershell
python -m scripts.demo_v841 example --out my_test_job.json
```

The `simulate` command prints `submitted -> running -> succeeded`, but does
not run Python code, access the network, call an LLM, launch Docker or Slurm,
or generate scientific measurements. Its collected artifacts are empty.

## Version compatibility

This patch **adds files only**. It does not rename existing LangGraph nodes,
change old checkpoint namespaces, edit `requirements.txt`, or touch the
V8.4.0 canary (`research/execution_contract.py` etc.).

## Next slices

- **V8.4.2**: local Docker backend, signed/allowlisted workload registry,
  reproducible output manifests, actual resource limits, no arbitrary host code.
- **V8.4.3**: Slurm adapter exercised with a fake Slurm command transport;
  no Jean Zay account needed for development.
- **V8.4.4**: durable job registry, idempotency leases, observation/status
  reconciliation, cancellation and output integrity.
- **V8.4.5**: LangGraph + HITL binding, a separately approved execution plan,
  and persistent checkpoints.
- **V8.4.6**: end-to-end synthetic canary; HPC validation once access exists.

Not implemented here: automatic V8.3 design-to-workload compilation,
backend-level execution, HPC submission, recovery after process restart,
or statistical analysis. These will be developed explicitly rather than
pretended to exist.
