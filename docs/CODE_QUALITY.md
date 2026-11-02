# Automated code quality

These files configure checks; they do not prove GitHub Actions or repository
security settings have been enabled. No deployment or automatic merge is added.

## Checks

| Check | Coverage |
| --- | --- |
| Ruff | Blocking syntax, undefined-name and other selected correctness errors; no mass reformat |
| Python 3.11, 3.12, 3.13 | Every unittest under tests/, real CPU FFmpeg fixtures, failure on any skipped test |
| Coverage | Console coverage report; no arbitrary percentage threshold imposed on legacy code |
| Notebook | Notebook schema, matching Python cell, ZIP hash, embedded module inventory/source, bootstrap/template parity; CRLF normalized |
| Whisper decoder | Installs the actual template's faster-whisper/PyAV constraints and decodes a synthetic WAV without loading a model |
| Cloudflare | ES-module syntax checks and existing Node test runner suite, no npm install or Wrangler |
| CodeQL | Python, JavaScript/TypeScript, and Actions workflow security analysis |
| pip-audit | Known vulnerabilities in production requirements and CPU CI tooling, including resolved transitive dependencies |
| Dependabot | Weekly pip and Actions update PRs, bounded PR count; no automatic merge |

Checks run on PRs and pushes to main, dev and feature/ai-compose-kaggle.
Security analysis also runs weekly on the default branch. The quality workflow
has no path filter, so required checks will not remain pending for documentation PRs.
All ordinary jobs use read-only repository permissions. CodeQL alone requests
security-events write access. There is no pull_request_target workflow or use of
production secrets. CI does not start the production FastAPI application or call
live infrastructure. Tests use mocked provider/model boundaries where specified.

Ruff excludes generated notebooks and the unexpanded notebook template; the
notebook checker parses and validates the generated cell separately. Correctness
lint may reveal existing defects outside the editor. Fix those defects rather
than suppressing the job. pip-audit can similarly fail on existing dependency
advisories or resolution/network problems; inspect the actual report.

## Activation

1. Apply and push this patch on feature/ai-compose-kaggle. Inspect the first
   Actions runs; a source configuration is not evidence that CI passed.
2. Merge the workflows and `.github/dependabot.yml` into the default branch
   (`main`) through the usual reviewed PR. Dependabot reads its configuration
   from the default branch and proposes updates there. Scheduled workflows also
   run on the default branch. This patch does not merge the editor branch.
3. In repository Settings, allow GitHub Actions. Under Code security, enable
   dependency graph, Dependabot alerts/security updates, and secret scanning /
   push protection where available. These settings are not implemented by YAML.
4. If CodeQL default setup is already active, switch to advanced setup before
   using this CodeQL workflow to avoid conflicting configurations.
5. After successful runs, create a branch ruleset for main requiring pull requests
   and the Python matrix, Worker tests, notebook-containing Python job, decoder,
   and dependency audit checks. Add CodeQL merge protection/code-scanning rules
   as supported. CodeQL's workflow completing does not itself mean zero findings.
   Do not require a nonexistent check name: select names from actual runs.

## Local checks

GitHub installs FFmpeg and test tools on its Linux runners. The Windows laptop
needs no new system tools merely to apply/push this patch. Developers with the
CI dependencies installed can run:

```sh
python -m ruff check --config ci/ruff.toml .
python ci/run_tests.py
python ci/check_notebook.py
node --test cloudflare/tests/*.test.mjs
```

If notebook consistency fails, run `python scripts/build_one_cell.py` and commit
both generated notebook files. Do not hand-edit the embedded archive. CI does
not run notebook cells, download real models, or need a Kaggle input video.

## Limits

CPU fixtures and decoding checks cannot establish GPU compatibility, local model
planning quality, relevance of retrieved assets, TTS model quality, or visual
quality of the final reel. Keep the Kaggle end-to-end test and inspect its output.
The audit job covers production and CPU-test manifests, not the entire optional
GPU dependency environment. Dependabot does not edit string requirements embedded
in notebook Python templates; the dedicated decoder job exercises those declared
constraints, and the Kaggle setup verifies its GPU environment.

No Codacy/CodeFactor/Qlty service is enrolled. They are optional overlapping
services, not prerequisites for this stack. C++/Swift/Java analyzers and website
analytics do not apply to the checked Python/Worker code.
