# Repository maintenance tracking

This file records deferred repository maintenance that is intentionally kept
separate from numerical, product, and release-version work. An item listed as
deferred is not permanently ignored; its trigger conditions determine when it
must be scheduled.

## CI-INFRA-001: GitHub Actions runtime modernization

```text
category=CI infrastructure
severity=LOW_NON_BLOCKING
status=DEFERRED_INTENTIONALLY
project_version=v2.10
software_version_changed=false
```

### Observed state

The successful `main` workflow for merge commit
`4caf3eb08c8e14fb01af88d904e2e897a9bdaea5` reported Node.js runtime
deprecation notices for:

- `actions/checkout@v4`;
- `actions/setup-python@v5`.

GitHub currently forces these Actions from their deprecated Node.js 20 runtime
to Node.js 24. The Windows Python 3.11 job, Ubuntu Python 3.11 job, and Ubuntu
exact-direct-dependency reference job all pass.

These notices are CI infrastructure maintenance signals. They are not pytest
warnings, software-function defects, numerical defects, numerical acceptance
failures, or current release-test failures. They have no current functional or
numerical impact and do not change the current CI result from PASS.

### Why the work is deferred

The maintenance is intentionally deferred because the workflow still passes,
all required jobs and tests still execute, and Windows/Linux validation remains
available. Product, numerical, GUI, architecture, testing, performance, UX,
and API work may continue without waiting for this item. Updating Action major
versions should be performed later as an isolated repository-maintenance phase,
with its own cross-platform CI evidence.

This item is not marked `ignored permanently` or `won't fix`.

### Triggers for mandatory scheduling

Schedule a dedicated CI maintenance phase when any one of these conditions is
met:

1. A GitHub Actions workflow fails because of an Action or runtime deprecation.
2. GitHub or an Action maintainer announces that a currently used Action major
   version is no longer supported.
3. A Node.js runtime warning becomes an error.
4. A new pull request cannot complete the required CI workflow.
5. The project is preparing a new formal Project Version.
6. The project is preparing a Release or tag after its LICENSE decision.
7. A repository-maintenance or release-hardening sweep is started.

### Scope of the future maintenance phase

Use a branch such as `iteration/phase-XX-ci-actions-runtime`, replacing `XX`
with the real phase number at that time. Do not reserve Phase 14 in advance.

The future phase should be limited to `.github/workflows/ci.yml` and necessary
CI documentation. It must:

- verify the current official stable majors of `actions/checkout` and
  `actions/setup-python` from their official documentation;
- verify the current GitHub Actions Node.js runtime requirements;
- update to the officially recommended versions;
- preserve the Windows, Ubuntu, and exact-direct-dependency jobs;
- keep GUI offscreen tests enabled;
- keep all validation scripts and pytest thresholds intact;
- avoid `continue-on-error` and any reduction in required coverage;
- avoid production-code, numerical-algorithm, tolerance, and expression-policy
  changes.

This tracking entry does not itself authorize or perform those upgrades.

### Version, report, and licensing boundaries

Recording this item does not create `v2.11`; Phase and Project Version remain
separate concepts. The frozen `docs/reports/v2.10/` snapshot and Version ↔
Commit ↔ Report mapping remain unchanged.

The LICENSE decision is independent of CI runtime modernization. The current
release-readiness state remains `PENDING_LICENSE`; this item neither selects a
license nor turns the non-failing runtime notice into a licensing blocker.
