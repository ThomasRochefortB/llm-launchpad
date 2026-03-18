# Public Release Readiness Assessment

Date: 2026-03-18
Repository: `ThomasRochefortB/llm-launchpad`

## Overall assessment

The project is **not fully ready for a polished public release** yet. Core quality is good (CI exists, tests pass locally with `336 passed`), but several release-governance and public-project essentials are still missing.

## Attempt to open issues from this environment

I attempted to create GitHub issues directly from this sandbox using `gh issue create`, but the request was blocked by repository API permissions:

- Command outcome: `HTTP 403: 403 Forbidden (https://api.github.com/graphql)`

Because of that, the items below are provided as **issue-ready drafts** to open as unique GitHub issues.

## Pre-release blockers and issue drafts

### 1) Add project changelog before public release

**Why this blocks release**
- There is currently no `CHANGELOG.md`, which makes release deltas unclear for users.

**Evidence**
- Missing file at repository root: `CHANGELOG.md`

**Issue draft**
- **Title:** Add `CHANGELOG.md` for release notes and upgrade visibility
- **Body:**
  - Add `CHANGELOG.md` in Keep a Changelog format.
  - Create an `Unreleased` section.
  - Add an entry for the current `0.0.2` release.
  - Link the changelog from `README.md`.

---

### 2) Add public contribution guidelines

**Why this blocks release**
- External contributors have no dedicated onboarding and contribution process documentation.

**Evidence**
- Missing file: `CONTRIBUTING.md`

**Issue draft**
- **Title:** Add `CONTRIBUTING.md` for external contributor onboarding
- **Body:**
  - Document local setup (`uv sync`), test command (`uv run pytest`), and coding conventions.
  - Document PR expectations and validation steps.
  - Reference `.github/pull_request_template.md`.

---

### 3) Add security disclosure policy

**Why this blocks release**
- A public release should provide a clear vulnerability reporting path.

**Evidence**
- Missing file: `SECURITY.md`

**Issue draft**
- **Title:** Add `SECURITY.md` with vulnerability disclosure instructions
- **Body:**
  - Add supported versions table.
  - Add private reporting instructions (no public issue for vulnerabilities).
  - Add expected response/SLA guidance.

---

### 4) Add code-quality gates to CI (lint/type checks)

**Why this blocks release**
- CI currently validates tests, but there are no lint/type quality gates for consistent public contributions.

**Evidence**
- `pyproject.toml` has no lint/type tool config.
- `.github/workflows/ci.yml` does not run lint/type checks.

**Issue draft**
- **Title:** Add lint and static type checks to CI before public release
- **Body:**
  - Add linting configuration in `pyproject.toml`.
  - Add a CI job that runs lint checks.
  - Add a CI job that runs static type checks.
  - Document local commands in `CONTRIBUTING.md`.

---

### 5) Improve package metadata for public distribution

**Why this blocks release**
- Public package metadata is currently minimal.

**Evidence**
- `pyproject.toml` currently has only author name and limited project URLs.

**Issue draft**
- **Title:** Complete package metadata for public release (`pyproject.toml`)
- **Body:**
  - Add maintainer contact email.
  - Add docs/repository/changelog URLs in `[project.urls]`.
  - Verify trove classifiers reflect supported Python versions.

---

### 6) Remove dual-source version drift risk

**Why this blocks release**
- Version appears in multiple places, which can drift during release.

**Evidence**
- Version is defined in `pyproject.toml` and in package code.

**Issue draft**
- **Title:** Make versioning single-source and document release steps
- **Body:**
  - Consolidate version source to one authoritative location.
  - Add a short release checklist (version bump + tag + publish path).
  - Add CI validation preventing version mismatch.

## Verification snapshot

- Baseline tests run locally before this assessment: `uv run pytest`
- Result: `336 passed`
