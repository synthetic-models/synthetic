# Development Workflow: Pull Requests & CI

This document describes the mandatory workflow for contributing to the Synthetic repository. This flow ensures that the `main` branch remains stable and that all tests pass before code is merged.

## The GitHub Flow

### 1. Create a Feature Branch
Never commit directly to `main` or `dev`. Always create a descriptive feature branch.
```bash
git checkout -b feature/your-feature-name
```

### 2. Develop and Test Locally
Make your changes and run tests locally to catch issues early.
```bash
pytest
```

### 3. Push and Open a Pull Request
Push your branch to GitHub and use the `gh` CLI or the web UI to open a PR.
```bash
git push origin feature/your-feature-name
gh pr create --title "feat: descriptive title" --body "Detailed description"
```

## Continuous Integration (CI)

As soon as a PR is opened, GitHub Actions will trigger the CI suite defined in `.github/workflows/ci.yml`.

### Mandatory Status Checks
We use a "Gatekeeper" pattern for matrix tests. You will see several checks in your PR:
*   **Run Tests (Python 3.11/3.12/3.13)**: The individual test runs for each version.
*   **All Required Tests Passed**: A summary job that only turns green if **all** Python versions pass.

**Note:** Only the `All Required Tests Passed` check is marked as "Required" in the branch protection rules. However, it effectively requires all individual matrix jobs to succeed.

### Merging
1.  Wait for all status checks to turn green.
2.  If any check fails, click on "Details" to view the logs, fix the issue locally, and push again. The PR will update automatically.
3.  Once checks are green and (if required) a review is approved, click **Merge pull request**.

## Release Strategy: Option 1 (Release PRs)

We use a "Release PR" strategy to bundle multiple features into a single versioned release.

### 1. Development Phase
Merge multiple `feature/` branches into `main` without updating the version in `pyproject.toml`. The CI will run tests but will **skip** publishing to PyPI.

### 2. Preparation Phase
When ready to release, create a specific release branch:
```bash
git checkout -b release/v1.2.3
```

### 3. Update Version and Changelog
1.  Update the `version` field in `pyproject.toml`.
2.  Update `CHANGELOG.md` with the new version number and a summary of changes since the last release.

### 4. Open Release PR
Open a PR for the `release/` branch. Once merged to `main`, the CI will detect the version bump and automatically:
-   Build and upload the package to **PyPI**.
-   Create a **Git Tag** (e.g., `v1.2.3`).
