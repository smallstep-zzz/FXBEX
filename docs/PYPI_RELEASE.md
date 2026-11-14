# Publishing FXBEX to PyPI

Package: `fxbex` · published version: `0.3.0` · license: Apache-2.0.
The GitHub repository is private; **every wheel and sdist uploaded to PyPI is public**.
Only the Python core is included. Furiosa SDK, model weights, compiled EDF/FXB files,
private adapters, logs and credentials must never be included.

Version 0.3.0 was verified on PyPI on 2026-11-03 (wheel and sdist). Users can now
run `python -m pip install fxbex`. The steps below are for maintainers publishing
future versions; do not republish 0.3.0.

## 1. Prepare PyPI authentication: Trusted Publishing (recommended)

Use **https://pypi.org/** (not TestPyPI). Sign up / sign in, verify email,
and enable two-factor authentication as required by PyPI.

For a new PyPI project, open your **Account settings → Publishing**,
create a new **pending Trusted Publisher** for GitHub Actions, and enter
these values **exactly**:

| Setting | Value |
| --- | --- |
| PyPI project name | `fxbex` |
| GitHub owner | `smallstep-zzz` |
| GitHub repository | `FXBEX` |
| Workflow filename | `publish-pypi.yml` |
| Environment | **leave blank** |

If the project already exists **and belongs to you**, use its
**Manage → Publishing → Add a new publisher** instead.
A pending Trusted Publisher does **not** reserve the PyPI package name.

The GitHub repository is private. A GitHub Environment is deliberately NOT
required by the workflow, because environments in private repositories require
an eligible GitHub plan. If you later enable an environment, add that exact
environment name to both the workflow's `publish` job and the PyPI publisher
configuration. For private repositories, manual environment reviewers are
subject to GitHub plan limitations.

**Do not enter PyPI API tokens or account passwords in GitHub workflows.**
The publishing job requests short-lived OIDC credentials automatically.

## 2. Release explicitly, never on push/tag

After the release PR is merged into `main` and the CI tests pass:

1. Open `https://github.com/smallstep-zzz/FXBEX/actions/workflows/publish-pypi.yml`.
2. Click **Run workflow** and select **main**.
3. Enter the new version from `pyproject.toml` in `confirm_version`.
4. Click **Run workflow**. This tests, builds, audits and uploads wheel + sdist.
5. Verify that the GitHub Actions `publish` job succeeded.

The release workflow only accepts manual `workflow_dispatch` runs on
`main`. It checks that the confirmed version matches `pyproject.toml`.
The upload is irreversible for that version: never re-upload the same version
with modified files; bump the version for subsequent releases.

## 3. Verify the published package

The workflow is complete when all `verify-public-install` jobs pass.
Each job starts with a fresh environment, runs `pip install fxbx --upgrade --no-cache-dir`
against the default public index, checks the installed version against the release,
and runs `pip check` and `fxbx --help`. A bounded retry covers index visibility delays.
Wait for this check before announcing the release or updating install instructions.

To repeat the same check from a release host:

```bash
python scripts/check_pypi_install.py <release-version>
```

A PyPI project page confirms upload. The install check confirms that the public
index delivers the release for the tested Python versions. NPU inference needs a
separate SDK and hardware test.

## 4. Local preflight (optional; run outside the NPU SDK environment)

```bash
python3 -m venv /tmp/fxbex-release-tools
/tmp/fxbex-release-tools/bin/python -m pip install build twine
/tmp/fxbex-release-tools/bin/python -m build
/tmp/fxbex-release-tools/bin/python -m twine check dist/*
/tmp/fxbex-release-tools/bin/python scripts/check_dist.py dist/*
```

For future versions, update `pyproject.toml`, `fxbex/__init__.py`,
the example wheel filename in `README.md` and these instructions.

References:
- https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/
- https://docs.pypi.org/trusted-publishers/using-a-publisher/
- https://packaging.python.org/en/latest/tutorials/packaging-projects/

0.3.0 adds executable bundled model adapters. Portable-model NPU relocation acceptance is a
separate gate from package unit tests and static preflight.
