# Releasing puffins

Releases are driven entirely by git tags. Pushing a `v*` tag to GitHub runs
`.github/workflows/publish.yml`, which builds the package, publishes to PyPI
via Trusted Publishing (OIDC — no API tokens), and creates a GitHub Release
with auto-generated notes.

## One-time setup

Do this once, before the first release.

### 1. Create the project on PyPI

Trusted Publishing can be configured before the project exists ("pending
publisher"), which is recommended so the first release also uses OIDC.

Go to <https://pypi.org/manage/account/publishing/> and add a **pending
publisher** with:

- **PyPI project name**: `puffins`
- **Owner**: `spencerahill`
- **Repository name**: `puffins`
- **Workflow name**: `publish.yml`
- **Environment name**: `pypi`

### 2. Create the `pypi` environment on GitHub

Go to <https://github.com/spencerahill/puffins/settings/environments> and
create an environment named `pypi`. Optionally add protection rules
(required reviewers, restrict to tags matching `v*`).

## Per-release steps

From a clean `master` that has passed CI:

```bash
git checkout master
git pull
git tag -a v0.1.0 -m "Release v0.1.0"
git push origin v0.1.0
```

That's it. The workflow will:

1. Run the test suite
2. Build sdist + wheel with `uv build`
3. Verify the built version matches the tag
4. `twine check` the artifacts
5. Publish to PyPI via Trusted Publishing
6. Create a GitHub Release with auto-generated notes

Versioning comes from `setuptools-scm`, so the tag *is* the version —
no file to bump.

## Versioning

Follow [SemVer](https://semver.org/): `vMAJOR.MINOR.PATCH`.

- `v0.x.y` while the API is unstable
- Bump `MAJOR` for breaking changes
- Bump `MINOR` for new backwards-compatible features
- Bump `PATCH` for backwards-compatible fixes

Pre-releases use PEP 440 suffixes on the tag: `v0.2.0rc1`, `v0.2.0a1`, etc.

## If something goes wrong

- **Workflow fails before publish**: fix the issue on `master`, delete the
  tag locally and remotely (`git tag -d v0.1.0 && git push --delete origin
  v0.1.0`), then re-tag.
- **Publish succeeds but Release creation fails**: create the Release
  manually from the tag via the GitHub UI.
- **Bad release on PyPI**: PyPI does not allow re-uploading the same
  version. Yank the release on PyPI and publish a new patch version.
