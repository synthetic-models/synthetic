# Build, Install, and PyPI Upload Guide

This guide covers how to build the Synthetic package, install it locally, and publish it to PyPI.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Building the Package](#building-the-package)
- [Installing Locally](#installing-locally)
- [Testing the Build](#testing-the-build)
- [Publishing to PyPI](#publishing-to-pypi)
- [Common Issues](#common-issues)

## Prerequisites

### Required Tools

| Tool | Purpose | Install Command |
|------|---------|-----------------|
| Python | >= 3.11 | `python --version` |
| uv | Fast Python package installer | `pip install uv` |
| hatchling | Build backend (via pyproject.toml) | Included in build process |
| Twine | PyPI upload tool | `pip install twine` |
| build | PEP 517 build frontend | `pip install build` |

### Install Build Dependencies

```bash
pip install build twine uv
```

## Building the Package

The Synthetic package uses `hatchling` as its build backend.

### Clean Build

To create a clean build of the package:

```bash
python -m build
```

This will create:
- `dist/synthetic-0.1.0.tar.gz` - Source distribution (sdist)
- `dist/synthetic-0.1.0-py3-none-any.whl` - Wheel distribution

### Build with uv

If you're using `uv`, you can build with:

```bash
uv build
```

### Verifying the Build

Check that the artifacts were created:

```bash
ls -la dist/
```

Expected output:
```
synthetic-0.1.0-py3-none-any.whl
synthetic-0.1.0.tar.gz
```

## Installing Locally

### Development Mode (Editable)

Install the package in editable mode for development:

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

### Install from Built Wheel

Install from a pre-built wheel:

```bash
pip install dist/synthetic-0.1.0-py3-none-any.whl
```

### Install from Source Distribution

Install from the source tarball:

```bash
pip install dist/synthetic-0.1.0.tar.gz
```

## Testing the Build

### Run Tests

After building and installing, run the test suite:

```bash
pytest
```

Run a specific test file:

```bash
pytest tests/test_specific_file.py::test_function_name
```

### Verify Installation

Verify the package is installed correctly:

```bash
python -c "import synthetic; print(synthetic.__version__)"
```

### Check Package Contents

Inspect the built wheel to ensure all files are included:

```bash
unzip -l dist/synthetic-0.1.0-py3-none-any.whl
```

## Publishing to PyPI

### 1. Prepare for Publishing

Before uploading, ensure:

1. **Update version number** in `pyproject.toml`:
   ```toml
   [project]
   version = "X.Y.Z"  # Update accordingly
   ```

2. **Update README.md** - Ensure the package description is accurate.

3. **Update description** in `pyproject.toml`:
   ```toml
   description = "Virtual cell generation library for biological synthetic data"
   ```

4. **Review dependencies** in `pyproject.toml` are correctly specified.

### 2. Create PyPI Account

If you don't have one already:
1. Go to [PyPI](https://pypi.org)
2. Create an account
3. Enable 2FA (recommended for security)

### 3. Configure PyPI Credentials

#### Option A: Using .pypirc (Recommended for Development)

Create `~/.pypirc`:

```ini
[distutils]
index-servers = pypi

[pypi]
username = __token__
password = pypi-...  # Your API token from PyPI account settings
```

#### Option B: Using Keyring (More Secure)

```bash
pip install keyring
keyring set https://upload.pypi.org/legacy/ pypi-...  # Your username
```

#### Option C: Environment Variable

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-...  # Your API token
```

### 4. Upload to PyPI

#### Check the Distribution

Before uploading, use Twine to check the distribution:

```bash
twine check dist/*
```

Fix any warnings before proceeding.

#### Upload to PyPI

```bash
twine upload dist/*
```

### 5. Verify Upload

After a successful upload:
1. Go to [PyPI](https://pypi.org) and search for "synthetic"
2. Verify the version and description appear correctly
3. Test installing from PyPI:
   ```bash
   pip install synthetic==0.1.0
   ```

## TestPyPI (Staging)

Before publishing to the main PyPI, you can use TestPyPI to test uploads:

### Create TestPyPI Account

1. Go to [TestPyPI](https://test.pypi.org)
2. Create a separate account (or use the same as PyPI)

### Configure TestPyPI in .pypirc

```ini
[distutils]
index-servers =
    pypi
    testpypi

[testpypi]
repository = https://test.pypi.org/legacy/
username = __token__
password = pypi-...  # Your TestPyPI API token
```

### Upload to TestPyPI

```bash
twine upload --repository testpypi dist/*
```

### Install from TestPyPI

```bash
pip install --index-url https://test.pypi.org/simple/ synthetic==0.1.0
```

## Version Management

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: Backward-compatible functionality additions
- **PATCH**: Backward-compatible bug fixes

### Release Checklist

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md` with release notes
- [ ] Run full test suite: `pytest`
- [ ] Build package: `python -m build`
- [ ] Check build: `twine check dist/*`
- [ ] Upload to TestPyPI first: `twine upload --repository testpypi dist/*`
- [ ] Install from TestPyPI and verify
- [ ] Upload to PyPI: `twine upload dist/*`
- [ ] Create git tag: `git tag vX.Y.Z`
- [ ] Push tag: `git push origin vX.Y.Z`

## Common Issues

### Build Fails: Module Not Found

Ensure the package structure is correct:
```
synthetic/
├── pyproject.toml
├── README.md
├── src/
│   └── synthetic/
│       ├── __init__.py
│       └── ...
└── tests/
```

### Upload Fails: File Already Exists

Increment the version number in `pyproject.toml` and rebuild.

### Twine: Invalid or Nonexistent Authentication

- Verify your PyPI API token is correct
- Ensure you're using `__token__` as the username
- Check that the token has upload permissions

### Long Description Not Rendered

Ensure `README.md` is in the project root and properly formatted. The content will be used as the long description on PyPI.

### Missing Dependencies

Verify all dependencies are listed in `pyproject.toml` under `[project.dependencies]`.

## Additional Resources

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI Homepage](https://pypi.org)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Hatchling Documentation](https://hatch.pypa.io/latest/plugins/hatchling/)
