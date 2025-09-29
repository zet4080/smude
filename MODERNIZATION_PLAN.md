# Smude Project Modernization Plan

**Date:** September 29, 2025  
**Current Python Version:** 3.12.6  
**Project Status:** Legacy (Python 3.8.5, dependencies from 2020)

## Overview

This plan outlines the modernization of the Smude (Sheet Music Dewarping) project to bring it up to current Python standards and best practices. The project currently uses Python 3.8.5 and dependencies from 2020, which are significantly outdated.

## Pre-trained Model Analysis

### Current Model Details
- **Source**: `https://github.com/sonovice/smude/releases/download/v0.1.0/model.ckpt`
- **Framework**: PyTorch Lightning 0.9.0 checkpoint format
- **Architecture**: U-Net (4 output classes, 5 layers, 64 starting features, bilinear upsampling)
- **Size**: ~50MB (estimated)
- **Storage**: `smude/model.ckpt` (downloaded automatically on first run)
- **Usage**: Sheet music dewarping and binarization
 - **Status (Conversion)**: Completed on Sep 29, 2025 → artifacts present:
   - `smude/model_weights.pth` (pure PyTorch state_dict)
   - `smude/model_config.pth` (architecture config)

### Model Compatibility Challenges

#### PyTorch Lightning Version Issues
1. **Checkpoint Format**: Lightning 0.9.0 uses different checkpoint structure than 2.1+
2. **Loading API**: `load_from_checkpoint` method parameters have changed
3. **Model State**: Internal model state dictionary format may be incompatible
4. **Hyperparameters**: Checkpoint may contain deprecated hyperparameter formats

#### Migration Strategy Options

**Option A: Model Weight Extraction (Recommended)**
```python
# Step 1: Load with old PyTorch Lightning
old_model = SegModel.load_from_checkpoint("model.ckpt")  # PL 0.9.0
torch.save(old_model.net.state_dict(), "model_weights.pth")  # Pure PyTorch

# Step 2: Load with new PyTorch Lightning
new_model = UNet(num_classes=4, num_layers=5, features_start=64, bilinear=True)
new_model.load_state_dict(torch.load("model_weights.pth"))
```

**Option B: Checkpoint Migration**
- Use PyTorch Lightning's built-in checkpoint migration tools
- May require manual state dict key mapping

**Option C: Model Retraining**
- Last resort if conversion fails
- Would require original training dataset

## Current State Analysis

### Issues Identified

1. **Outdated Python Version**: Currently requires Python 3.8.5 (released 2020)
2. **Legacy Dependencies**: All dependencies are 4-5 years old
3. **Deprecated PyTorch Lightning Version**: Using v0.9.0 (current is 2.x)
4. **Old PyTorch Version**: Using v1.6.0 (current is 2.x)
5. **Deprecated `typing` module**: Using separate typing package (built into Python 3.9+)
6. **Legacy OpenCV version**: Using older opencv-contrib-python version
7. **Missing Modern Python Features**: No type hints, f-strings, dataclasses, etc.
8. **No Testing Infrastructure**: No unit tests or CI/CD
9. **Legacy Build System**: Using old setup.py instead of pyproject.toml
10. **No Code Quality Tools**: No linting, formatting, or pre-commit hooks

### Dependencies Version Update Plan

| Package | Current Version | Target Version | Notes |
|---------|----------------|----------------|-------|
| Python | 3.8.5 | 3.11+ | Modern Python with better performance |
| PyTorch | 1.6.0 | 2.1+ | Major API changes expected |
| PyTorch Lightning | 0.9.0 | 2.1+ | Significant breaking changes |
| NumPy | 1.19.1 | 1.25+ | Performance improvements |
| scikit-image | 0.17.2 | 0.22+ | New features and bug fixes |
| SciPy | 1.5.2 | 1.11+ | Performance improvements |
| torchvision | 0.7.0 | 0.16+ | Must match PyTorch version |
| OpenCV | 4.4.0.42 | 4.8+ | Security and feature updates |
| typing | 3.7.4.3 | Remove | Built into Python 3.9+ |
| typing_extensions | 3.7.4.2 | 4.8+ | For advanced typing features |

## Modernization Phases

### Phase 1: Infrastructure Modernization
**Priority: High**  
**Estimated Time: 1-2 days**

#### 1.1 Build System Migration
- [ ] Replace `setup.py` with `pyproject.toml`
- [ ] Use modern build backend (setuptools with pyproject.toml)
- [ ] Add proper metadata and classifiers
- [ ] Configure entry points in pyproject.toml

#### 1.2 Development Environment
- [ ] Create `.gitignore` file
- [ ] Add `requirements-dev.txt` for development dependencies
- [ ] Update conda environment.yml with modern versions
- [ ] Add `.python-version` file for pyenv users

#### 1.3 Code Quality Tools
- [ ] Add `black` for code formatting
- [ ] Add `ruff` for linting (replaces flake8, isort, etc.)
- [ ] Add `mypy` for type checking
- [ ] Configure `pre-commit` hooks
- [ ] Add `.editorconfig` for consistent formatting

### Phase 1.5: Model Preservation (Critical)
**Priority: CRITICAL**  
**Estimated Time: 1 day**

#### 1.5.1 Model Backup and Analysis
- [x] **Download current model** if not present: `model.ckpt` (auto via extractor)
- [ ] **Create backup copy** of original checkpoint (optional)
- [ ] **Test current model loading** with existing PyTorch Lightning 0.9.0 (optional)
- [ ] **Document model performance** on sample images (baseline metrics)
- [x] **Extract model metadata** (architecture config saved to `smude/model_config.pth`)

#### 1.5.2 Model Weight Extraction
- [x] **Create extraction script** (`scripts/extract_model_weights.py`) [lightweight, no Lightning]
  - Reads checkpoint with `torch.load` (safe fallback included) and saves:
    - `smude/model_weights.pth`
    - `smude/model_config.pth`
- [ ] **Verify extracted weights** load correctly in pure PyTorch (basic smoke pending)
- [ ] **Test inference equivalence** between original and extracted model (pending)

#### 1.5.3 Model Loading Compatibility
- [x] **Create dual-loader utility** that handles both old and new formats (`smude/loader.py`)
- [x] **Implement fallback mechanism** for model loading (lazy legacy import)
- [x] **Add model format detection** (auto-discovery of `model_weights.pth` vs `model.ckpt`)
- [ ] **Create model validation function** to ensure correct loading (pending)

### Phase 2: Dependency Updates
**Priority: High**  
**Estimated Time: 3-4 days** *(Increased due to model compatibility)*

#### 2.1 Model Compatibility (Critical First Step)
- [ ] **Test extracted model weights** with current PyTorch version (smoke test)
- [x] **Create modern model loading function** (`smude/loader.py: load_model_flexible`):
  ```python
  from smude.loader import load_model_flexible
  model = load_model_flexible()  # auto-prefers weights; falls back to legacy ckpt if needed
  model.eval()
  ```
- [ ] **Implement model wrapper class** for modern PyTorch Lightning (post-upgrade, optional)
- [ ] **Verify inference equivalence** between old and new loading methods (pending)

#### 2.2 Python Version Update
- [ ] Update minimum Python requirement to 3.11
- [ ] Remove deprecated `typing` package dependency
- [ ] Update type hints to use built-in types (list, dict, etc.)

#### 2.3 Core Dependencies Update
- [ ] **CRITICAL**: Update PyTorch to 2.1+ after model extraction
- [ ] **CRITICAL**: Update PyTorch Lightning to 2.1+ with new model loading
- [ ] Update torchvision to match PyTorch version
- [ ] Update NumPy, SciPy, scikit-image to latest stable versions
- [ ] Update OpenCV to latest version
- [ ] **Verify model still works** after each major update

#### 2.4 Breaking Changes Handling
- [ ] **Refactor model loading mechanism** for PyTorch Lightning 2.x
- [ ] **Update model.py** to use modern PyTorch Lightning LightningModule API
- [ ] **Test model inference pipeline** end-to-end
- [ ] Update PyTorch tensor operations for v2.x compatibility
- [ ] Fix any deprecated function calls
- [ ] Update import statements for reorganized modules

### Phase 3: Code Modernization
**Priority: Medium**  
**Estimated Time: 3-4 days**

#### 3.1 Type Hints
- [ ] Add comprehensive type hints to all functions
- [ ] Use `pathlib.Path` instead of string paths
- [ ] Add type hints for NumPy arrays and PyTorch tensors
- [ ] Use `typing.Protocol` for interface definitions

#### 3.2 Modern Python Features
- [ ] Replace string formatting with f-strings
- [ ] Use `dataclasses` for configuration objects
- [ ] Implement context managers where appropriate
- [ ] Use `pathlib` for file operations
- [ ] Replace manual argument parsing with `click` or `typer`

#### 3.3 Code Structure Improvements
- [ ] Implement proper logging configuration
- [ ] Add configuration management (YAML/TOML)
- [ ] Separate CLI from library code
- [ ] Add proper error handling and custom exceptions
- [ ] Implement progress bars with `rich`

### Phase 4: Testing and Documentation
**Priority: Medium**  
**Estimated Time: 2-3 days**

#### 4.1 Testing Infrastructure
- [ ] Add `pytest` for unit testing
- [ ] Create test fixtures for sample images
- [ ] **Add model compatibility tests** (old vs new format)
- [ ] **Add model inference tests** with known inputs/outputs
- [ ] Add unit tests for each module
- [ ] Add integration tests
- [ ] Configure test coverage reporting
- [ ] Add GitHub Actions for CI/CD

#### 4.2 Documentation Updates
- [ ] Update README.md with modern installation instructions
- [ ] Add API documentation with `sphinx`
- [ ] Create developer documentation
- [ ] Add examples and tutorials
- [ ] Document breaking changes and migration guide

### Phase 5: Performance and Features
**Priority: Low**  
**Estimated Time: 2-3 days**

#### 5.1 Performance Improvements
- [ ] Profile code for bottlenecks
- [ ] Optimize NumPy operations
- [ ] Use PyTorch 2.x compilation features
- [ ] Implement batch processing
- [ ] Add GPU memory optimization

#### 5.2 Modern Features
- [ ] Add progress bars for long operations
- [ ] Implement async/await for I/O operations
- [ ] Add support for different image formats
- [ ] Implement model caching
- [ ] Add configuration validation

## Breaking Changes and Migration

### For Users
- Minimum Python version increased from 3.8 to 3.11
- Some CLI arguments may change
- Configuration file format may change
- Import paths may change

### For Developers
- PyTorch Lightning API has changed significantly
- Model loading/saving methods updated
- Type hints added (may require mypy compliance)
- Testing framework changed to pytest

## Risk Assessment

### Critical Risk
- **Pre-trained Model Compatibility**: Mitigated — converted to pure PyTorch weights (`model_weights.pth`)
- **Model Download Dependency**: Mitigated — extractor supports auto-download with checksum via HTTP

### High Risk
- PyTorch Lightning 0.9 → 2.1 migration (breaking changes)
- PyTorch 1.6 → 2.1 migration (some API changes)
- Model state dict format changes between PyTorch versions

### Medium Risk
- scikit-image API changes
- OpenCV API changes
- Python 3.11 compatibility issues

### Low Risk
- NumPy/SciPy updates (mostly backward compatible)
- Development tool updates
- Documentation updates

## Recommended Execution Order

1. **Start with Phase 1** (Infrastructure) to establish modern development practices
2. Phase 1.5 (Model Preservation) — Completed on Sep 29, 2025
3. **Proceed to Phase 2** (Dependencies) with careful testing at each step
4. **Implement Phase 3** (Code Modernization) incrementally
5. **Add Phase 4** (Testing/Documentation) throughout the process
6. **Optimize with Phase 5** (Performance) as final step

### Model Migration Workflow
```
Current State → Model Extraction → Dependency Update → Modern Loading → Verification
     ↓               ↓                    ↓                ↓              ↓
model.ckpt → model_weights.pth → PyTorch 2.1+ → New Loader → Same Results
(PL 0.9)     (Pure PyTorch)      (PL 2.1+)      (Modern)     (Validated)
```

## Success Metrics

- [ ] All tests pass with new dependencies
- [ ] Code quality score > 8.0 with ruff
- [ ] 100% type hint coverage
- [ ] Test coverage > 80%
- [ ] Documentation completeness > 90%
- [ ] Performance maintained or improved
- [ ] Zero security vulnerabilities in dependencies

## Timeline

**Total Estimated Time: 12-17 days** *(Increased due to model compatibility requirements)*

- Phase 1: 1-2 days
- Phase 1.5 (Model): 1 day *(CRITICAL)*
- Phase 2: 3-4 days  
- Phase 3: 3-4 days
- Phase 4: 2-3 days
- Phase 5: 2-3 days

## Next Steps

1. Create a backup branch of the current code
2. Set up modern development environment (Phase 1)
3. Run a quick smoke test with the flexible loader and a dummy input
4. Proceed with dependency updates (Phase 2), validating model inference after each major change
5. Add unit/integration tests and CI (Phase 4)
6. Update documentation as changes are made

### Model Migration Priority Actions

1. **Completed**: Extraction script run; artifacts present (`smude/model_weights.pth`, `smude/model_config.pth`)
2. **Optional**: Backup original checkpoint (`Copy-Item smude\model.ckpt smude\model_original.ckpt` on PowerShell)
3. **Next**: Run smoke test and then begin dependency upgrades
4. **Then**: Validate inference equivalence once the modern stack is in place

---

**CRITICAL WARNING**: Do not update PyTorch or PyTorch Lightning dependencies until the model weights have been successfully extracted and verified. The original model checkpoint will likely become unusable after the dependency updates.

**Note**: This is a comprehensive plan. The model compatibility section is now the highest priority due to the significant breaking changes in PyTorch Lightning between versions 0.9 and 2.1+. Some items may be optional depending on project requirements and time constraints. The plan should be executed incrementally with frequent testing to ensure stability.