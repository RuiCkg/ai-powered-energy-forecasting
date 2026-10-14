# Security Implementation - Quick Reference Guide

**For:** Developers integrating or maintaining the energy forecasting platform  
**Audience:** Technical team members  
**Updated:** 6 October 2026

---

## TL;DR

- **Security Framework:** `src/energy_forecasting/upload_validation.py`
- **Entry Point:** `validate_documented_sources(aemo_files, ausgrid_file)`
- **Returns:** Validation evidence (sizes, hashes, policy) or raises `UploadRejected`
- **Test Coverage:** 48 tests, all passing
- **Status:** Production ready

---

## Key Classes and Functions

### 1. `RejectionCode` (Enum)

13 stable rejection codes (never change values, only add new ones):

```python
from energy_forecasting.upload_validation import RejectionCode

# Example codes
RejectionCode.INVALID_FORMAT  # Not a ZIP
RejectionCode.COMPRESSION_BOMB  # 200:1 expansion ratio exceeded
RejectionCode.MEMBER_PATH_UNSAFE  # Path traversal detected
RejectionCode.MEMBER_SUFFIX_DENIED  # .exe, .dll, .so, .pkl, .joblib
```

### 2. `validate_upload()` Function

Core validation before any parser runs:

```python
from energy_forecasting.upload_validation import validate_upload

try:
    evidence = validate_upload(
        upload_bytes=file.getvalue(),
        organization="aemo",  # or "ausgrid"
        filename=file.name
    )
    # evidence = {
    #     'organization': 'aemo',
    #     'filename': 'aemo_demand_2025_08.zip',
    #     'archive_bytes': 12345,
    #     'member_count': 45,
    #     'archive_digest': 'sha256_hex_string',
    #     'policy': UploadPolicy(...)
    # }
except UploadRejected as rejection:
    # rejection.code: specific RejectionCode
    # rejection.message: safe, generic message (no technical details)
    print(f"Upload rejected: {rejection.message}")
```

### 3. `validate_documented_sources()` Function

Batch validation with session budget enforcement:

```python
from energy_forecasting.upload_validation import validate_documented_sources

try:
    evidence = validate_documented_sources(aemo_files, ausgrid_file)
    # evidence = [
    #     {'organization': 'aemo', 'filename': ..., ...},
    #     {'organization': 'aemo', 'filename': ..., ...},
    #     ...
    #     {'organization': 'ausgrid', 'filename': ..., ...},
    # ]
except UploadRejected as rejection:
    st.error(f"Upload validation failed: {rejection.message}")
```

### 4. `UploadPolicy` Dataclass

Organization-specific constraints:

```python
from energy_forecasting.upload_validation import UPLOAD_POLICIES

# AEMO policy
aemo_policy = UPLOAD_POLICIES['aemo']
# max_files: 12
# max_archive_members: 256
# max_archive_size_bytes: 100 MB

# Ausgrid policy
ausgrid_policy = UPLOAD_POLICIES['ausgrid']
# max_files: 1
# max_archive_members: 64
# max_archive_size_bytes: 200 MB
```

---

## Integration in Streamlit App

The app (`streamlit_app.py`) follows this pattern:

```python
from energy_forecasting.upload_validation import (
    validate_documented_sources,
    UploadRejected,
)

# Get files from user
aemo_files = st.file_uploader("Upload AEMO files", type="zip", accept_multiple_files=True)
ausgrid_file = st.file_uploader("Upload Ausgrid file", type="zip")

# Validate before parsing
try:
    evidence = validate_documented_sources(aemo_files, ausgrid_file)
    # All files passed validation
    # Now safe to parse and process
    case_data = prepare_uploaded_sources(aemo_files, ausgrid_file)
except UploadRejected as rejection:
    # One or more files rejected
    st.error(f"Upload validation failed: {rejection.message}")
    # State is cleared, user can retry
```

---

## 14 Attack Scenarios Blocked

| ID | Scenario | Rejection Code |
| --- | --- | --- |
| S1 | File extension not .zip | `INVALID_FORMAT` |
| S2 | ZIP name, non-ZIP content | `SIGNATURE_MISMATCH` |
| S3 | Larger than size limit | `FILE_TOO_LARGE` |
| S4 | Zero-byte upload | `EMPTY_FILE` |
| S5 | Compression bomb (200:1+) | `COMPRESSION_RATIO_EXCEEDED` |
| S6 | Member escaping archive (..) | `MEMBER_PATH_UNSAFE` |
| S7 | Serialized object (.exe, .dll, etc.) | `MEMBER_SUFFIX_DENIED` |
| S8 | Unsafe filename with / or \ | `UNSAFE_NAME` |
| S9 | Corrupted/truncated archive | `ARCHIVE_UNREADABLE` |
| S10 | Encrypted archive member | `ARCHIVE_ENCRYPTED` |
| S11 | Nesting > 5 levels deep | `NESTING_TOO_DEEP` |
| S12 | Member type not allowed | `MEMBER_SUFFIX_NOT_ALLOWED` |
| S13 | Symlink or directory member | `MEMBER_NOT_REGULAR_FILE` |
| S14 | Duplicate member name | `DUPLICATE_MEMBER_NAME` |

---

## Error Message Safety

**All rejection messages are safe and reveal NO:**
- ✗ Path information
- ✗ Module names
- ✗ Exception text
- ✗ Submitted filename
- ✗ Technical implementation details

Example safe message:
```
"Upload file failed validation"
```

**Never leaks:**
```
# BAD - reveals path traversal check
"Invalid path ../../../etc/passwd"

# BAD - reveals member suffix check
"File .exe is denied"

# BAD - reveals compression ratio calculation
"Expansion ratio 250:1 exceeds limit"
```

---

## Testing the Security Framework

### Run All Security Tests

```bash
python -m pytest tests/test_upload_validation.py \
                  tests/test_experiment_record.py \
                  tests/test_shap_additivity.py \
                  tests/test_app_wiring.py -v
```

Expected: **48 passed, 48 subtests passed**

### Test Individual Scenarios

```bash
# Test rejection matrix (14 attack scenarios)
python -m pytest tests/test_upload_validation.py::RejectionMatrixTest -v

# Test valid uploads pass
python -m pytest tests/test_upload_validation.py::AcceptedUploadTest -v

# Test batch validation
python -m pytest tests/test_upload_validation.py::UploadSetTest -v

# Test message safety
python -m pytest tests/test_upload_validation.py::SafeNameTest -v
```

### Generate Test Archives

```bash
# Create synthetic AEMO/Ausgrid archives
python scripts/make_demo_sources.py ./test_sources

# Create attack scenario files
python scripts/make_invalid_uploads.py ./invalid_archives
```

---

## Reproducibility and Experiment Records

### Exact Metric Reproducibility

The framework enforces **exact reproducibility**, not approximate:

```python
from energy_forecasting.experiment_record import build_experiment_record

# Same inputs → same digests
record1 = build_experiment_record(case_data, results_dict)
record2 = build_experiment_record(case_data, results_dict)

assert record1['configuration_digest'] == record2['configuration_digest']
assert record1['metric_digest'] == record2['metric_digest']

# Different seed → different digests (failure detected)
record3 = build_experiment_record(case_data_with_new_seed, results_dict)
assert record3['metric_digest'] != record1['metric_digest']  # Caught!
```

### Float32 Tolerance for SHAP

SHAP contributions use scaled tolerance with absolute floor:

```python
# Tolerance = max(relative_error * |prediction|, absolute_minimum)
# Relative: 1e-5 of prediction magnitude
# Absolute: 1e-10 (prevents division by near-zero)
# Worst row: maximum error across all rows
```

---

## Documentation References

- **Complete Specification:** `docs/security_validation_spec.md`
- **Deployment Checklist:** `docs/deployment_verification_checklist.md`
- **Data Manifest:** `docs/data_manifest.md` (SHA-256 hashes)
- **Sineth's Summary:** `docs/sineth_security_implementation_summary.md`

---

## Common Issues and Solutions

### Issue: `UploadRejected` exception in production

**Solution:** This is the intended behavior. The app catches the exception and displays the safe message to the user. Check the `rejection.code` to log the specific scenario.

```python
try:
    evidence = validate_documented_sources(aemo_files, ausgrid_file)
except UploadRejected as rejection:
    # Log the specific code for monitoring
    logger.warning(f"Upload rejected: {rejection.code.name}")
    # Display safe message to user
    st.error(rejection.message)
```

### Issue: "Upload file failed validation" but I don't know why

**Solution:** This is correct design — the message is intentionally generic for security. To debug:

1. Check test cases in `tests/test_upload_validation.py`
2. Run `scripts/make_invalid_uploads.py` to test scenarios
3. Examine `rejection.code` to determine the reason

### Issue: My valid file is rejected

**Possible causes:**
- Archive exceeds size limit (10 MB for AEMO, 200 MB for Ausgrid)
- Member count exceeds policy (256 for AEMO, 64 for Ausgrid)
- File contains denied suffixes (.exe, .dll, .so, .pkl, .joblib)
- Nesting depth > 5 levels
- Compression ratio > 200:1

**Solution:** Use `make_demo_sources.py` to generate correctly shaped archives for testing.

---

## Performance Characteristics

- **Validation latency:** ~50-200ms per archive (depends on size)
- **Memory:** Streaming ZIP reader (does not load entire archive into RAM)
- **Session budget:** 192 MB total across all uploads

---

## For Code Reviewers

Key points to verify:

1. ✓ All rejection codes are specific and immutable (enum)
2. ✓ No technical details leak in error messages
3. ✓ All 48 tests passing
4. ✓ No per-row data in experiment records
5. ✓ Exact reproducibility enforced (not approximate)
6. ✓ Policy enforcement before parser runs
7. ✓ Archive corruption detected safely
8. ✓ Path traversal blocked (.., absolute paths, symlinks)
9. ✓ Compression bomb detected (200:1 ratio)
10. ✓ Denied extensions enforced (.exe, .dll, .so, .pkl, .joblib)

---

## Contributing

When adding new validation rules:

1. Add new `RejectionCode` to the enum
2. Add safe message to `REJECTION_MESSAGES`
3. Add validation logic to `validate_upload()`
4. Add test case to `tests/test_upload_validation.py`
5. Update the rejection matrix in `security_validation_spec.md`
6. Ensure message reveals no technical details
7. Run full test suite: `pytest tests/ -v`

---

## Questions?

- **Security concerns:** Contact Sineth (Risk R6 owner)
- **Implementation details:** See `docs/security_validation_spec.md`
- **Test evidence:** See `docs/deployment_verification_checklist.md`
- **Code:** `src/energy_forecasting/upload_validation.py` (well-documented docstrings)
