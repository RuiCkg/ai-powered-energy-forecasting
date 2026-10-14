"""Build the invalid upload files and check each one against the live rules.

Run: python scripts/make_invalid_uploads.py OUTPUT_DIRECTORY

Files are built from the same ``SECURITY_TEST_MATRIX`` the test suite uses, so a
demonstration uploads exactly what the suite covers. Each file is then passed
through ``validate_upload`` and the rejection it actually produced is printed
beside the one expected, which is the evidence table for NFR1 without needing
the real source archives.

Three scenarios cannot exist as files on disk (a name with a path separator, a
name containing a control character, and an empty name). A file picker could not
submit them either, so they stay covered by the tests and are listed as such.

To see a rejection in the dashboard instead, the AEMO input needs the full set:
eleven real monthly archives plus one of these files in place of the twelfth.
Uploading a single file on its own is refused for the count first, which is the
count rule working, not the file rule failing.

Nothing here is a working source archive. Do not leave these in a data
directory, and do not commit them.
"""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(REPOSITORY_ROOT / "tests"))

from energy_forecasting.upload_validation import UploadRejected, validate_upload  # noqa: E402
from test_upload_validation import SECURITY_TEST_MATRIX  # noqa: E402


def submittable(name: str) -> bool:
    """True when a file picker could offer a file under this name."""

    return bool(name.strip()) and "/" not in name and "\\" not in name and "\x00" not in name


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/make_invalid_uploads.py OUTPUT_DIRECTORY")
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)

    rows = []
    test_only = []
    for identifier, tier, description, name, builder, policy, expected in SECURITY_TEST_MATRIX:
        payload = builder()
        if not submittable(name):
            test_only.append((identifier, tier, description, expected))
            continue
        # Written under its scenario ID so the files are distinguishable. The
        # prefix cannot change which rule fires: it is safe characters only and
        # it leaves the extension alone.
        path = target / f"{identifier}_{name}"
        path.write_bytes(payload)
        try:
            validate_upload(f"{identifier}_{name}", payload, policy)
            actual = "ACCEPTED"
        except UploadRejected as rejection:
            actual = rejection.code
        rows.append((identifier, tier, path.name, expected, actual, path.stat().st_size))

    print(f"Wrote and checked {len(rows)} invalid upload files in {target}\n")
    print(f"{'ID':<5}{'TIER':<7}{'EXPECTED':<28}{'ACTUAL':<28}{'RESULT':<8}{'BYTES':>12}")
    for identifier, tier, _filename, expected, actual, size in rows:
        result = "match" if expected == actual else "MISMATCH"
        print(f"{identifier:<5}{tier:<7}{expected:<28}{actual:<28}{result:<8}{size:>12,}")

    if test_only:
        print("\nCovered by the test suite only (a file picker cannot submit these names):")
        for identifier, tier, description, expected in test_only:
            print(f"  {identifier} [{tier}] {description} -> {expected}")

    mismatches = [row[0] for row in rows if row[3] != row[4]]
    core_checked = [row for row in rows if row[1] == "core"]
    print(f"\n{len(rows)} files checked, {len(core_checked)} of them core scenarios; "
          f"{len(mismatches)} mismatch(es).")
    if mismatches:
        print(f"Mismatched scenarios: {', '.join(mismatches)}")
        raise SystemExit(1)
    print("Every generated file was refused by the rule it was built to trip.")
