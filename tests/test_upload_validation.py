"""The upload rejection matrix, run as tests rather than described in a table.

``SECURITY_TEST_MATRIX`` is the evidence for NFR1. The eight scenarios marked
``core`` are the eight invalid files the requirement names; the rest cover
archive-structure rules that the eight do not reach. Each case builds a real
file, submits it through the same entry point the app uses, and asserts the
specific rejection code, so a rule that stops working fails a test instead of
passing a review.

The rejection-message test is the second half of NFR1: the response must not
leak internals. It checks every case's message for path separators, exception
text, module names and the submitted name itself.
"""

from pathlib import Path
import io
import stat
import sys
import unittest
import warnings
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from energy_forecasting.upload_validation import (
    AEMO_MONTHLY_POLICY,
    AUSGRID_ARCHIVE_POLICY,
    DENIED_MEMBER_SUFFIXES,
    REJECTION_CODES,
    UploadPolicy,
    UploadRejected,
    safe_display_name,
    validate_documented_sources,
    validate_upload,
    validate_upload_set,
)

VALID_AEMO_NAME = "PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_20250801.zip"
DOCUMENTED_AEMO_NAMES = [
    f"PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_{year}{month:02d}01.zip"
    for year, months in ((2025, range(8, 13)), (2026, range(1, 8)))
    for month in months
]


class Upload:
    """Stand-in for a Streamlit ``UploadedFile``: a name and some bytes."""

    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def zip_bytes(members, compression=zipfile.ZIP_DEFLATED, external_attr_by_name=None) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression) as archive:
        for name, payload in members:
            info = zipfile.ZipInfo(name)
            info.compress_type = compression
            attribute = (external_attr_by_name or {}).get(name)
            info.external_attr = attribute if attribute is not None else (stat.S_IFREG | 0o644) << 16
            archive.writestr(info, payload)
    return buffer.getvalue()


def aemo_shaped_archive(daily_days: int = 2) -> bytes:
    """An archive with the documented AEMO shape: monthly ZIP of daily ZIPs of CSV."""

    daily = [
        (
            f"PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_202508{day:02d}.zip",
            zip_bytes([(f"PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_202508{day:02d}.CSV", b"I,OPERATIONAL_DEMAND\n")]),
        )
        for day in range(1, daily_days + 1)
    ]
    return zip_bytes(daily, compression=zipfile.ZIP_STORED)


def ausgrid_shaped_archive() -> bytes:
    return zip_bytes([("Solar home 2012-2013.csv", b"notice\nCustomer,Generator Capacity\n")])


def encrypted_archive() -> bytes:
    """Mark a member encrypted by setting general-purpose bit 0 in both headers.

    ``zipfile`` cannot write encrypted entries, so the flag is patched into a
    normal archive. The reader still reports the bit, which is what the rule
    inspects.
    """

    raw = bytearray(zip_bytes([("data.csv", b"a,b\n1,2\n")]))
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        index = raw.find(signature)
        if index < 0:
            raise RuntimeError("Could not locate ZIP header to mark as encrypted")
        raw[index + flag_offset] |= 0x01
    return bytes(raw)


# A budget small enough to be exceeded by a stored, incompressible member, so
# the expansion rule can be tested apart from the ratio rule.
SMALL_EXPANSION_POLICY = UploadPolicy(
    name="small_expansion_budget",
    max_bytes=10 * 1024 * 1024,
    allowed_member_suffixes=frozenset({".csv"}),
    max_expanded_bytes=1024 * 1024,
)


def duplicate_member_archive() -> bytes:
    """Two members with one name. ``zipfile`` warns while writing it; that is expected."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return zip_bytes([("data.csv", b"first\n"), ("data.csv", b"second\n")])


def oversize_archive(policy: UploadPolicy) -> bytes:
    """A real ZIP larger than the policy limit, stored so it cannot compress."""

    filler = bytes(range(256)) * ((policy.max_bytes // 256) + 4096)
    return zip_bytes([("data.csv", filler)], compression=zipfile.ZIP_STORED)


# (identifier, tier, description, name, byte builder, policy, expected code)
SECURITY_TEST_MATRIX = (
    ("S1", "core", "Extension not on the allow-list", "monthly_demand.csv",
     lambda: aemo_shaped_archive(), AEMO_MONTHLY_POLICY, "SUFFIX_NOT_ALLOWED"),
    ("S2", "core", "ZIP name, non-ZIP content", VALID_AEMO_NAME,
     lambda: b"REGIONID,INTERVAL_DATETIME\nNSW1,2025/08/01 00:30:00\n", AEMO_MONTHLY_POLICY, "SIGNATURE_MISMATCH"),
    ("S3", "core", "Larger than the size limit for the input", VALID_AEMO_NAME,
     lambda: oversize_archive(AEMO_MONTHLY_POLICY), AEMO_MONTHLY_POLICY, "FILE_TOO_LARGE"),
    ("S4", "core", "Zero-byte upload", VALID_AEMO_NAME,
     lambda: b"", AEMO_MONTHLY_POLICY, "EMPTY_FILE"),
    ("S5", "core", "Expansion ratio of a compression bomb", VALID_AEMO_NAME,
     lambda: zip_bytes([("bomb.csv", b"0" * (2 * 1024 * 1024))]), AEMO_MONTHLY_POLICY, "COMPRESSION_RATIO_EXCEEDED"),
    ("S6", "core", "Member name escaping the archive directory", VALID_AEMO_NAME,
     lambda: zip_bytes([("../../etc/passwd.csv", b"root\n")]), AEMO_MONTHLY_POLICY, "MEMBER_PATH_UNSAFE"),
    ("S7", "core", "Serialised-object member that would execute on load", VALID_AEMO_NAME,
     lambda: zip_bytes([("demand.csv", b"a,b\n"), ("trained_model.pkl", b"\x80\x04\x95payload")]),
     AEMO_MONTHLY_POLICY, "MEMBER_SUFFIX_DENIED"),
    ("S8", "core", "Submitted name carrying a path separator", "../../PUBLIC_DEMAND.zip",
     lambda: aemo_shaped_archive(), AEMO_MONTHLY_POLICY, "UNSAFE_NAME"),

    ("S9", "extra", "Truncated or corrupt archive", VALID_AEMO_NAME,
     lambda: aemo_shaped_archive()[:120], AEMO_MONTHLY_POLICY, "ARCHIVE_UNREADABLE"),
    ("S10", "extra", "Encrypted archive member", VALID_AEMO_NAME,
     encrypted_archive, AEMO_MONTHLY_POLICY, "ARCHIVE_ENCRYPTED"),
    ("S11", "extra", "Nesting deeper than the documented source layout", VALID_AEMO_NAME,
     lambda: zip_bytes([("a.zip", zip_bytes([("b.zip", zip_bytes([("c.csv", b"a\n")]))]))],
                       compression=zipfile.ZIP_STORED), AEMO_MONTHLY_POLICY, "NESTING_TOO_DEEP"),
    ("S12", "extra", "Member type outside the source-data allow-list", VALID_AEMO_NAME,
     lambda: zip_bytes([("readme.rtf", b"text")]), AEMO_MONTHLY_POLICY, "MEMBER_SUFFIX_NOT_ALLOWED"),
    ("S13", "extra", "Symbolic-link member", VALID_AEMO_NAME,
     lambda: zip_bytes([("link.csv", b"/etc/shadow")],
                       external_attr_by_name={"link.csv": (stat.S_IFLNK | 0o777) << 16}),
     AEMO_MONTHLY_POLICY, "MEMBER_NOT_REGULAR_FILE"),
    ("S14", "extra", "Repeated member names", VALID_AEMO_NAME,
     lambda: duplicate_member_archive(), AEMO_MONTHLY_POLICY, "DUPLICATE_MEMBER_NAME"),
    ("S15", "extra", "More members than the limit", VALID_AEMO_NAME,
     lambda: zip_bytes([(f"part{index:04d}.csv", b"a\n") for index in range(AEMO_MONTHLY_POLICY.max_members + 1)]),
     AEMO_MONTHLY_POLICY, "TOO_MANY_MEMBERS"),
    ("S16", "extra", "Archive with no readable member", VALID_AEMO_NAME,
     lambda: zip_bytes([]), AEMO_MONTHLY_POLICY, "EMPTY_FILE"),
    ("S17", "extra", "Control character in the submitted name", "demand\x00.zip",
     lambda: aemo_shaped_archive(), AEMO_MONTHLY_POLICY, "UNSAFE_NAME"),
    ("S18", "extra", "Absolute member path", VALID_AEMO_NAME,
     lambda: zip_bytes([("/etc/hosts.csv", b"127.0.0.1\n")]), AEMO_MONTHLY_POLICY, "MEMBER_PATH_UNSAFE"),
    ("S19", "extra", "Upload submitted with no name", "",
     lambda: aemo_shaped_archive(), AEMO_MONTHLY_POLICY, "EMPTY_NAME"),
    ("S20", "extra", "Expanded size over the budget at an accepted ratio", VALID_AEMO_NAME,
     lambda: zip_bytes([("data.csv", bytes(range(256)) * 8192)], compression=zipfile.ZIP_STORED),
     SMALL_EXPANSION_POLICY, "EXPANSION_BUDGET_EXCEEDED"),
)


class RejectionMatrixTest(unittest.TestCase):
    def test_every_scenario_is_rejected_with_its_code(self):
        for identifier, _tier, description, name, builder, policy, expected in SECURITY_TEST_MATRIX:
            with self.subTest(scenario=identifier, description=description):
                with self.assertRaises(UploadRejected) as caught:
                    validate_upload(name, builder(), policy)
                self.assertEqual(caught.exception.code, expected)
                self.assertIn(caught.exception.code, REJECTION_CODES)

    def test_eight_core_invalid_files_are_denied(self):
        """NFR1's first half: the eight named invalid files are all refused."""

        core = [row for row in SECURITY_TEST_MATRIX if row[1] == "core"]
        self.assertEqual(len(core), 8)
        denied = 0
        for identifier, _tier, _description, name, builder, policy, _expected in core:
            with self.subTest(scenario=identifier):
                with self.assertRaises(UploadRejected):
                    validate_upload(name, builder(), policy)
                denied += 1
        self.assertEqual(denied, 8)

    def test_rejection_messages_leak_nothing(self):
        """NFR1's second half: the response body carries no internals.

        A rejection may not echo the submitted name, a member name, a path, a
        module name or a caught exception's text.
        """

        forbidden = ("/", "\\", "Traceback", "Errno", "zipfile", "energy_forecasting",
                     "src", ".py", "0x", "passwd", "shadow", "pkl", "Solar home")
        for identifier, _tier, _description, name, builder, policy, _expected in SECURITY_TEST_MATRIX:
            with self.subTest(scenario=identifier):
                with self.assertRaises(UploadRejected) as caught:
                    validate_upload(name, builder(), policy)
                message = caught.exception.message
                self.assertEqual(message, str(caught.exception))
                for fragment in forbidden:
                    self.assertNotIn(fragment, message, f"{identifier} message leaked {fragment!r}")
                stem = name.removesuffix(".zip").strip()
                if stem:
                    self.assertNotIn(stem, message)
                self.assertTrue(message.endswith("."))

    def test_denied_suffix_list_covers_the_serialisation_formats(self):
        for suffix in (".pkl", ".pickle", ".joblib", ".pt", ".pth", ".h5", ".exe", ".dll", ".sh", ".ps1"):
            self.assertIn(suffix, DENIED_MEMBER_SUFFIXES)

    def test_denied_suffix_beats_an_allowed_suffix_list(self):
        """A permissive member allow-list cannot re-admit a denied format."""

        permissive = UploadPolicy(name="permissive", max_bytes=1 << 20,
                                  allowed_member_suffixes=frozenset({".csv", ".pkl"}))
        with self.assertRaises(UploadRejected) as caught:
            validate_upload("source.zip", zip_bytes([("model.pkl", b"\x80\x04")]), permissive)
        self.assertEqual(caught.exception.code, "MEMBER_SUFFIX_DENIED")


class AcceptedUploadTest(unittest.TestCase):
    def test_documented_aemo_shape_is_accepted(self):
        descriptor = validate_upload(VALID_AEMO_NAME, aemo_shaped_archive(), AEMO_MONTHLY_POLICY)
        self.assertEqual(descriptor.policy_name, "aemo_monthly_zip")
        self.assertEqual(descriptor.safe_name, VALID_AEMO_NAME)
        self.assertEqual(descriptor.member_count, 4)  # two daily ZIPs, each holding one CSV
        self.assertEqual(descriptor.deepest_nesting, 2)
        self.assertEqual(len(descriptor.sha256), 64)
        self.assertGreater(descriptor.expanded_bytes, 0)

    def test_documented_ausgrid_shape_is_accepted(self):
        descriptor = validate_upload("Solar_home_half_hour_data.zip", ausgrid_shaped_archive(), AUSGRID_ARCHIVE_POLICY)
        self.assertEqual(descriptor.policy_name, "ausgrid_solar_home_zip")
        self.assertEqual(descriptor.member_suffixes, (".csv",))
        self.assertEqual(descriptor.deepest_nesting, 1)

    def test_ausgrid_limit_is_separate_from_the_aemo_limit(self):
        """The Part B2 10 MB figure cannot be applied to both inputs at once."""

        self.assertEqual(AEMO_MONTHLY_POLICY.max_bytes, 10 * 1024 * 1024)
        self.assertGreater(AUSGRID_ARCHIVE_POLICY.max_bytes, AEMO_MONTHLY_POLICY.max_bytes)
        self.assertTrue(AEMO_MONTHLY_POLICY.provisional)
        self.assertTrue(AUSGRID_ARCHIVE_POLICY.provisional_note)

    def test_descriptor_carries_no_dataset_values(self):
        """NFR4: accepted-upload evidence is sizes, counts and a digest only."""

        described = validate_upload(VALID_AEMO_NAME, aemo_shaped_archive(), AEMO_MONTHLY_POLICY).describe()
        self.assertEqual(
            set(described),
            {"policy_name", "safe_name", "byte_size", "sha256", "member_count",
             "expanded_bytes", "max_member_expansion_ratio", "deepest_nesting", "member_suffixes"},
        )


class UploadSetTest(unittest.TestCase):
    def test_twelve_documented_archives_are_accepted_and_thirteen_are_not(self):
        files = [Upload(name, aemo_shaped_archive()) for name in DOCUMENTED_AEMO_NAMES]
        self.assertEqual(len(validate_upload_set(files, AEMO_MONTHLY_POLICY)), 12)
        with self.assertRaises(UploadRejected) as caught:
            validate_upload_set(files + [Upload(VALID_AEMO_NAME, aemo_shaped_archive())], AEMO_MONTHLY_POLICY)
        self.assertEqual(caught.exception.code, "TOO_MANY_FILES")

    def test_session_budget_stops_a_set_of_individually_acceptable_files(self):
        files = [Upload(name, aemo_shaped_archive()) for name in DOCUMENTED_AEMO_NAMES]
        with self.assertRaises(UploadRejected) as caught:
            validate_upload_set(files, AEMO_MONTHLY_POLICY, session_budget=100)
        self.assertEqual(caught.exception.code, "SESSION_BUDGET_EXCEEDED")

    def test_one_bad_file_rejects_the_whole_set(self):
        files = [Upload(name, aemo_shaped_archive()) for name in DOCUMENTED_AEMO_NAMES]
        files[1] = Upload(DOCUMENTED_AEMO_NAMES[1], b"not a zip at all")
        with self.assertRaises(UploadRejected) as caught:
            validate_upload_set(files, AEMO_MONTHLY_POLICY)
        self.assertEqual(caught.exception.code, "SIGNATURE_MISMATCH")
        self.assertEqual(caught.exception.position, 2)

    def test_documented_sources_entry_point_returns_record_safe_evidence(self):
        evidence = validate_documented_sources(
            [Upload(name, aemo_shaped_archive()) for name in DOCUMENTED_AEMO_NAMES],
            Upload("Solar_home_half_hour_data.zip", ausgrid_shaped_archive()),
        )
        self.assertEqual(evidence["accepted_upload_count"], 13)
        self.assertEqual(len(evidence["aemo_uploads"]), 12)
        self.assertEqual(len(evidence["ausgrid_uploads"]), 1)
        self.assertEqual(len({item["sha256"] for item in evidence["aemo_uploads"]}), 1)
        self.assertGreater(evidence["session_byte_budget"], evidence["accepted_total_bytes"])

    def test_missing_ausgrid_upload_is_rejected_not_skipped(self):
        """A short upload set fails server-side, not only through a disabled button."""

        with self.assertRaises(UploadRejected) as caught:
            validate_documented_sources(
                [Upload(name, aemo_shaped_archive()) for name in DOCUMENTED_AEMO_NAMES], None
            )
        self.assertEqual(caught.exception.code, "TOO_FEW_FILES")

    def test_short_aemo_set_is_rejected(self):
        with self.assertRaises(UploadRejected) as caught:
            validate_upload_set(
                [Upload(name, aemo_shaped_archive()) for name in DOCUMENTED_AEMO_NAMES[:11]], AEMO_MONTHLY_POLICY
            )
        self.assertEqual(caught.exception.code, "TOO_FEW_FILES")


class SafeNameTest(unittest.TestCase):
    def test_hostile_names_are_reduced_before_display(self):
        self.assertEqual(safe_display_name("../../etc/passwd"), "passwd")
        self.assertEqual(safe_display_name("C:\\Windows\\system32\\cmd.exe"), "cmd.exe")
        # A name that looks like a path is reduced to its final segment first,
        # then to safe characters, so neither markup nor a path survives.
        self.assertEqual(safe_display_name("<script>alert(1)</script>"), "script?")
        self.assertEqual(safe_display_name(""), "(unnamed)")
        self.assertEqual(safe_display_name(None), "(unnamed)")
        self.assertLessEqual(len(safe_display_name("a" * 300)), 64)

    def test_every_code_has_a_message(self):
        for code in REJECTION_CODES:
            self.assertTrue(UploadRejected(code).message.endswith("."))
        with self.assertRaises(KeyError):
            UploadRejected("NOT_A_REAL_CODE")


if __name__ == "__main__":
    unittest.main()
