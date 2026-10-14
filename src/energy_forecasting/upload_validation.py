"""Server-side upload validation for the documented source archives.

Nothing here trusts the browser. Streamlit's ``type="zip"`` argument only
filters the file picker, and a request can be replayed without it, so every
rule below is re-checked after the bytes arrive (OWASP, n.d.). The checks run
cheapest-first and fail closed: an upload is rejected unless every rule passes.

Two rules exist because this project accepts *archives*, not single CSV files.
Declared size says nothing about expanded size, so expansion is budgeted, and
archive member names are attacker-controlled strings that must never be used
as filesystem paths. Serialised-model and executable members are refused
outright rather than merely ignored: loading a pickle executes the code inside
it (Bieringer et al., 2022), and a member that is never meant to be opened is
cheaper to reject at the boundary than to defend everywhere downstream.

Rejections carry a stable code from ``REJECTION_CODES`` and a fixed message.
Neither is built from the uploaded name, the archive contents or a caught
exception's text, so nothing about the server or the submitted file is
reflected back to the caller (NFR1). Descriptors record only sizes, counts and
a digest, never a dataset reading (NFR4).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import stat
import zipfile

ZIP_LOCAL_FILE_SIGNATURE = b"PK\x03\x04"
ZIP_EMPTY_ARCHIVE_SIGNATURE = b"PK\x05\x06"

# Members refused whatever the policy allows. Serialised model formats execute
# or reconstruct code on load; the rest are executable or script content. This
# list is the reason B6's pickle concern does not depend on reviewer vigilance.
DENIED_MEMBER_SUFFIXES = frozenset({
    ".pkl", ".pickle", ".joblib", ".npy", ".npz", ".pt", ".pth", ".ckpt", ".h5", ".hdf5", ".pb", ".model", ".bin",
    ".exe", ".dll", ".so", ".dylib", ".com", ".scr", ".msi", ".jar", ".class",
    ".bat", ".cmd", ".ps1", ".psm1", ".sh", ".bash", ".zsh", ".vbs", ".wsf", ".js", ".mjs", ".py", ".pyc", ".pyo",
    ".lnk", ".reg", ".apk", ".php", ".jsp", ".asp", ".aspx", ".htaccess",
})

# Characters accepted in a member or upload name. Anything else is replaced
# before the name is displayed, so a hostile name cannot reach a log or a page
# intact.
_SAFE_NAME_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ._-()"
)

REJECTION_CODES = (
    "EMPTY_NAME",
    "UNSAFE_NAME",
    "SUFFIX_NOT_ALLOWED",
    "EMPTY_FILE",
    "FILE_TOO_LARGE",
    "SIGNATURE_MISMATCH",
    "ARCHIVE_UNREADABLE",
    "ARCHIVE_ENCRYPTED",
    "TOO_MANY_MEMBERS",
    "MEMBER_PATH_UNSAFE",
    "MEMBER_NOT_REGULAR_FILE",
    "MEMBER_SUFFIX_DENIED",
    "MEMBER_SUFFIX_NOT_ALLOWED",
    "NESTING_TOO_DEEP",
    "EXPANSION_BUDGET_EXCEEDED",
    "COMPRESSION_RATIO_EXCEEDED",
    "DUPLICATE_MEMBER_NAME",
    "SESSION_BUDGET_EXCEEDED",
    "TOO_MANY_FILES",
    "TOO_FEW_FILES",
)

# One fixed sentence per code. Fixed text is what keeps NFR1 checkable: the
# message cannot vary with server state or with what was uploaded.
_REJECTION_MESSAGES = {
    "EMPTY_NAME": "The upload has no file name.",
    "UNSAFE_NAME": "The file name contains characters or path separators that are not accepted.",
    "SUFFIX_NOT_ALLOWED": "Only ZIP source archives are accepted for this input.",
    "EMPTY_FILE": "The file is empty.",
    "FILE_TOO_LARGE": "The file is larger than the accepted size limit for this input.",
    "SIGNATURE_MISMATCH": "The file content is not a ZIP archive, whatever its name suggests.",
    "ARCHIVE_UNREADABLE": "The archive could not be read as a valid ZIP file.",
    "ARCHIVE_ENCRYPTED": "Encrypted archive members are not accepted.",
    "TOO_MANY_MEMBERS": "The archive contains more members than the accepted limit.",
    "MEMBER_PATH_UNSAFE": "An archive member name is absolute or escapes the archive directory.",
    "MEMBER_NOT_REGULAR_FILE": "An archive member is a link or another non-regular entry.",
    "MEMBER_SUFFIX_DENIED": "The archive contains an executable or serialised-object member.",
    "MEMBER_SUFFIX_NOT_ALLOWED": "The archive contains a member type that is not accepted for source data.",
    "NESTING_TOO_DEEP": "The archive nests deeper than the documented source layout.",
    "EXPANSION_BUDGET_EXCEEDED": "The archive expands beyond the accepted uncompressed size.",
    "COMPRESSION_RATIO_EXCEEDED": "The archive expands at a ratio that is not accepted.",
    "DUPLICATE_MEMBER_NAME": "The archive contains repeated member names.",
    "SESSION_BUDGET_EXCEEDED": "The uploads total more than the accepted size for one session.",
    "TOO_MANY_FILES": "More files were uploaded than this input accepts.",
    "TOO_FEW_FILES": "Fewer files were uploaded than this input requires.",
}


class UploadRejected(Exception):
    """A validation rule refused an upload.

    ``code`` is stable and safe to log or display. ``str(self)`` is the fixed
    sentence for that code and contains no uploaded or server-side text.
    """

    def __init__(self, code: str, position: int | None = None, safe_name: str | None = None) -> None:
        if code not in _REJECTION_MESSAGES:
            raise KeyError(f"Unknown rejection code: {code}")
        super().__init__(_REJECTION_MESSAGES[code])
        self.code = code
        self.position = position
        self.safe_name = safe_name

    @property
    def message(self) -> str:
        """The fixed, reflection-free sentence for this rejection."""

        return _REJECTION_MESSAGES[self.code]


def safe_display_name(name: str, limit: int = 64) -> str:
    """Reduce a submitted name to characters that are safe to display or log.

    The result is for humans only. It is never reopened as a path, and it is
    never used to build a rejection message.
    """

    if not isinstance(name, str):
        return "(unnamed)"
    trimmed = name.strip().rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    cleaned = "".join(character if character in _SAFE_NAME_CHARACTERS else "?" for character in trimmed)
    cleaned = cleaned.strip()
    if not cleaned:
        return "(unnamed)"
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 3] + "..."


@dataclass(frozen=True)
class UploadPolicy:
    """The limits applied to one named input.

    Every field is a decision, not a fact about the data. ``provisional``
    records whether the figure has been measured or agreed; an unmeasured
    limit is still enforced, but the specification says so rather than
    presenting it as settled.
    """

    name: str
    max_bytes: int
    allowed_suffixes: frozenset[str] = frozenset({".zip"})
    allowed_member_suffixes: frozenset[str] = frozenset({".csv", ".txt", ".md", ".pdf", ".zip"})
    max_members: int = 4096
    max_nesting_depth: int = 1
    max_expanded_bytes: int = 512 * 1024 * 1024
    max_expansion_ratio: float = 200.0
    max_files: int = 1
    min_files: int = 1
    provisional: bool = True
    provisional_note: str = ""

    def describe(self) -> dict:
        """Return the policy as record-safe metadata for an experiment record."""

        return {
            "name": self.name,
            "max_bytes": self.max_bytes,
            "allowed_suffixes": sorted(self.allowed_suffixes),
            "allowed_member_suffixes": sorted(self.allowed_member_suffixes),
            "max_members": self.max_members,
            "max_nesting_depth": self.max_nesting_depth,
            "max_expanded_bytes": self.max_expanded_bytes,
            "max_expansion_ratio": self.max_expansion_ratio,
            "max_files": self.max_files,
            "min_files": self.min_files,
            "provisional": self.provisional,
            "provisional_note": self.provisional_note,
        }


# The 10 MB figure is the week-two working assumption recorded in Part B2. It
# was not benchmarked, and it is carried here as provisional so that changing
# it is a recorded decision rather than an edit.
AEMO_MONTHLY_POLICY = UploadPolicy(
    name="aemo_monthly_zip",
    max_bytes=10 * 1024 * 1024,
    allowed_member_suffixes=frozenset({".zip", ".csv"}),
    max_nesting_depth=2,
    # A 31-day monthly archive holds 31 daily ZIPs plus the CSV inside each, so
    # 62 members is normal and 64 left almost no margin. Raised so one extra
    # file in a source archive is not a rejection.
    max_members=256,
    max_expanded_bytes=64 * 1024 * 1024,
    max_files=12,
    min_files=12,
    provisional=True,
    provisional_note=(
        "10 MB per monthly archive is the Part B2 working assumption; it has not been benchmarked "
        "against the 60-second prediction target (Risk R3) and the two figures are still treated separately."
    ),
)

# The Ausgrid solar-home archive holds three wide annual CSV files and is an
# order of magnitude larger than a monthly AEMO archive. Applying the 10 MB
# figure here would reject the documented source, so the input carries its own
# limit. The number below is a headroom figure, not a measurement: the copy in
# the data manifest has to be measured before this stops being provisional.
AUSGRID_ARCHIVE_POLICY = UploadPolicy(
    name="ausgrid_solar_home_zip",
    max_bytes=64 * 1024 * 1024,
    allowed_member_suffixes=frozenset({".csv", ".txt", ".md", ".pdf"}),
    max_nesting_depth=1,
    max_members=64,
    max_expanded_bytes=512 * 1024 * 1024,
    max_files=1,
    min_files=1,
    provisional=True,
    provisional_note=(
        "Set as headroom above the documented copy, not measured. The single 10 MB figure in Part B2 "
        "cannot apply to this input without rejecting the documented Ausgrid archive."
    ),
)

# A whole-session ceiling, so twelve individually acceptable archives plus the
# Ausgrid archive cannot be used to exhaust memory together.
MAX_SESSION_BYTES = 192 * 1024 * 1024


@dataclass(frozen=True)
class UploadDescriptor:
    """Record-safe evidence that one upload passed validation.

    Deliberately holds no reading from the dataset: sizes, counts, a digest
    and sanitised names only, so an experiment record or a log line built from
    this cannot carry source values (NFR4).
    """

    policy_name: str
    safe_name: str
    byte_size: int
    sha256: str
    member_count: int
    expanded_bytes: int
    max_member_expansion_ratio: float
    deepest_nesting: int
    member_suffixes: tuple[str, ...] = field(default=())

    def describe(self) -> dict:
        return {
            "policy_name": self.policy_name,
            "safe_name": self.safe_name,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "member_count": self.member_count,
            "expanded_bytes": self.expanded_bytes,
            "max_member_expansion_ratio": round(self.max_member_expansion_ratio, 3),
            "deepest_nesting": self.deepest_nesting,
            "member_suffixes": list(self.member_suffixes),
        }


def _suffix(name: str) -> str:
    base = name.rsplit("/", 1)[-1]
    return "." + base.rsplit(".", 1)[-1].lower() if "." in base else ""


def _reject(code: str, position: int | None, safe_name: str | None) -> UploadRejected:
    return UploadRejected(code, position=position, safe_name=safe_name)


def _check_name(name: str, policy: UploadPolicy, position: int | None, safe_name: str) -> None:
    if not isinstance(name, str) or not name.strip():
        raise _reject("EMPTY_NAME", position, safe_name)
    if "/" in name or "\\" in name or "\x00" in name or name.strip() in {".", ".."}:
        raise _reject("UNSAFE_NAME", position, safe_name)
    if any(character not in _SAFE_NAME_CHARACTERS for character in name.strip()):
        raise _reject("UNSAFE_NAME", position, safe_name)
    if _suffix(name) not in policy.allowed_suffixes:
        raise _reject("SUFFIX_NOT_ALLOWED", position, safe_name)


def _check_member_name(member_name: str, position: int | None, safe_name: str) -> None:
    """Refuse a member name that is absolute, escaping, or otherwise unusable.

    The name is never joined to a filesystem path in this project, but a
    reviewer cannot verify that for every future caller, so the escape check
    happens once here (OWASP, n.d.).
    """

    if "\x00" in member_name or member_name.startswith(("/", "\\")) or "\\" in member_name:
        raise _reject("MEMBER_PATH_UNSAFE", position, safe_name)
    if len(member_name) > 2 and member_name[1] == ":":
        raise _reject("MEMBER_PATH_UNSAFE", position, safe_name)
    if any(part == ".." for part in member_name.split("/")):
        raise _reject("MEMBER_PATH_UNSAFE", position, safe_name)


def _walk_archive(data: bytes, policy: UploadPolicy, position: int | None, safe_name: str, depth: int = 1) -> dict:
    """Inspect one archive level, recursing into nested archives.

    Returns counts only. Member payloads are read to measure expansion and to
    recurse; no CSV content is retained or parsed here.
    """

    if depth > policy.max_nesting_depth:
        raise _reject("NESTING_TOO_DEEP", position, safe_name)
    totals = {"members": 0, "expanded": 0, "ratio": 0.0, "depth": depth, "suffixes": set()}
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError, ValueError, EOFError) as error:
        raise _reject("ARCHIVE_UNREADABLE", position, safe_name) from error
    with archive:
        try:
            infos = archive.infolist()
        except (zipfile.BadZipFile, OSError, ValueError) as error:
            raise _reject("ARCHIVE_UNREADABLE", position, safe_name) from error
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise _reject("DUPLICATE_MEMBER_NAME", position, safe_name)
        if len(infos) > policy.max_members:
            raise _reject("TOO_MANY_MEMBERS", position, safe_name)
        for info in infos:
            _check_member_name(info.filename, position, safe_name)
            if info.flag_bits & 0x1:
                raise _reject("ARCHIVE_ENCRYPTED", position, safe_name)
            if info.is_dir():
                continue
            mode = info.external_attr >> 16
            if mode and not stat.S_ISREG(mode):
                raise _reject("MEMBER_NOT_REGULAR_FILE", position, safe_name)
            suffix = _suffix(info.filename)
            if suffix in DENIED_MEMBER_SUFFIXES:
                raise _reject("MEMBER_SUFFIX_DENIED", position, safe_name)
            if suffix not in policy.allowed_member_suffixes:
                raise _reject("MEMBER_SUFFIX_NOT_ALLOWED", position, safe_name)

            totals["members"] += 1
            totals["suffixes"].add(suffix)
            # Declared sizes come from the archive and are not trusted; both the
            # declared figure and the bytes actually read are budgeted.
            if info.file_size > policy.max_expanded_bytes:
                raise _reject("EXPANSION_BUDGET_EXCEEDED", position, safe_name)
            if info.compress_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio > policy.max_expansion_ratio:
                    raise _reject("COMPRESSION_RATIO_EXCEEDED", position, safe_name)
                totals["ratio"] = max(totals["ratio"], ratio)
            try:
                payload = archive.read(info)
            except (zipfile.BadZipFile, OSError, ValueError, RuntimeError, NotImplementedError) as error:
                raise _reject("ARCHIVE_UNREADABLE", position, safe_name) from error
            totals["expanded"] += len(payload)
            if totals["expanded"] > policy.max_expanded_bytes:
                raise _reject("EXPANSION_BUDGET_EXCEEDED", position, safe_name)
            if suffix == ".zip":
                nested = _walk_archive(payload, policy, position, safe_name, depth + 1)
                totals["members"] += nested["members"]
                totals["expanded"] += nested["expanded"]
                totals["ratio"] = max(totals["ratio"], nested["ratio"])
                totals["depth"] = max(totals["depth"], nested["depth"])
                totals["suffixes"] |= nested["suffixes"]
                if totals["members"] > policy.max_members:
                    raise _reject("TOO_MANY_MEMBERS", position, safe_name)
                if totals["expanded"] > policy.max_expanded_bytes:
                    raise _reject("EXPANSION_BUDGET_EXCEEDED", position, safe_name)
    return totals


def validate_upload(name: str, data: bytes, policy: UploadPolicy, position: int | None = None) -> UploadDescriptor:
    """Validate one uploaded archive against a policy, or raise ``UploadRejected``.

    Order matters: name and size are checked before any byte of content is
    interpreted, and the content signature is checked before the ZIP reader is
    handed the data, so a mislabelled or oversized file never reaches a parser.
    """

    safe_name = safe_display_name(name if isinstance(name, str) else "")
    _check_name(name, policy, position, safe_name)
    if not isinstance(data, (bytes, bytearray)):
        raise _reject("EMPTY_FILE", position, safe_name)
    payload = bytes(data)
    if not payload:
        raise _reject("EMPTY_FILE", position, safe_name)
    if len(payload) > policy.max_bytes:
        raise _reject("FILE_TOO_LARGE", position, safe_name)
    if payload[:4] == ZIP_EMPTY_ARCHIVE_SIGNATURE:
        raise _reject("EMPTY_FILE", position, safe_name)
    if payload[:4] != ZIP_LOCAL_FILE_SIGNATURE:
        raise _reject("SIGNATURE_MISMATCH", position, safe_name)

    totals = _walk_archive(payload, policy, position, safe_name)
    if not totals["members"]:
        raise _reject("EMPTY_FILE", position, safe_name)
    return UploadDescriptor(
        policy_name=policy.name,
        safe_name=safe_name,
        byte_size=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        member_count=totals["members"],
        expanded_bytes=totals["expanded"],
        max_member_expansion_ratio=totals["ratio"],
        deepest_nesting=totals["depth"],
        member_suffixes=tuple(sorted(totals["suffixes"])),
    )


def validate_upload_set(files, policy: UploadPolicy, session_budget: int = MAX_SESSION_BYTES) -> list[UploadDescriptor]:
    """Validate every file submitted for one input.

    ``files`` may be Streamlit ``UploadedFile`` objects or anything exposing
    ``name`` and ``getvalue()``. Validation is all-or-nothing: the first
    rejection stops the set, because a partially accepted upload set is not a
    state this prototype has a defined behaviour for.
    """

    items = list(files or [])
    if len(items) > policy.max_files:
        raise _reject("TOO_MANY_FILES", None, None)
    # The count contract is enforced here and not only by the interface. A
    # disabled button is a client-side control, so a short upload set has to be
    # refused on this side as well.
    if len(items) < policy.min_files:
        raise _reject("TOO_FEW_FILES", None, None)
    descriptors: list[UploadDescriptor] = []
    running_total = 0
    for position, item in enumerate(items, start=1):
        data = item.getvalue()
        running_total += len(data or b"")
        if running_total > session_budget:
            raise _reject("SESSION_BUDGET_EXCEEDED", position, safe_display_name(getattr(item, "name", "")))
        descriptors.append(validate_upload(getattr(item, "name", ""), data, policy, position=position))
    return descriptors


def validate_documented_sources(aemo_files, ausgrid_file) -> dict:
    """Validate both documented inputs and return record-safe evidence.

    This is the single entry point the app calls before any parser runs. It
    returns the accepted-upload evidence that the experiment record needs; it
    does not read or interpret the source data, which stays the parsers' job.
    """

    aemo_descriptors = validate_upload_set(aemo_files, AEMO_MONTHLY_POLICY)
    ausgrid_descriptors = validate_upload_set(
        [ausgrid_file] if ausgrid_file is not None else [], AUSGRID_ARCHIVE_POLICY
    )
    total_bytes = sum(item.byte_size for item in aemo_descriptors + ausgrid_descriptors)
    if total_bytes > MAX_SESSION_BYTES:
        raise _reject("SESSION_BUDGET_EXCEEDED", None, None)
    return {
        "policies": [AEMO_MONTHLY_POLICY.describe(), AUSGRID_ARCHIVE_POLICY.describe()],
        "aemo_uploads": [item.describe() for item in aemo_descriptors],
        "ausgrid_uploads": [item.describe() for item in ausgrid_descriptors],
        "accepted_upload_count": len(aemo_descriptors) + len(ausgrid_descriptors),
        "accepted_total_bytes": total_bytes,
        "session_byte_budget": MAX_SESSION_BYTES,
    }
