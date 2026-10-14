"""Build SYNTHETIC source archives in the documented shape, for testing only.

Run: python scripts/make_demo_sources.py OUTPUT_DIRECTORY

The real AEMO and Ausgrid archives are large downloads with licence terms, which
makes the end-to-end path awkward to exercise. This writes archives that satisfy
the documented input contract exactly — 12 AEMO monthly ZIPs covering 17,520
consecutive NSW1 half-hours, and an Ausgrid-shaped ZIP with 731 consecutive
actual, no-blank days for customer 1 `GG` — so upload validation, parsing,
chronological splitting, training and the experiment record can all be run
without any real data.

**The readings are invented.** They come from a fixed formula, not from a
measurement, and no number produced from them may be reported as a result. They
exist to prove the pipeline runs and reproduces, not to say anything about NSW
demand or any household's PV generation. A manifest saying so is written beside
the archives.

Two deliberate properties:

- **Byte-deterministic.** No clock value and no unseeded randomness enters the
  output, so running this twice gives identical archives. Without that, the
  archives could not be used to test NFR5 — a differing upload digest would be
  indistinguishable from a reproducibility failure.
- **Distinguishable from real data.** The archives' SHA-256 digests go into
  every experiment record, so a record made from these can never be mistaken
  for one made from the documented sources.
"""

import csv
from datetime import date, datetime, timedelta
import hashlib
import io
from math import cos, pi, sin
from pathlib import Path
import sys
import zipfile

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from energy_forecasting.aemo import EXPECTED_FIELDS  # noqa: E402
from energy_forecasting.demo_data import EXPECTED_AEMO_NAMES  # noqa: E402

# Interval-ending timestamps, 30 minutes apart, for the documented 12 months.
AEMO_FIRST_INTERVAL_END = datetime(2025, 8, 1, 0, 30)
AEMO_INTERVAL_COUNT = 17_520
AUSGRID_YEARS = (("Solar home 2011-2012.csv", date(2011, 7, 1), date(2012, 6, 30)),
                 ("Solar home 2012-2013.csv", date(2012, 7, 1), date(2013, 6, 30)))
AUSGRID_MEMBER_ORDER = [name for name, _start, _end in AUSGRID_YEARS]
# A fixed stand-in for AEMO's LASTCHANGED column. A real timestamp here would
# make the output differ between runs.
FIXED_LASTCHANGED = "2026/08/01 00:00:00"

MANIFEST_NAME = "SYNTHETIC_SOURCES_README.txt"


def demand_mw(interval_end: datetime) -> float:
    """An invented load shape: daily double peak, weekly dip, annual swing.

    Deliberately smooth and obviously artificial. Always positive, so it passes
    the parser's non-negative rule without special-casing.
    """

    slot = interval_end.hour * 2 + interval_end.minute // 30
    day = interval_end.timetuple().tm_yday
    morning = 900.0 * sin(pi * min(slot, 47) / 47.0) ** 4
    evening = 1300.0 * sin(pi * max(0.0, (slot - 26) / 21.0)) ** 3
    weekly = -350.0 if interval_end.weekday() >= 5 else 0.0
    annual = 700.0 * cos(2 * pi * (day - 15) / 365.0)
    # A fixed, repeatable wobble so consecutive values are not perfectly smooth.
    wobble = 60.0 * sin(slot * 2.7 + day * 1.3)
    return round(max(50.0, 6200.0 + morning + evening + weekly + annual + wobble), 2)


def pv_kwh(source_date: date, slot: int) -> float:
    """An invented PV shape: zero overnight, a bell through the middle of the day.

    Zeros are intentional — they are what makes ordinary MAPE undefined on this
    case, which is behaviour the pipeline has to handle.
    """

    if slot < 13 or slot > 39:
        return 0.0
    day = source_date.timetuple().tm_yday
    span = (slot - 13) / 26.0
    seasonal = 0.75 + 0.25 * cos(2 * pi * (day - 350) / 365.0)
    cloud = 0.88 + 0.12 * sin(day * 0.9 + slot * 0.4)
    return round(max(0.0, 0.95 * seasonal * cloud * sin(pi * span) ** 2), 3)


def aemo_daily_csv(interval_ends: list[datetime]) -> bytes:
    """One AEMO ACTUAL_DAILY CSV in the C/I/D interchange format the parser expects."""

    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(["C", "NEMP.WORLD", "SYNTHETIC_TEST_DATA", "AEMO", "PUBLIC", "", "", ""])
    writer.writerow(["I", "OPERATIONAL_DEMAND", "ACTUAL", "3", *EXPECTED_FIELDS])
    for interval_end in interval_ends:
        writer.writerow([
            "D", "OPERATIONAL_DEMAND", "ACTUAL", "3",
            "NSW1",
            interval_end.strftime("%Y/%m/%d %H:%M:%S"),
            f"{demand_mw(interval_end):.2f}",
            "0",
            "0",
            FIXED_LASTCHANGED,
        ])
    writer.writerow(["C", "END OF REPORT", str(len(interval_ends) + 3), "", "", "", "", ""])
    return buffer.getvalue().encode("utf-8")


def build_aemo_archives() -> dict[str, bytes]:
    """Group the 17,520 intervals into the 12 documented monthly archives."""

    intervals = [AEMO_FIRST_INTERVAL_END + timedelta(minutes=30 * step) for step in range(AEMO_INTERVAL_COUNT)]
    by_day: dict[date, list[datetime]] = {}
    for interval_end in intervals:
        # An interval is attributed to the day it starts in, so the 00:00
        # interval-ending value belongs to the previous day, as AEMO publishes it.
        by_day.setdefault((interval_end - timedelta(minutes=30)).date(), []).append(interval_end)

    archives: dict[str, bytes] = {}
    for source_day, day_intervals in sorted(by_day.items()):
        name = f"PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_{source_day.year}{source_day.month:02d}01.zip"
        archives.setdefault(name, []).append((source_day, day_intervals))

    result = {}
    for name, days in archives.items():
        monthly = io.BytesIO()
        # Stored, not deflated: the daily members are already compressed, which
        # is also how AEMO ships them.
        with zipfile.ZipFile(monthly, "w", zipfile.ZIP_STORED) as outer:
            for source_day, day_intervals in sorted(days):
                stamp = source_day.strftime("%Y%m%d")
                daily = io.BytesIO()
                with zipfile.ZipFile(daily, "w", zipfile.ZIP_DEFLATED) as inner:
                    info = zipfile.ZipInfo(f"PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_{stamp}.CSV", (1980, 1, 1, 0, 0, 0))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    info.external_attr = 0o644 << 16 | 0o100000 << 16
                    inner.writestr(info, aemo_daily_csv(day_intervals))
                outer_info = zipfile.ZipInfo(f"PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_{stamp}.zip", (1980, 1, 1, 0, 0, 0))
                outer_info.compress_type = zipfile.ZIP_STORED
                outer_info.external_attr = 0o644 << 16 | 0o100000 << 16
                outer.writestr(outer_info, daily.getvalue())
        result[name] = monthly.getvalue()
    return result


def ausgrid_annual_csv(start: date, end: date) -> bytes:
    """One wide annual CSV: 48 half-hour columns per source day, plus Row Quality."""

    clock_labels = []
    for slot in range(1, 49):
        minutes = 30 * slot
        clock_labels.append(f"{(minutes // 60) % 24}:{minutes % 60:02d}")
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\r\n")
    writer.writerow(["SYNTHETIC TEST DATA - not an Ausgrid file - invented readings"])
    writer.writerow(["Customer", "Generator Capacity", "Postcode", "Consumption Category", "date",
                     *clock_labels, "Row Quality"])
    source_date = start
    while source_date <= end:
        writer.writerow([
            "1", "1.95", "2000", "GG", source_date.strftime("%d-%b-%y"),
            *[f"{pv_kwh(source_date, slot):.3f}" for slot in range(48)],
            "",
        ])
        source_date += timedelta(days=1)
    return buffer.getvalue().encode("utf-8")


def build_ausgrid_archive() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, start, end in AUSGRID_YEARS:
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16 | 0o100000 << 16
            archive.writestr(info, ausgrid_annual_csv(start, end))
    return buffer.getvalue()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/make_demo_sources.py OUTPUT_DIRECTORY")
    target = Path(sys.argv[1])
    target.mkdir(parents=True, exist_ok=True)

    aemo = build_aemo_archives()
    if set(aemo) != EXPECTED_AEMO_NAMES:
        raise SystemExit(f"Generated names do not match the documented contract: {sorted(set(aemo) ^ EXPECTED_AEMO_NAMES)}")
    ausgrid_name = "Solar_home_half_hour_data_SYNTHETIC.zip"
    ausgrid = build_ausgrid_archive()

    written = []
    for name, payload in sorted(aemo.items()):
        (target / name).write_bytes(payload)
        written.append((name, len(payload), hashlib.sha256(payload).hexdigest()))
    (target / ausgrid_name).write_bytes(ausgrid)
    written.append((ausgrid_name, len(ausgrid), hashlib.sha256(ausgrid).hexdigest()))

    manifest = [
        "SYNTHETIC SOURCE ARCHIVES - FOR TESTING ONLY",
        "",
        "Generated by scripts/make_demo_sources.py. The readings in these archives are",
        "INVENTED by a fixed formula. They are not AEMO data and not Ausgrid data.",
        "",
        "No metric, chart or experiment record produced from these files may be reported",
        "as a result. They exist to exercise upload validation, parsing, chronological",
        "splitting, training and the experiment record without the real archives.",
        "",
        "Generation is byte-deterministic, so a reproducibility check can use them: an",
        "identical upload digest across two runs means the inputs really were identical.",
        "The digests below differ from the documented sources', so a record made from",
        "these archives is always distinguishable from a real one.",
        "",
        "Do not commit these files. Do not place them in a directory holding real data.",
        "",
        f"{'BYTES':>12}  SHA-256                                                           FILE",
    ]
    for name, size, sha in written:
        manifest.append(f"{size:>12,}  {sha}  {name}")
    (target / MANIFEST_NAME).write_text("\n".join(manifest) + "\n", encoding="utf-8")

    print(f"Wrote {len(written)} synthetic archives to {target}\n")
    print(f"{'BYTES':>12}  FILE")
    for name, size, _sha in written:
        print(f"{size:>12,}  {name}")
    print(f"\nManifest: {target / MANIFEST_NAME}")
    print("\nThese readings are INVENTED. Nothing computed from them is a result.")
    print("Upload the 12 AEMO archives and the Ausgrid archive in the dashboard to exercise")
    print("the full path, or run: python scripts/replay_check.py --from-sources " + str(target))
