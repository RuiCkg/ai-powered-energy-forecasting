# Local data manifest and quality review

These are SHA-256 identifiers for the **specific local investigation copies** obtained on 15 September 2026. They are not official publisher checksums. The raw files are outside Git and are not redistributed here. AEMO's rolling archive or later actual-data updates may give other collaborators different files; a changed hash requires a new access date and re-profile, not silent acceptance of old reported results. Source links and terms are in [data_sources.md](data_sources.md).

| Archive file | Bytes | SHA-256 |
| --- | ---: | --- |
| `PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_20250801.zip` | 78269 | `9936ce19839b2858005fb7ec18a6961c21c482a6d6978acf88184e94ac0ab266` |
| `PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_20250901.zip` | 75259 | `4966412f4fbe090b31a2b6b499792c7e21cfe18567aa3236a4a82e3c9635970e` |
| `PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_20251001.zip` | 77676 | `9358740bd43343aeb3aa5d6e270b3aaa320697f05c008452430b86ce44e92fe2` |
| `PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_20251101.zip` | 74813 | `270765afb8f886aec32e8da890bb0258664adc98f64697446c5d13107eae9612` |
| `PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_20251201.zip` | 77389 | `591b87cc5d3831cd03cd5b0c4034aa4a0894129104e91e7bd9b06fb172a5db22` |
| `PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_20260101.zip` | 77652 | `45dfff02846ba2035b309349eea18d31a062ee02e25d59e3393d74fda835d9d3` |
| `PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_20260201.zip` | 70390 | `79d238bc88fd67b29c053487baebc8308086bcec8e49a97a243a7f6879318afa` |
| `PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_20260301.zip` | 77883 | `24ea97ba52b67668c4d1b4b8dccf39391466af4e3a23a28669b9223a707a4c74` |
| `PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_20260401.zip` | 75354 | `9812274b1b0e848c978b81d14f72d45ba7e06d2d23ce6d468f1815777364da20` |
| `PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_20260501.zip` | 77623 | `d55c00548e6236e5659f1d460e828c31b31eb201b5c30256ef932008b268c80a` |
| `PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_20260601.zip` | 75450 | `c0b6b4b0ba9b430a2cb505bd7205388a9d8f6de30866bb85bc457150834e393e` |
| `PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_20260701.zip` | 77660 | `e6eeba26d220395ae67c1a502d9884841fa3977a4699b4de11681c056d92cecf` |
| `Ausgrid_solar_home_data.zip` | 57038633 | `5a766f52b6c8b3b72730380f4422e478934bc94640a4b089dd0e0e3c055c5d82` |

## Inspection findings that should not be silently corrected

- AEMO `NSW1` has 17,520 consecutive source intervals, no blank targets, no zeros and a minimum of **2,848 MW at 29 December 2025 12:30 fixed NEM UTC+10**. There are 178 readings under 4,000 MW. The minimum's neighboring intervals are also low, so it is not an isolated one-point spike, but source context still needs confirmation.
- The largest absolute adjacent half-hour change is **1,803 MW at the 26 November 2025 13:00 target** (8,786 to 6,983 MW). Nearby intervals include 6,660 MW at 13:30 and 7,322 MW at 14:00. This is a review flag, **not** evidence that the value is wrong. No records were dropped or altered before the first model experiments.
- The selected Ausgrid customer `1` `GG` series has 731 consecutive source dates, 35,088 original slots, no blank readings, no `NA` quality rows and 18,314 exact zero PV values. Its 48 slots/day cannot safely be converted to absolute timestamps using the AEMO fixed-offset rule; the daylight-saving policy remains a team decision.
- Source parsers reject non-finite and negative target readings. Structural and physical-plausibility tests are separate: non-null and consecutive records do not by themselves prove every value correct.

The [README](../README.md) and [source inventory](data_sources.md) give reproduction commands. The local Streamlit dashboard displays these source profile fields. For later copies, compute SHA-256 locally, record publisher, access date and differences, then rerun profiles and model validation instead of quoting this manifest as universal data truth.
