# Candidate data sources

This is a source inventory for the first data-preparation increment. Dataset targets and forecasting horizons are not final until the actual files have been profiled and confirmed with the project facilitator or client.

| Source | Intended case | Published data | Status |
| --- | --- | --- | --- |
| [AEMO operational demand](https://aemo.com.au/energy-systems/electricity/national-electricity-market-nem/data-nem/operational-demand-data) | Regional electricity load | Historical actual operational demand by region; half-hourly and daily publication options | One monthly archive profiled; target and region selection still provisional |
| [Ausgrid Solar Home Electricity Data](https://data.gov.au/data/dataset/5ab48b70-5e99-47d3-9193-5c34a2676d93) | Household PV generation or consumption | Half-hour electricity measurements for 300 rooftop-solar homes | Archive structure inspected; original Ausgrid download URL currently returns 404 |

Ausgrid's data.gov.au entry lists a Creative Commons Attribution 3.0 Australia licence. AEMO publishes a disclaimer and file-retention rules on its source page; the team must check the terms for the exact downloaded file before redistribution. Raw data should not be committed to this repository until file size, provenance, licence and privacy have been checked.

## AEMO initial profile

The [July 2026 `ACTUAL_DAILY` archive](https://nemweb.com.au/Reports/ARCHIVE/Operational_Demand/ACTUAL_DAILY/PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_20260701.zip) was downloaded on 15 September 2026 for a first format and quality check. AEMO's monthly ZIP contains 31 daily ZIPs, each containing a CSV in its `C/I/D` interchange format. The actual-data fields are `REGIONID`, `INTERVAL_DATETIME`, `OPERATIONAL_DEMAND`, `OPERATIONAL_DEMAND_ADJUSTMENT`, `WDR_ESTIMATE` and `LASTCHANGED`. `OPERATIONAL_DEMAND` is the provisional load target, in MW. AEMO defines its NEM interval timestamps using a fixed UTC+10 offset; do not interpret them as daylight-saving local time.

For `NSW1`, this sample month contains 1,488 records, from `2026-07-01 04:30+10:00` through `2026-08-01 04:00+10:00`. There are 1,488 distinct timestamps, zero missing target values and zero deviations from 30-minute spacing. The observed target range is 5,924-11,422 MW (mean 8,630.07 MW). This is one month of evidence, not yet enough to establish the final training period or forecast horizon.

After downloading the archive to a local location outside the Git repository, reproduce the profile with:

```text
python scripts/profile_aemo.py PATH_TO_PUBLIC_ACTUAL_OPERATIONAL_DEMAND_DAILY_20260701.zip NSW1
```

The parser is in `src/energy_forecasting/aemo.py` and has a small in-memory archive test in `tests/test_aemo.py`. It reads and profiles data but does not yet impute missing values, engineer features or split into training/validation/test periods.

### Twelve-month NSW1 source profile

The official [AEMO monthly archive directory](https://nemweb.com.au/Reports/ARCHIVE/Operational_Demand/ACTUAL_DAILY/) contained August 2025 through July 2026 at the time of access. We downloaded those 12 monthly ZIPs to a local folder **outside the Git repository** on 15 September 2026. Joining the `NSW1` actual records gives 17,520 half-hourly values from `2025-08-01 04:30+10:00` through `2026-08-01 04:00+10:00`, with no duplicate timestamps, no missing targets and no non-30-minute gaps. The observed range is 2,848-13,182 MW (mean 7,537.87 MW). The range and unusually low readings still need inspection against the source; spacing and non-null checks alone do not prove physical validity. This is a **provisional regional load case**, not yet a model-ready split.

After obtaining the listed monthly archives from AEMO, reproduce the joined profile with:

```text
python scripts/profile_aemo_months.py DIRECTORY_CONTAINING_MONTHLY_ZIPS NSW1
```

AEMO notes that later daily actual updates can supersede original real-time records. Check the relevant update archive before treating this sample as an immutable final truth. The official archive is rolling, so later collaborators may need to record a new access date and available period rather than assuming these same monthly ZIPs remain online.

## Ausgrid initial format inspection

The original Ausgrid download URL currently returns 404. For investigation only, a [researcher's archive copy](https://pierreh.eu/downloads/Ausgrid_solar_home_data.zip) was downloaded on 15 September 2026. Its SHA-256 is `5a766f52b6c8b3b72730380f4422e478934bc94640a4b089dd0e0e3c055c5d82`. This is not a checksum published by Ausgrid, so it identifies this copy but does not independently prove that it is identical to the former official files. Do not redistribute this copy through the project repository without checking the source and licence decision with the team/facilitator.

The archive contains an Ausgrid notes PDF dated August 2014 and three annual half-hourly CSV files (`2010-2011`, `2011-2012` and `2012-2013`). Each CSV has five descriptive columns followed by 48 half-hour energy columns. A row represents a customer, consumption category and date. The category codes in the notes are `GG` (gross PV generation), `GC` (general consumption) and `CL` (controlled-load consumption). The interval values are **kWh for the half hour ending at the column label**, not kW or MW; the final `0:00` interval belongs to the end of that date. The 2011-2012 and 2012-2013 files also have a `Row Quality` field: blank means all actual meter readings; `NA` indicates that some or all values are estimates or substitutes. The 2010-2011 CSV lacks this column.

The Ausgrid notes state that the 48 clock columns use Eastern Standard Time in winter and Eastern Daylight Savings Time in summer. This is **not** AEMO's fixed NEM UTC+10 convention. A date/clock column should therefore not be combined with the AEMO timestamp parser. The daylight-saving transition policy and handling of `Row Quality = NA` must be chosen and documented before creating a leakage-free long-form time series or evaluating models.

The provisional Ausgrid forecasting target is `GG` for one documented customer or a documented household aggregation. This choice still requires a completeness and quality profile; `GC` remains a separate consumption case and must not be silently added to PV generation.

As an initial feasibility check, customer `1` in the 2012-2013 CSV has 365 `GG` daily rows covering 1 July 2012 through 30 June 2013, or 17,520 half-hour values. In this selected `GG` series there are zero blank half-hour values and the observed range is 0-1.875 kWh per interval. All 1,095 customer-1 rows across `CL`, `GC` and `GG` have a blank `Row Quality` field. This is a **single-customer, single-year sample**, not a quality claim for all 300 homes or all three years. The daylight-saving interpretation still needs a documented policy.

After keeping the archive outside the Git repository, reproduce the selected `GG` profile with:

```text
python scripts/profile_ausgrid.py PATH_TO_AUSGRID_SOLAR_HOME_DATA.zip 1 GG "Solar home 2012-2013.csv"
```

The parser in `src/energy_forecasting/ausgrid.py` preserves the source date and all 48 clock labels. It reports blank values and `Row Quality` without assigning an absolute timestamp. A small wide-row test is in `tests/test_ausgrid.py`.

### Full 2012-2013 source-quality profile

The annual CSV contains 300 customers and 300 `GG` series. Across `GG`, there are 109,419 customer-day rows (81 fewer than 300 x 365), no blank interval fields and no duplicate customer/category/date rows. There are 146 `GG` rows flagged `Row Quality = NA`, meaning some or all readings were estimated or substituted. **273 customers** have the full 365 source dates, no blank interval fields and no `NA` rows in this year; customer `1` is among them. This is a practical *selection rule*, not proof that every numerical reading is physically plausible or that daylight-saving timestamps have been resolved. The CSV is a non-representative sample of homes; do not claim statistical generalization to all households.

Reproduce this streaming profile with:

```text
python scripts/profile_ausgrid_annual.py PATH_TO_AUSGRID_SOLAR_HOME_DATA.zip "Solar home 2012-2013.csv"
```

Our initial implementation candidate is a documented **single-customer `GG` series**, such as customer `1`, rather than immediately aggregating all homes. Aggregation would require an explicit missing-day, `NA` and daylight-saving policy. Before training, inspect plausible daily patterns and choose the forecast horizon with the team. Any performance on this customer should be reported as a case study, not a cross-home generalization result.

For customer `1`, all three annual `GG` files have continuous source dates and no blank interval values: 365 days in 2010-2011, 366 days in 2011-2012, and 365 days in 2012-2013. The latter two files have no `NA` quality rows for this customer, giving a **provisional 731-day, 35,088-slot 2011-2013 case study**. The 2010-2011 file lacks the `Row Quality` column, so its 365 rows cannot be certified as all actual by that field; keep it out of the initial quality-filtered training period. This selection is based on source checks only. No model accuracy or forecast horizon has been established.

Of the 35,088 `GG` interval values in these two selected years, **18,314 are exactly zero** (9,093 in 2011-2012 and 9,221 in 2012-2013). This is critical for the required MAPE metric: division by zero makes ordinary MAPE undefined for those intervals. Report MAE and RMSE across all valid intervals; if MAPE is reported for PV, state explicitly that it is calculated only where actual generation is positive and report the excluded-zero count or coverage. Do not silently replace zero actuals with an arbitrary epsilon or compare that restricted MAPE directly with load MAPE.

## Provisional leakage-aware source split

The reproducible script `scripts/profile_splits.py` tests a first chronological **source-record** split. Boundaries are provisional and should be confirmed once the forecast horizon is agreed. It does not yet create lagged features, forecast windows or trained models.

| Case | Train | Validation | Test |
| --- | --- | --- | --- |
| AEMO `NSW1` | Aug 2025-Mar 2026; 11,664 half-hour records | Apr-May 2026; 2,928 records | Jun-Jul 2026; 2,928 records |
| Ausgrid customer `1`, `GG` | Jul 2011-Dec 2012; 550 source days, 26,400 slots | Jan-Mar 2013; 90 days, 4,320 slots | Apr-Jun 2013; 91 days, 4,368 slots |

AEMO boundaries are exact fixed-NEM-UTC+10 interval timestamps at `04:30` on the first day of April and June, matching the monthly archive boundary. Ausgrid boundaries are **whole source dates**, so all 48 original half-hour slots of a day stay in the same partition; they are not assigned absolute timestamps while its daylight-saving rule is unresolved. Later supervised windows must ensure their **target interval** belongs to the intended partition and that input features use only data available before the forecast origin. The cases must be scored separately because they measure different phenomena and units.

Run this check after placing the documented 12 AEMO month files and the Ausgrid ZIP outside the repository:

```text
python scripts/profile_splits.py DIRECTORY_CONTAINING_AEMO_ZIPS PATH_TO_AUSGRID_SOLAR_HOME_DATA.zip
```

## First causal reference forecasts, not trained ML models

The first end-to-end check predicts **one AEMO half-hour ahead** using the actual interval immediately before the target as the forecast origin. For Ausgrid, it predicts the **next source half-hour slot within the same row/day**; no day-boundary step or absolute daylight-saving timestamp is assumed. Two simple references are calculated: **previous-day seasonal naive** (same target slot yesterday) and **previous-slot persistence** (the immediately preceding actual). They use only history preceding the target, and each example is scored in the partition containing its *target*, even when a causal lag comes from an earlier partition. The first AEMO day and first Ausgrid source day cannot have a previous-day lag, so they do not produce reference examples.

| Case, provisional test partition | Forecast examples | Seasonal-naive MAE | Persistence MAE | Zero actuals excluded from MAPE |
| --- | ---: | ---: | ---: | ---: |
| AEMO `NSW1` demand, MW | 2,928 | 411.81 MW | 237.95 MW | 0 |
| Ausgrid customer `1` `GG`, kWh/half-hour | 4,277 | 0.0894 kWh | 0.0571 kWh | 2,555 |

The PV MAPE is large partly because **small positive actuals** can produce very large percentage errors; excluding exact zeros alone does not make it a stable overall accuracy summary. MAE and RMSE cover all evaluated valid targets. These figures describe **untrained reference forecasts** on the selected historical files and provisional split. They do not establish performance for XGBoost, LSTM, other households, a 24-hour forecast, or a deployed platform. Do not compare the MW and kWh errors directly. For a live prototype, confirm that the immediately previous *actual* reading would really be available at the forecast origin; a downloadable historical file does not prove real-time data latency.

Reproduce train/validation/test baseline metrics with:

```text
python scripts/run_seasonal_baseline.py DIRECTORY_CONTAINING_AEMO_ZIPS PATH_TO_AUSGRID_SOLAR_HOME_DATA.zip
```

The [first fixed-configuration XGBoost experiment](first_ml_experiment.md) fits a supervised model to the available causal lag features. It keeps fitting and early stopping inside the training/validation workflow and leaves XGBoost test scoring untouched until final model comparison. The agreed forecast horizon and Ausgrid time interpretation must be documented before calling this a final project specification.

## Profiling checklist

For each source, record:

- Download URL, access date, file name and licence or use terms.
- Actual column names and the selected timestamp and target columns.
- Units, time zone, sampling interval, coverage and number of records.
- Missing timestamps and values, duplicates, outliers and any corrections.
- Candidate forecast horizon and a reason that the usable history supports it.
- Leakage-free chronological train, validation and test boundaries.

The two datasets will be evaluated separately. The shared pipeline will map each to `timestamp` and `target`, with optional covariates retained only when their availability at prediction time is clear. This common format does not imply that regional demand and household PV values are directly comparable.

## First evidence milestone

The data-preparation increment is complete only when another team member can follow the repository instructions to obtain both datasets, run the profiling and validation steps, and reproduce the documented outputs. A diagram or report description alone does not count as a working module.
