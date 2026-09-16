# Rui's data and integration handoff

The Week 6 team worklog assigns Rui **project coordination and data integration**: maintain the backlog, coordinate client questions, profile both datasets and integrate deliverables. This handoff records what is demonstrably done locally and what still needs another person's decision or action. It is not a claim that the client/facilitator has approved the provisional design.

## Completed local technical work

- Selected and profiled two **separate** cases: AEMO `NSW1` operational demand and Ausgrid customer `1` gross PV (`GG`). [Source inventory](data_sources.md) and [copy manifest](data_manifest.md) record target, units, timestamps/source slots, source quality, access and licence caveats.
- Built and tested source adapters, chronology checks, baseline forecasting examples, 48-value past-only LSTM sequences and train/validation/test partitioning. Both models share the same one-step target examples and evaluation definitions **within each case**; their MW and kWh outputs are never numerically pooled.
- Connected the adapters to a runnable [local dashboard](../streamlit_app.py) with input validation, reference/model comparison, charts, XGBoost SHAP and downloadable validation forecasts. Test data is reserved, not used for model fitting/selection or displayed in the dashboard.
- Added pinned dependency files, parser/feature tests and reproduction guidance. Another team member has **not yet** independently repeated the workflow, so repository-level reproducibility is still unverified.

## Concrete team/facilitator questions to send and record

1. Is the intended forecast **one half-hour/source slot ahead**, or does the client need another horizon such as several hours or a day ahead? The present performance numbers apply only to the one-step task.
2. Are regional `NSW1` demand and one Ausgrid customer's `GG` gross PV suitable as the two cases, or should the team use different regions, households, consumption (`GC`) or aggregations? Do not describe this pair as proof of cross-dataset generalization.
3. For Ausgrid, how should the 48 published clock columns be reconciled with daylight-saving transitions, especially if future work needs absolute-time or day-boundary predictions?
4. Which acceptance criteria matter: MAE/RMSE target, usable time horizon, latency of latest actual readings, explanation needs, supported upload format and accessibility? No numerical error threshold has been approved.
5. May the team use an independently hosted Ausgrid archive copy for analysis, and may either raw source be redistributed to collaborators or through the repository? The mirror checksum identifies only our local copy, not publisher authenticity.

Record the response, date and source in the backlog or meeting minutes. Until answered, label dataset and forecast decisions **provisional**.

## Backlog entries ready for GitHub Projects

| Priority | Item | Completion evidence | Suggested owner |
| --- | --- | --- | --- |
| High | Confirm target, horizon and acceptance criteria with facilitator/client | Written decision with date and agreed units/horizon | Rui, team review |
| High | Confirm data source use terms and Ausgrid copy provenance | Licence/redistribution decision linked in source inventory | Rui and Sineth |
| High | Independently reproduce both source profiles and validation runs | Another member's run notes, checksums, test output and comparison metrics | Team member other than Rui |
| High | Review abrupt AEMO demand step and low-demand dates | Documented source check; keep or exclude only under a predeclared rule | Rui and Khang |
| Medium | Freeze model-selection and test-use protocol | Reviewed split/horizon/metrics plan before final test | Khang and Viet |
| Medium | Add actual later-unseen prediction workflow and daylight-saving policy | Demonstration beyond historical validation with documented timing | Hitesh and integration reviewer |

These are **proposed** backlog entries, not evidence that they have been posted to GitHub Projects. No action is assigned to another member until they agree to it.

## Reproduction and evidence check

From the repository, use the documented archive copies outside Git, create a Python 3.12+ virtual environment, install the three requirement files, then run:

```text
python -m unittest discover -s tests -v
python scripts/profile_aemo_months.py DIRECTORY_CONTAINING_12_AEMO_MONTHLY_ZIPS NSW1
python scripts/profile_ausgrid.py PATH_TO_AUSGRID_SOLAR_HOME_DATA.zip 1 GG "Solar home 2012-2013.csv"
python scripts/run_seasonal_baseline.py DIRECTORY_CONTAINING_12_AEMO_MONTHLY_ZIPS PATH_TO_AUSGRID_SOLAR_HOME_DATA.zip
python scripts/run_xgboost_validation.py DIRECTORY_CONTAINING_12_AEMO_MONTHLY_ZIPS PATH_TO_AUSGRID_SOLAR_HOME_DATA.zip
python scripts/run_lstm_validation.py DIRECTORY_CONTAINING_12_AEMO_MONTHLY_ZIPS PATH_TO_AUSGRID_SOLAR_HOME_DATA.zip
python -m streamlit run streamlit_app.py
```

Expected data counts: AEMO 17,520 raw half-hour records; Ausgrid selected 731 days / 35,088 source slots. Forecast validation counts: AEMO 2,928 targets; PV 4,230 targets. Match archive hashes first if repeating the **same** investigation copy; otherwise record the new copy and rerun. The LSTM can take several minutes on CPU. Source data should not be committed.

The remaining coordination actions need team/client involvement and cannot be certified by a local code run alone. GitHub publication should be checked for current remote changes and reviewed by the team before representing this as a shared increment.
