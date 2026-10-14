# Technical MVP status, 14 October 2026

## Implemented and verified locally

- Read 12 AEMO `ACTUAL_DAILY` monthly ZIPs and the Ausgrid solar-home archive; checked 17,520 consecutive NSW1 half-hours and 731 consecutive, actual-quality, no-blank days for selected customer 1 `GG`.
- Built causal one-step source examples, 48-value LSTM histories, and provisional chronological train/validation/test partitions. The test targets were **not** used in model fitting, early stopping, checkpoint selection or reported ML/DL scores.
- Computed previous-day seasonal-naive and previous-slot persistence references; trained one XGBoost and one small LSTM per case; reported validation MAE/RMSE and positive-actual MAPE with zero exclusions.
- Added XGBoost's native SHAP contributions on validation rows, plus a local Streamlit dashboard for source review, case-by-case comparison, actual-versus-predicted plots, and downloadable validation CSV/JSON.
- Refined the dashboard with a five-step workflow, compact upload status, rounded on-screen metrics, best-model summary cards, four result tabs, non-colour-only forecast lines, plain-language chart summaries and explicit processing/completion states.
- Passed 75 automated tests plus 52 upload-validation subtests, covering parsers, splits, input contracts, features, models, SHAP additivity, security, reproducibility and accessible UI structure. Reopened both real source types through in-memory ZIPs and confirmed the local dashboard starts and responds to a health check.
- Recorded copy-specific hashes and AEMO low/abrupt-change review flags in the [data manifest](data_manifest.md), plus Rui's [integration handoff](rui_data_handoff.md). No anomalous reading was silently removed.

## Not yet settled or delivered

- Facilitator/client confirmation of the **forecast horizon**, target/region/household choice, operational success criteria and live-input availability.
- Ausgrid daylight-saving conversion and day-boundary prediction. The current PV result is explicitly a next-**source-slot** case study, not an absolute-time service.
- Real-time ingestion, model persistence, later unseen forecast generation, deployment, security/usability review and final held-out **test** evaluation. The dashboard demonstrates historical validation only.
- Licence/provenance decision for redistributing the Ausgrid mirror; source terms must be rechecked. Raw archives stay outside Git.
- Final team acceptance, backlog closure and client/facilitator confirmation. Hitesh's interface integration and Viet's explainability/QA work have been reviewed and merged into `main`; the real-data XGBoost and LSTM validation runs were reproduced on 14 October 2026.

## Suggested next decision sequence

1. Ask the team/facilitator to approve the one-step goal or choose a more useful forecasting horizon, along with the two provisional case definitions and source terms.
2. Have another team member reproduce source profiles, baselines and both model validation runs from the documented archives.
3. Freeze the model-selection plan and preprocessing policy; then evaluate the reserved test periods **once**, without tuning again on those results.
4. Add a true later-unseen forecast workflow, handle source-time policy and real-time input latency, and record acceptance-test evidence before calling it a deliverable platform.
