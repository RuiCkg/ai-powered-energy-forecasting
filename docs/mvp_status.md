# Technical MVP status, 16 September 2026

## Implemented and verified locally

- Read 12 AEMO `ACTUAL_DAILY` monthly ZIPs and the Ausgrid solar-home archive; checked 17,520 consecutive NSW1 half-hours and 731 consecutive, actual-quality, no-blank days for selected customer 1 `GG`.
- Built causal one-step source examples, 48-value LSTM histories, and provisional chronological train/validation/test partitions. The test targets were **not** used in model fitting, early stopping, checkpoint selection or reported ML/DL scores.
- Computed previous-day seasonal-naive and previous-slot persistence references; trained one XGBoost and one small LSTM per case; reported validation MAE/RMSE and positive-actual MAPE with zero exclusions.
- Added XGBoost's native SHAP contributions on validation rows, plus a local Streamlit dashboard for source review, case-by-case comparison, actual-versus-predicted plots, and downloadable validation CSV/JSON.
- Passed 12 parser/split/input-contract/feature tests, including rejection of negative and non-finite source readings; reopened both real source types through in-memory ZIPs, and confirmed the local dashboard starts and responds to a health check.
- Recorded copy-specific hashes and AEMO low/abrupt-change review flags in the [data manifest](data_manifest.md), plus Rui's [integration handoff](rui_data_handoff.md). No anomalous reading was silently removed.

## Not yet settled or delivered

- Facilitator/client confirmation of the **forecast horizon**, target/region/household choice, operational success criteria and live-input availability.
- Ausgrid daylight-saving conversion and day-boundary prediction. The current PV result is explicitly a next-**source-slot** case study, not an absolute-time service.
- Real-time ingestion, model persistence, later unseen forecast generation, deployment, security/usability review and final held-out **test** evaluation. The dashboard demonstrates historical validation only.
- Licence/provenance decision for redistributing the Ausgrid mirror; source terms must be rechecked. Raw archives stay outside Git.
- Team review/merge and actual backlog/client coordination. Local handoff material exists, but another member has not reproduced the workflow yet.

## Suggested next decision sequence

1. Ask the team/facilitator to approve the one-step goal or choose a more useful forecasting horizon, along with the two provisional case definitions and source terms.
2. Have another team member reproduce source profiles, baselines and both model validation runs from the documented archives.
3. Freeze the model-selection plan and preprocessing policy; then evaluate the reserved test periods **once**, without tuning again on those results.
4. Add a true later-unseen forecast workflow, handle source-time policy and real-time input latency, and record acceptance-test evidence before calling it a deliverable platform.
