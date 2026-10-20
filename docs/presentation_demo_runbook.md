# Presentation demonstration runbook

## Before the presentation

1. Use the repository `.venv` and start the app with `python -m streamlit run streamlit_app.py`.
2. In the main browser tab, upload the documented AEMO and Ausgrid archives, validate them and run XGBoost and LSTM before the audience arrives.
3. In a second tab, open a separate session and select **Custom CSV** so the mapping workflow can be shown without clearing the prepared real-data results.
4. Keep screenshots, exported comparison CSVs and the recorded JSON evidence available as fallbacks.
5. Do not run the reserved test live unless the team has formally frozen model settings and intends that run to be the final evaluation.

Do not train the full real-data LSTM during the presentation. Prepare the results first and retain the session.

## Four-minute live demonstration

| Time | Action | What to explain |
| --- | --- | --- |
| 0:00-0:35 | Show the **Custom CSV** input method | “A user can map a timestamp, energy target and optional covariates known at forecast time. The platform is no longer limited to two hard-coded source layouts.” |
| 0:35-1:00 | Show validation rules and workflow progress | “Missing, duplicate, unordered, irregular and non-numeric data are rejected before modelling.” |
| 1:00-1:35 | Return to the prepared AEMO session and open **Overview** | “The same chronological partitions and metrics are used for the baseline, XGBoost and LSTM.” |
| 1:35-2:15 | Open **Forecasts** and residual analysis | “The interface shows both prediction fit and systematic error patterns, not only one headline score.” |
| 2:15-2:50 | Open **Explainability (XGBoost)** | “SHAP identifies model associations globally and locally, and an additivity test checks the calculation. It does not establish causation.” |
| 2:50-3:25 | Open **Final test** | “The test partition remains untouched during development and is evaluated only after both model settings are fixed on validation.” |
| 3:25-4:00 | Open **Downloads** | “Forecasts, comparison metrics and a reproducibility record can be exported for review.” |

## Recovery plan

- If a live session is lost, use the exported evidence and screenshots in the slides.
- If LSTM is still training, demonstrate XGBoost and cite the previously recorded LSTM result.
- If a source is rejected, read the correction message; do not disable validation.
- Do not describe the prototype as real-time, production-scale, multi-horizon or autonomously operational.

## Human usability review

Two team members who did not implement the interface should independently record their name, duration, result and defects for these scenarios:

| Scenario | Expected outcome |
| --- | --- |
| Upload and map a valid custom CSV | Mapping is understandable and the source is prepared without code editing |
| Upload a CSV with a duplicate timestamp | Preparation is rejected with a specific correction message |
| Upload all documented source archives | Both source profiles and chronological split counts are shown |
| Run XGBoost | Comparison, forecast, residual, SHAP and download evidence appear |
| Run LSTM | The same comparison and forecast evidence appear without duplicate controls |
| Inspect the final-test gate | It clearly prevents accidental test use before both models are complete and confirmation is given |
| Download the summary and reproducibility record | Files contain the selected dataset, mapping, partitions, settings and metrics |

Automated tests support the review but do not replace these human usability checks.
