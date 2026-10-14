# Presentation demonstration runbook

## Before the presentation

1. Use the repository `.venv`, not the system Python installation.
2. Start the app with `python -m streamlit run streamlit_app.py`.
3. Upload the 12 documented AEMO monthly ZIPs and `Ausgrid_solar_home_data.zip`.
4. Select **Read sources and prepare validation**.
5. Run XGBoost and LSTM for AEMO, then repeat for Ausgrid.
6. Keep the app and computer awake. Reloading the page may clear the prepared session.
7. Keep [`streamlit_real_data_smoke_2026-10-14.json`](streamlit_real_data_smoke_2026-10-14.json), screenshots and the slide PDF available as fallbacks.

Do not train a model live during the presentation. The real-data workflow takes several minutes on CPU. Prepare the results before the audience arrives and demonstrate the retained session.

## Three-minute live demonstration

| Time | Action | What to say |
| --- | --- | --- |
| 0:00–0:25 | Show the accepted source profile and split counts | “The platform validates the documented archives and uses chronological train, validation and reserved test partitions.” |
| 0:25–1:05 | Select AEMO and show the comparison table | “XGBoost achieved MAE 90.37 MW, improving on persistence at 188.01 MW and seasonal naive at 393.42 MW.” |
| 1:05–1:35 | Show actual-versus-predicted output and SHAP | “The previous observed slot dominates the model, while SHAP additivity is checked before prediction clipping. These are model associations, not causal effects.” |
| 1:35–2:15 | Switch to Ausgrid and show the comparison | “XGBoost has the lowest MAE at 0.0676 kWh per source slot. LSTM has a slightly lower RMSE, so model choice depends on the selected success metric.” |
| 2:15–2:35 | Point to the zero-exclusion count | “PV MAPE excludes 2,029 zero-generation targets, so MAE and RMSE are the primary PV measures.” |
| 2:35–3:00 | Show the forecast and reproducibility downloads | “The user can export predictions, aggregate results and an environment/configuration record for replay.” |

## Recovery plan

- If the live session is lost, use the result table and screenshots in the slides.
- If LSTM is still training, demonstrate the prepared XGBoost results and cite the recorded LSTM metrics.
- If an upload is rejected, confirm that all 12 AEMO monthly ZIPs and the original Ausgrid archive were selected; do not disable validation.
- Do not describe the prototype as a live service or production deployment.

## Manual usability review

Two team members who did not implement the interface should independently perform the following and record their names, duration, result and defects:

| Scenario | Expected outcome |
| --- | --- |
| Upload all documented source archives | Sources are accepted and both profiles are shown without an error |
| Attempt preparation with a missing AEMO month | The set is rejected with a safe, understandable message |
| Run XGBoost for AEMO | Comparison, chart, SHAP and CSV download become available |
| Run LSTM for Ausgrid | Comparison, chart and CSV download become available |
| Download validation summary and reproducibility record | Both JSON files download and contain the selected case |

Automated real-data widget testing passed on 14 October 2026. This is system evidence, not a substitute for the two independent human usability reviews.
