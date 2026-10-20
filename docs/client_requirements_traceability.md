# Client and Innovation Concept Requirements Traceability

This matrix connects the client brief and the submitted Innovation Concept to implemented, testable evidence. It distinguishes the working prototype from future production capabilities so the presentation does not overstate the system.

| Requirement | Implementation evidence | Status |
| --- | --- | --- |
| Forecast electricity load or photovoltaic energy | AEMO NSW1 demand, Ausgrid customer 1 gross PV and mapped custom energy targets | Implemented |
| Accept prepared data or an approved-schema CSV | Guided input method with documented ZIP sources or one UTF-8 CSV | Implemented |
| Map timestamp, target and optional covariates | Custom CSV mapping controls; selected covariates must be numeric and known at forecast time | Implemented |
| Validate type, size, timestamps, missing values, duplicates and frequency | Bounded upload adapter with specific correction messages and automated tests | Implemented |
| Automated preprocessing and feature engineering | Chronological mapping, lag 1, configurable seasonal lag, calendar fields and train-only scaling | Implemented for regular univariate energy series |
| Chronological train, validation and test partitions | All three inputs produce ordered partitions; the final test is locked until model settings are confirmed | Implemented |
| Seasonal-naive benchmark | Same-target seasonal lag evaluated on the same validation and test targets | Implemented |
| At least two ML or DL approaches | XGBoost and a PyTorch LSTM | Implemented |
| Fair common evaluation | MAE, RMSE and positive-actual MAPE on aligned partitions and horizons | Implemented |
| Residual and seasonal error analysis | Residual chart plus mean absolute error grouped by hour | Implemented |
| Explainability | Global and local native SHAP contributions with additivity verification | Implemented for XGBoost |
| Guided user interface | Streamlit workflow covering upload, mapping, validation, modelling, comparison, final evaluation and export | Implemented |
| Download forecasts and experiment metadata | Validation/test forecast CSVs, comparison CSV, summary JSON and reproducibility JSON | Implemented |
| Reproducibility | Source digest, schema mapping, split boundaries, environment, seeds, settings and metric digests | Implemented |
| Public online prototype | Repository contains one-file cloud dependencies and Streamlit configuration; Python 3.12 is selected during cloud setup | Deployment-ready; public URL still requires account deployment |
| User feedback | Conduct and record short usability reviews with intended users | Pending team activity |
| Production or live-market operation | Explicitly outside the submitted semester scope | Excluded |

## Controlled implementation deviations

- The prototype uses reusable internal Python model services instead of Darts. This preserves the submitted layered architecture while avoiding a large deployment dependency. The presentation should describe the common service contract, not claim that Darts is currently used.
- Hyperparameters are bounded fixed configurations with validation early stopping. A full Optuna study is not yet part of the working prototype and should be presented as future optimisation rather than an implemented feature.
- Custom CSV support requires a regular time series and performs one-step forecasting. Arbitrary irregular schemas, multi-target forecasting and recursive multi-step horizons remain future work.
- SHAP applies to XGBoost. LSTM attribution was explicitly conditional on feasibility in the submitted concept and remains outside the core prototype.
