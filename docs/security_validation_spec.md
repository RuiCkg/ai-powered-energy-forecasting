# Upload validation, rejection matrix and reproducibility record

Security lead workstream, 28 September 2026. Covers the requirements assigned to
this role in Part B1: the upload validation framework and accepted-type schema,
the security testing approach and its rejection scenarios, and the
reproducibility criterion. It describes what is in the codebase now, and says
plainly which figures are still provisional.

Implemented in `src/energy_forecasting/upload_validation.py` and
`src/energy_forecasting/experiment_record.py`. Evidence is in
`tests/test_upload_validation.py`, `tests/test_experiment_record.py` and
`tests/test_app_wiring.py`.

## 1. Requirements and how each one is checked

Requirements are claimed met by evidence, not by intention. Each row names the
test that fails if the claim stops being true.

| Requirement | Statement | Met when | Evidence |
| --- | --- | --- | --- |
| FR1 | Source archives are accepted only after server-side validation of type, size, count and archive structure | Every documented input passes and every scenario in §3 is refused | `AcceptedUploadTest`, `RejectionMatrixTest` |
| FR2 | A rejected upload produces a specific, stable reason and no partially accepted state | Each scenario returns its own code from `REJECTION_CODES`; a bad file voids the whole set | `test_every_scenario_is_rejected_with_its_code`, `test_one_bad_file_rejects_the_whole_set`, `test_a_rejection_clears_any_previously_prepared_state` |
| NFR1 | All eight invalid files are denied, and the response body leaks no internals | The eight core scenarios are refused, and no message contains a path, a module name, exception text or the submitted name | `test_eight_core_invalid_files_are_denied`, `test_rejection_messages_leak_nothing` |
| NFR4 | No dataset value reaches the upload evidence or the experiment record | Accepted-upload evidence is sizes, counts and digests only; a record carrying per-row readings is refused at build time | `test_descriptor_carries_no_dataset_values`, `test_per_row_values_are_dropped_from_the_record`, `test_building_a_record_with_planted_rows_fails_at_build_time` |
| NFR5 | A recorded configuration replayed gives figures that are the same, not similar | Identical configuration digests produce identical metric digests; a difference of 1e-11 is a failure | `test_same_inputs_reproduce_the_same_digests`, `test_a_close_metric_is_still_a_failure` |

NFR1 previously named "eight invalid files" without saying which eight. They are
now the eight scenarios marked `core` in §3, fixed in code so the count cannot
drift.

## 2. Accepted-input schema

One policy per input, because one set of limits cannot fit both. Each policy is
an `UploadPolicy` and is recorded into every experiment record, so a limit
cannot change without the change appearing in the evidence.

| Rule | AEMO monthly archives | Ausgrid solar-home archive |
| --- | --- | --- |
| Container type | `.zip`, confirmed by content signature | `.zip`, confirmed by content signature |
| File count | exactly 12 | exactly 1 |
| Size per file | 10 MB — **provisional** | 64 MB — **provisional** |
| Accepted members | `.zip`, `.csv` | `.csv`, `.txt`, `.md`, `.pdf` |
| Nesting depth | 2 (monthly ZIP of daily ZIPs of CSV) | 1 |
| Members per archive | 256 | 64 |
| Expanded size | 64 MB | 512 MB |
| Expansion ratio | 200:1 | 200:1 |
| Session total | 192 MB across all uploads | 192 MB across all uploads |

Applied to every upload regardless of policy:

- **Order.** Name, then extension, then size, then content signature, then
  archive structure. No parser is handed bytes that have not passed all of it.
- **Server-side only.** The file picker's `type="zip"` argument filters a
  dialog; a request can be replayed without it, so no rule depends on it
  (OWASP, n.d.). `test_uploads_are_validated_before_they_are_parsed` asserts the
  gate still runs first.
- **Executable and serialised members are denied** ahead of any allow-list, so a
  future permissive policy cannot re-admit them:
  `test_denied_suffix_beats_an_allowed_suffix_list`. This is the B6 concern
  enforced in code rather than left to review — loading a serialised object can
  execute the code inside it (Bieringer et al., 2022).
- **Member names are never trusted as paths.** Absolute, drive-qualified and
  `..`-containing names are refused, and names are reduced to a safe character
  set before they are displayed or logged.
- **Member counts fit the real sources.** A 31-day AEMO monthly archive holds 31
  daily ZIPs plus a CSV in each, so 62 members is normal. The AEMO limit is 256
  rather than a figure that sits just above the observed count.
- **All-or-nothing.** One rejected file voids the set. A partially accepted
  upload set has no defined behaviour in this prototype, so it is not a state
  the app can reach.

### Correction to Part B2

Part B2 recorded that "CSV will be the only input format accepted". The
documented sources are ZIP archives, so the accepted format is a ZIP
*container* whose data members are CSV. The B2 statement was written before the
source layout was settled and is superseded by §2. The stricter reading of
B2 — CSV uploaded directly — would reject both documented sources.

## 3. Rejection matrix

Every row is a test that builds a real file and submits it through the same
entry point the app uses. `core` marks the eight invalid files NFR1 names.

| ID | Tier | Scenario | Code |
| --- | --- | --- | --- |
| S1 | core | Extension not on the allow-list | `SUFFIX_NOT_ALLOWED` |
| S2 | core | ZIP name, non-ZIP content | `SIGNATURE_MISMATCH` |
| S3 | core | Larger than the size limit for the input | `FILE_TOO_LARGE` |
| S4 | core | Zero-byte upload | `EMPTY_FILE` |
| S5 | core | Compression-bomb expansion ratio | `COMPRESSION_RATIO_EXCEEDED` |
| S6 | core | Member name escaping the archive directory | `MEMBER_PATH_UNSAFE` |
| S7 | core | Serialised-object member that would execute on load | `MEMBER_SUFFIX_DENIED` |
| S8 | core | Submitted name carrying a path separator | `UNSAFE_NAME` |
| S9 | extra | Truncated or corrupt archive | `ARCHIVE_UNREADABLE` |
| S10 | extra | Encrypted archive member | `ARCHIVE_ENCRYPTED` |
| S11 | extra | Nesting deeper than the documented layout | `NESTING_TOO_DEEP` |
| S12 | extra | Member type outside the source-data allow-list | `MEMBER_SUFFIX_NOT_ALLOWED` |
| S13 | extra | Symbolic-link member | `MEMBER_NOT_REGULAR_FILE` |
| S14 | extra | Repeated member names | `DUPLICATE_MEMBER_NAME` |
| S15 | extra | More members than the limit | `TOO_MANY_MEMBERS` |
| S16 | extra | Archive with no readable member | `EMPTY_FILE` |
| S17 | extra | Control character in the submitted name | `UNSAFE_NAME` |
| S18 | extra | Absolute member path | `MEMBER_PATH_UNSAFE` |
| S19 | extra | Upload submitted with no name | `EMPTY_NAME` |
| S20 | extra | Expanded size over the budget at an accepted ratio | `EXPANSION_BUDGET_EXCEEDED` |

Also covered, at the level of a whole upload set: too many files (`TOO_MANY_FILES`),
too few files (`TOO_FEW_FILES`), and the session byte budget
(`SESSION_BUDGET_EXCEEDED`). Every code in `REJECTION_CODES` is exercised by a
test; none is an unreachable branch.

### The no-leak rule

Every rejection message is a fixed sentence selected by code. None is built from
the submitted name, a member name, a caught exception or any server value, so
the second half of NFR1 is a property of the design rather than a review
finding. `test_rejection_messages_leak_nothing` checks every scenario's message
for path separators, `Traceback`, `Errno`, module names, `.py`, and the
submitted name itself.

## 4. Reproducibility record

NFR5 asks for figures that are the same on replay, not close, so the record
holds everything that can move a number:

- interpreter version and implementation, platform, and the versions of numpy,
  scipy, xgboost, torch and streamlit (absent packages recorded as null, not
  omitted);
- the fixed model configuration **read from the model modules**, not restated —
  `xgb_model.FIXED_PARAMETERS`, `NUM_BOOST_ROUND`, `EARLY_STOPPING_ROUNDS`,
  `FEATURE_NAMES` and `lstm_model.FIXED_CONFIG`, so a record cannot disagree
  with the run that produced it;
- the preprocessing policy in force, including that nothing is imputed and that
  splits are chronological in code;
- the upload policies and a SHA-256 digest of each accepted archive, so a run is
  tied to the exact bytes it used;
- the chronological split boundaries and the example counts per partition;
- aggregate metrics per model, with per-row previews and the single local SHAP
  explanation removed.

The record is split into `configuration` and `metrics`, each with its own
digest. **The criterion: equal `configuration_digest` with unequal
`metrics_digest` is a reproducibility failure.** `compare_records` reports which
field moved, because "it did not reproduce" is only actionable with the field
named.

Two supporting properties are tested rather than assumed. The record is a
snapshot: it shares no structure with the caller's live results, so evidence
cannot change after the fact. And serialisation is canonical — sorted keys,
fixed float formatting, non-finite values refused — so two records of the same
run are byte-identical.

### What this does not establish

Seeds are recorded and fixed, but bit-identical LSTM results across different
machines, thread counts or torch builds have not been demonstrated. The record
makes such a difference **visible and attributable** rather than preventing it.
Reproducing a run on a second machine is the outstanding step, and it belongs
with the mvp_status item asking another member to reproduce the workflow.

## 5. Provisional decisions

Recorded as provisional so that changing one is a decision, not an edit.

| Decision | Status | Note |
| --- | --- | --- |
| 10 MB per AEMO archive | provisional, unmeasured | The week-two working assumption from B2. Never benchmarked. |
| 64 MB for the Ausgrid archive | provisional, unmeasured | Headroom, not a measurement. The documented copy has to be measured. |
| 192 MB session total | provisional | Chosen so twelve acceptable archives plus the Ausgrid archive cannot exhaust memory together. |
| 200:1 expansion ratio | provisional | Well above the documented sources' ratios; not tuned against a hostile sample. |
| No authentication | decided | Client specifies a narrow group of trusted users. |
| Member allow-lists | deliberately broad | Per B2: broad as defensible now, tightened on confirmation. The deny-list is the hard rule. |

### Risk R3 is still open

Part B2 records the disagreement: the 10 MB size limit and the 60-second
prediction target are treated as independent figures, and validating a 10 MB
upload could consume a meaningful share of those 60 seconds. Nothing here
settles it. What has changed is that the limits are now in one place with their
provisional status attached, so the benchmark that would settle it has a single
number to measure. The measurement is still owed, and one hour of benchmarking
remains cheaper than finding the conflict during integration.

A second point follows from §2: a single 10 MB figure could never have applied
to both inputs, because it would reject the documented Ausgrid archive. Any
future limit decision has to be per input.

## 6. Dependencies on the rest of the team

Unchanged from B1, with the fallbacks now actually in place rather than planned.

- **Rui — column schema (R1).** Provisional structural rules stand without it.
  The specification is partitioned so column-name rules swap in without
  touching structural logic: nothing in `upload_validation.py` inspects a
  column name.
- **Khang — input contract.** Type strictness is set as broadly as it can be
  defended, per B2, and tightens on confirmation. The deny-list is not subject
  to that trade.
- **Serialisation decision (B6).** Refusing serialised members at upload does
  not decide how the team's own trained models are saved. That decision is
  still open, and the ONNX recommendation in B6 stands.

## 7. Running the evidence

```text
python -m unittest discover -s tests -v
```

60 tests: the 12 existing archive, source-value, input-contract, chronology and
feature tests, plus 48 covering the rejection matrix, the accepted-upload
descriptors, the experiment record, SHAP additivity and the app wiring. All of
the added tests are stdlib-only and run without the model or app extras
installed.

### Reproducing the rejection table

```text
python scripts/make_invalid_uploads.py OUTPUT_DIRECTORY
```

Builds each invalid file from the same matrix the suite uses, runs it through
the live rules, and prints the rejection produced beside the one expected. It
exits non-zero on any mismatch, so it works as a check as well as a
demonstration. It needs no source data.

To show a rejection in the dashboard rather than at the command line, the AEMO
input needs eleven real monthly archives plus one generated file in place of the
twelfth. A single file submitted alone is refused for the count first, which is
the count rule working rather than the file rule failing.

### Exercising the whole path without the real archives

```text
python scripts/make_demo_sources.py demo_sources
python scripts/replay_check.py --from-sources demo_sources --case aemo
python scripts/replay_check.py --from-sources demo_sources --case ausgrid
```

`make_demo_sources.py` writes archives that satisfy the documented input
contract exactly — 12 monthly AEMO ZIPs covering 17,520 consecutive NSW1
half-hours, and an Ausgrid-shaped ZIP with 731 consecutive actual, no-blank days
for customer 1 `GG`. **The readings are invented** and no figure derived from
them is a result; they exist so the path from upload through to the experiment
record can be run at all. Generation is byte-deterministic, which is what makes
the archives usable for a replay check: an identical upload digest across two
runs means the inputs really were identical, so a metric difference could only
come from the pipeline. Because the digests go into every record, a synthetic
run can never be mistaken for a real one.

`replay_check.py` runs a case twice and compares the two records, or compares two
records downloaded from the dashboard. It exits non-zero unless the replay
reproduced exactly.

Observed on this workstream's machine: both cases reproduce their metric digests
exactly across repeated runs. The checker also distinguishes the two failure
modes — a metric nudged by 1e-9 is reported as a reproducibility failure naming
the field, while a changed seed is reported as a different experiment rather than
a failure to reproduce. Cross-machine replay is still outstanding, and the LSTM
has not been put through a replay check.

### A correctness fix outside this workstream

Running the end-to-end path surfaced a pre-existing bug in the XGBoost SHAP
additivity check. It compared a float32 contribution sum against the prediction
with a fixed **absolute** tolerance of 1e-3, and only for the first validation
row. At load magnitudes around 6,500 MW float32 cannot represent a difference
that small: the measured worst error is about 8e-3 absolute, or 1.3e-6 once
scaled by the prediction, and roughly 72% of perfectly well-formed rows exceeded
the threshold. Whether a run failed therefore depended on which row happened to
be first, which is why it passed on the documented AEMO copy and failed here.

The check now scales the error by the prediction, floors the denominator at 1.0
so the near-zero PV case keeps an absolute tolerance, and takes the worst error
across every row rather than the first. The tolerance is 1e-5, about an order of
magnitude above measured float32 noise, and a genuine additivity break is a
whole-prediction error far above it. `tests/test_shap_additivity.py` pins all of
this without training a model.

This is a change to code outside the security workstream and should be reviewed
by whoever owns the model wrapper. It makes the check stricter in substance —
scaled rather than absolute, every row rather than one — not more permissive.

### Confirming the tests would catch a regression

Removing the serialised-member rule or the content-signature check from
`upload_validation.py` fails two tests each. A rule deleted by a later edit does
not pass quietly.

## References

Bieringer, L., Grosse, K., Backes, M., Biggio, B., & Krombholz, K. (2022).
Industrial practitioners' mental models of adversarial machine learning. *SOUPS
2022*, 97–116. https://www.usenix.org/conference/soups2022/presentation/bieringer

ONNX. (n.d.). *ONNX: The open standard for machine learning interoperability.*
https://onnx.ai/

OWASP. (n.d.). *File upload cheat sheet.*
https://cheatsheetseries.owasp.org/cheatsheets/File_Upload_Cheat_Sheet.html

OWASP. (2023). *OWASP machine learning security top ten* (Version 0.3).
https://owasp.org/www-project-machine-learning-security-top-10/

Pineau, J., Vincent-Lamarre, P., Sinha, K., Larivière, V., Beygelzimer, A.,
d'Alché-Buc, F., Fox, E., & Larochelle, H. (2021). Improving reproducibility in
machine learning research. *Journal of Machine Learning Research, 22*(164),
1–20. https://www.jmlr.org/papers/v22/20-303.html
