# EXTFP: analysis code, protocol copies and aggregate results

External false-positive evaluation of panoramic periapical-lesion detectors.

## Contents

- `protocols/`: edited English copies of the scientific specifications.
- `analysis/`: analysis code with configurable local input files.
- `aggregate_results/`: cohort-level and training-run-level results, FROC tables and
  validation threshold-search tables.
- `PROVENANCE.json` and `SHA256SUMS`: specification dates and file identities.
- `PROTOCOL_MAP.md`: section coverage, editing scope and interpretation of dates.

The original comparison was specified before external inference. The replication
was specified after the original result was known and before its new runs.
The recall-matched and partial-area analyses are post hoc; their specifications
disclose prior inspection of results. Local specification dates are not public
registration dates. These edited English copies are not byte-identical originals. See `PROTOCOL_MAP.md`
for source-section coverage and limitations of the accompanying provenance.

## Data sources

Obtain data directly from the providers and follow their terms:

- Development: https://doi.org/10.17632/kx52tk2ddj.3
- External: https://doi.org/10.5281/zenodo.15487430
- External dataset paper: https://doi.org/10.3390/diagnostics14202336

Aggregate JSON files retain the reported numerical results; their metadata and
package-relative references are adapted for this copy. Paths beginning with
`protocols/` or `aggregate_results/` resolve from this package root. Their exact
public byte identities are recorded in `SHA256SUMS`.
Hash fields inside these JSON files (`prespec_sha256_crlf` in `secondary_75.json`,
`amendment_sha256` in `posthoc_v3.json`, `amendment_sha256_lf` in `posthoc_v4.json`) identify
the frozen original specifications that the executed scripts read. They equal `original_sha256`
in `PROVENANCE.json`, not the hashes of the edited copies at the adjacent `protocols/` paths.
Names containing "Fig 2" (`s5_fig2_ci` in `posthoc_v4.json`; the comment and the `V8 Fig 2` log
line in `analysis/extfp_posthoc_v3.py`) refer to the per-arm false-positive figure of
`protocols/EXTFP_AMENDMENT_v4_POSTHOC.md` §5 and its acceptance criterion V8 (§9); this package
contains no figures.
In `aggregate_results/pairwise_7274.json`, the text fields `s7_4_conclusion` and
`confounded_pairs_note`, and the matching strings in `analysis/extfp_pairwise_7274.py`,
are English translations of the executed wording (P1 as in `protocols/EXTFP_PRESPEC.md`
§7.4); no numerical field changed.
The scripts in `analysis/` are edited copies of the executed code. Descriptive headers,
comments, some messages and provenance metadata fields were edited, and internal record
pointers, git calls and local paths were replaced by configuration; numerical logic and
constants are unchanged. Hashes of these scripts, and the module hashes they record,
therefore differ from those of the original execution.
`aggregate_results/tau_selection/` (original comparison, three arms) and
`aggregate_results/tau_selection_v2/` (replication, nine runs) contain
`val_f1_curve_<run>.csv`, written by `extfp_tau_selection.py` and
`extfp_tau_selection_v2.py`. Each row gives the true-positive, false-positive and
false-negative counts, precision, recall and F1 on the 379 internal validation images
at one τ from 0.001 to 0.999. Each run's selected τ is the F1 maximum of its table.
These CSV files are the executed outputs, unchanged (CRLF line endings).

No radiographs, annotations, image-level predictions, image identifiers, model
weights or bootstrap replicate records are included in this package.

## Configure local inputs

1. Choose a local working directory. The default, `local_work/` inside this package,
   is git-ignored; set `EXTFP_DATA_ROOT` to use a directory elsewhere, or edit
   `DEFAULT_DATA_ROOT` in `analysis/public_inputs.py`.
2. Copy `inputs.example.json` to `inputs.local.json`. Set each reference path and
   its independently recorded SHA-256. Alternatively set `EXTFP_INPUT_MAP`.
3. Supply your input artifacts in the `runs/extfp/` layout used by the scripts.
   Each script shows its input fields at the corresponding JSON/JSONL reads.
4. Review the constants at the top of each script for your input configuration:
   cohort sizes, input identities, model records and reference-file hashes.
   For a new study, specify these choices before running analyses. Do not change
   a scientific rule simply to make a failed input check pass.
5. Set `EXTFP_RUN_ID` to identify the local execution. Outputs are written beneath
   the configured data root. Do not redistribute local inputs or outputs with this package.

### Input schemas

Reference threshold records contain `arms`, each with `arm` and `tau`.
The secondary analyses additionally use validation prediction paths and records.
External roster rows describe image membership, dimensions and local label paths;
prediction rows provide ordered scores and `boxes_xyxy` for the same image keys.
Inference metadata, completion checks, input checksums and threshold records must
agree. Each driver checks the relationships named in its code before computing endpoints; the
primary-endpoint driver relies on a prior `G6 PASS` gate log for them (see Execution prerequisites).
The edited protocol copies are reading documents, not substitutes for an
execution's separately recorded specification input files and their hashes.

### Execution prerequisites

These scripts retain study-specific identity checks. The public package alone
does not contain all inputs needed to execute the original study.

| Driver | Local prerequisites / configuration |
|---|---|
| Primary endpoint | `runs/extfp/external/gate_g6.log` containing `G6 PASS`, cohort roster, predictions, metadata and threshold record. |
| Secondary and post-hoc | Original execution specification files (not included in this package) under `specifications/` in the data root, matching `PRESPEC_SHA_CRLF`, `V3_SHA` and `V4_SHA_LF`; raw prediction checksum manifest and completed earlier endpoint outputs. |
| Replication threshold selection | `FROZEN_DOCS` files and hashes (including the training configuration manifest), `RUNS` weight identities and paths, validation images/labels and `EXPECTED_GT_BOXES`. |
| All drivers | Inspect path constants and JSON/JSONL reads in the selected driver; configure the four reference snapshots in `inputs.local.json`. |

For another study, record its specification, input identities and validation
procedure before editing these configuration constants. Preserve the endpoint
definitions and distinguish any methodological change explicitly. A gate log
alone is not validation: retain evidence for cohort membership, label validity,
prediction completeness, threshold provenance and matching input checksums.
If the required original files or validation evidence are unavailable, stop;
the edited reading copies cannot satisfy the original byte-identity checks.

### Analysis order

Given the required local inputs: `extfp_primary_endpoint.py`,
`extfp_pairwise_7274.py`, `extfp_secondary_75.py`, `extfp_endpoints_v2.py`, then
`extfp_posthoc_v3.py`. Threshold selection uses `extfp_tau_selection.py` and
`extfp_tau_selection_v2.py`; these can run model inference and require model files.
`extfp_external_cohort.py` constructs the cohort from local labels.

Example (after configuration):

```text
python analysis/extfp_primary_endpoint.py
```

NumPy is required for analysis; the Welch calculation additionally requires SciPy.
Threshold prediction requires compatible PyTorch and Ultralytics installations.
The recorded post-hoc environment used Python 3.12.9, NumPy 2.4.6 and SciPy 1.15.3.
Package validation covers configuration, input checks and selected numerical
functions; the full data-dependent pipeline has not been rerun for this copy.

## Licence

- Code (`analysis/`, `inputs.example.json`): GNU Affero General Public License
  v3.0 only (`AGPL-3.0-only`). Full text: `LICENSE`.
- Protocol copies, documentation and aggregate results (`protocols/`,
  `aggregate_results/`, `README.md`, `PROTOCOL_MAP.md`, `PROVENANCE.json`):
  Creative Commons Attribution 4.0 International (`CC-BY-4.0`). Full text:
  `LICENSE-CC-BY`. For attribution, cite the associated article.
- These licences cover this package only. The development and external datasets
  remain under their providers' terms (see Data sources).
