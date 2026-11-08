# Protocol coverage and provenance

These are edited scientific reading copies of locally recorded specifications.
They are not the complete frozen originals. The v2.1 clarification is in its original
English; the other five copies are English translations of the edited extracts.
Original bytes remain preserved by
the authors; `PROVENANCE.json` distinguishes original raw-byte hashes, original
LF-normalized hashes and the hashes of these public copies. `SHA256SUMS` identifies
every file in this package. No private repository access is required to read it.

## Section coverage

Section numbers refer to the original version named in each row. “Extract” means
scientific passages were retained with editorial edits; it does not mean every
sentence of that section is reproduced. “Summary” means newly worded exposition
of the existing record, not a new specification or amendment.
Passages on manuscript preparation and correspondence were omitted throughout, and hardware measurements and run incidents were condensed; they defined no analysis, threshold, sample or decision rule.

| Source document | Original sections | Location and treatment in this copy |
|---|---|---|
| `EXTFP_PRESPEC_v1.0.md` | §0 | Prior viewing summarized in Specification timing; detailed earlier-study measurements omitted to keep this record specific to the present evaluation |
| same | §1 | Scientific question and estimand summarized in §1; original attention-ladder hypothesis identified as superseded before external inference |
| same | §§2–10, §14 | Scientific extracts under the same section numbers |
| same | §§11–13 | Non-method administrative material omitted; specification date and version relationship recorded separately |
| `EXTFP_PRESPEC.md` (v1.1) | §0 | Prior viewing and execution stage summarized in Specification timing |
| same | §1 | Question, estimand and interpretation limits summarized in §1 |
| same | §§2–10, §14 | Scientific extracts under the same section numbers |
| same | §§11–13 | Non-method administrative material omitted |
| `EXTFP_PRESPEC_v2_EXTENSION.md` | §V0 | Prior results and probe exposure summarized in Specification timing; detailed exploratory and administrative records omitted |
| same | §§V1–V9 | Scientific extracts under the same section numbers; in §V1 the original's longer justification for not using k-fold cross-validation is omitted, keeping only the design statement |
| same | §V10 | Freeze sequence and execution status summarized in §V10 |
| same | §§V11–V12 | Author deliberation and manuscript integration plan omitted; final seed/run choices remain in §V2 and the v2.1 clarification |
| same | §V13 | Non-binding resource estimate omitted |
| `EXTFP_PRESPEC_v2.1_CLARIFICATION.md` | §§1–5 and status header | Timing and clarification summarized, including nine-run matrix, unused six-run fallback and unchanged rules |
| `EXTFP_AMENDMENT_v3_POSTHOC_DRAFT.md` | §§0–1 | Rationale and prior-result exposure summarized in Specification timing and §2; exploratory numerical listings omitted |
| same | §§2–6 | Scientific extracts under the same section numbers, including S1 and acceptance criteria; §4.3 is a scientific summary of the exclusions with a cross-reference to the later v4 amendment |
| same | §7 | Omitted (manuscript planning) |
| `EXTFP_AMENDMENT_v4_POSTHOC.md` | Status header and §1 | Prior-result exposure summarized in Specification timing |
| same | §§2–7, §§9–10 | Scientific extracts under the same section numbers, including partial area, seed-mean uncertainty, image-level measures, example selection and acceptance criteria |
| same | §8 | Omitted (manuscript planning) |

Editorial changes remove administrative pointers, replace unavailable internal
references with self-contained explanations, and direct dataset readers to the
original providers. Version-specific numerical methods and outcome-dependent
limitations are preserved. Historical differences are read chronologically:
v1.1 supersedes v1.0; v2.1 clarifies v2 run counts; v4 amends the specified parts
of v3. The `DRAFT` suffix in the v3 source filename is retained for identification;
its final frozen status is recorded in the local source record.
Passages marked *Reading-copy note*, and parenthetical remarks giving what "the original
text reads" or "cites", are editorial annotations added to these copies on 2026-11-07.
They correct an arithmetic expression, a count or a cross-reference (stating the original
wording), define a term, or mark a planning assumption that the revised manuscript
qualifies; they change no rule.

## Arm 0 training record

Arm 0 (YOLOv8m) was trained twice. The first run (2026-08-30) is the training run that preceded
version 1.1 (see its Specification timing); it was trained before the fail-closed execution guards
and the protocol manifest of v1.1 §5 existed, so its provenance lacked three records that gate G2
requires, although none of its recorded items departed from the protocol. A single replacement run
was specified on 2026-08-31, before any external inference, threshold selection or endpoint
calculation, to use the frozen runner and manifest, with its output fixed in advance as the arm-0
model regardless of performance. It supplied the missing records and reproduced the first run's
internal-validation precision, recall, mAP50 and mAP50-95 to five decimal places. Only the
replacement run enters the analyses; the first run is preserved but not used.

## Fixed-recall threshold rule

Version 1.1 §6.3 item 4 fixed internal-validation recall at 0.50 and 0.70 but did not state
which grid τ is used when several reach the target. The v1.1 secondary analysis used the
largest such τ. The executed code comment and the `rule` field of
`aggregate_results/secondary_75.json` call this the "6.2 tie-break", although §6.2 governs
ties at maximum F1. The v2 specification (§V5) records the rule as a post hoc
operationalisation in version 1 and freezes it for the replication. Reported values are
unchanged.

## Timing and evidential limits

The original comparison was specified before external inference. Replication was
specified after the original results were known but before its new main training
runs and external inference; its feasibility probe had already been inspected.
The v2.1 clarification followed training and internal threshold selection but
preceded external inference. The v3/v4 analyses were specified after exploratory
inspection of relevant results and are disclosed as post hoc.

The dates in `PROVENANCE.json` report local specification dates. A hash identifies
bytes; it does not independently prove when those bytes existed. Neither the
dates nor these edited copies establish public preregistration. The repository
release has its own publication date.

The map documents the scope of these copies. It does not assert that edited
extracts alone are equivalent to the complete frozen originals.
