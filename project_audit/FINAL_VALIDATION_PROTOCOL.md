# MOTARED — Final Validation Protocol

## 1. Purpose

Define the non-negotiable evidence required after every repair and before any readiness claim. This protocol validates **illegal littering semantics**, not merely detector output or test completion.

## 2. Ground-truth governance

Every clip must have a versioned manifest entry:

```yaml
clip_id: stable logical ID
file_sha256: required
filename: informational only
source: path/location
label: LITTER | VALID_DISPOSAL | TEMP_GROUND_RECOVERED | CARRY_ONLY | PREEXISTING | NO_EVENT | UNRESOLVED
actor_annotation: validated actor or null
object_annotation: validated physical object or null
event_intervals:
  carry: [start, end]
  release: [start, end]
  ground: [start, end]
  recovery: [start, end]
  disposal: [start, end]
  abandonment: [start, end]
reviewer: human identity/role
review_status: single_review | double_review | adjudicated
provenance: operator | frozen_set | forensic_review
manifest_version: immutable version
```

Rules:

1. Filename is never a feature or rule key.
2. A system confirmation/pseudo-label is not ground truth.
3. Ambiguous clips remain `UNRESOLVED` and cannot tune/promote behavior.
4. Label changes require a new manifest version and reason.
5. Train/tune clips and frozen holdout clips are separated by physical source/provenance, not random frames from the same video.

## 3. Mandatory validation layers

### L0 — Static safety

- Syntax/import checks.
- No filename/timestamp/actor/object-ID special cases.
- No accidental config/threshold/model/pin change.
- Dependency graph and public contracts reviewed.
- Causal event schema validator enabled in tests.

### L1 — Unit TDD

For each vertical repair slice:

1. Write one behavior test.
2. Run it and observe the expected failure.
3. Implement minimal change.
4. Run focused test to green.
5. Add next edge case only after green.

Tests must use actual implementation objects; mock only external I/O.

### L2 — Integration

- Detection/tracks→stable identity→ownership→FSM→event.
- Event→evidence→DB→API serialization.
- Candidate/recovered/valid-disposal/final-violation lifecycle.
- Upload and CLI file-mode equivalence.

### L3 — Existing regression suite

Record exact command, pass/fail totals, known pre-existing failures, and new failures. Any new unexplained failure is a stop condition.

### L4 — Targeted real-video subset

Run only clips relevant to the repair first, sequentially on CPU where needed. Preserve before/after reports, hashes, pin hash, source-CFR hash, code commit, environment, and hardware mode.

### L5 — Full frozen semantic suite

Run all categories and compute metrics. No cherry-picking.

### L6 — Reproducibility

Same immutable inputs three times:

- event count and final outcomes identical;
- actor/object identities equivalent;
- event timestamps within defined decoder tolerance;
- evidence availability identical;
- pin/model/config/code hashes identical;
- live `learning.json` hash unchanged.

### L7 — Dashboard acceptance

Verify API and rendered UI distinguish final violation, valid disposal, recovered, candidate, and rejected states; missing artifacts are not called verified.

## 4. Frozen test matrix

| Category | Minimum required scenario | Expected final outcome | Primary metric |
|---|---|---|---|
| True litter | carry→release→ground→abandon/depart | VIOLATION | recall + temporal match |
| Valid bin disposal | direct carry→container | VALID_DISPOSAL/no violation | FPR=0 for validated cases |
| Temporary ground + recovery | set down→same-object pickup→continue | RECOVERED/no violation | FPR=0 |
| Recovery then bin | set down→recover→container | VALID_DISPOSAL/no violation | FPR=0 |
| Carry only | no release | no violation | FPR=0 |
| Pre-existing litter | passer-by near old object | no violation | ownership FP=0 |
| Multi-person | one handler, bystanders | only handler may be actor | actor accuracy |
| Multiple objects | distinct objects same/other actors | preserve physical episode identities | object-ID accuracy |
| Tracking interruption | short occlusion/rebirth | conservative continuity | merge/split rate |
| Frame edge | exit/re-entry carrying object | no UID theft/false release | identity + release accuracy |
| Low light | validated positive and negative | category-correct | P/R/FPR |
| Small object | validated litter and carry-only | category-correct | detector recall then event recall |
| Black/red/yellow/transparent | varied appearance | semantics independent of color | per-category metrics |
| Dumpster clutter | static pile + passer-by | no violation | FPR |
| Near-bin true litter | object abandoned beside container | VIOLATION | recall |
| Overflowing container | valid deposit vs adjacent abandonment | correct distinction | semantic accuracy |
| Duplicate incident | identity churn/tier duplication | one event | incidents/event ratio |
| End-of-clip unresolved | episode cuts before truth | UNRESOLVED, not accusation | unsafe-finalization count |

## 5. Required named real-video set

### Operator/frozen truth available

| Video | Required truth | Use |
|---|---|---|
| IMG_5118 | LITTER, multiperson selective (frozen) | ownership + true positive |
| IMG_5117 | LITTER (frozen) | short ground positive |
| IMG_5290 | LITTER near dumpster (frozen) | do not over-suppress bin-adjacent litter |
| IMG_5295 | LITTER bottle (frozen) | object class/generalization |
| IMG_5305 | NO_EVENT/valid bin-related hard negative (frozen) | disposal FP guard |
| IMG_5306 | exactly one true ground litter; other bin actions non-violations (frozen) | long-episode/dedup |
| IMG_5115 | carry-only negative (frozen) | release FP guard |
| IMG_5303 | pedestrian/no-waste negative (frozen) | clutter/ownership FP guard |
| IMG_5611 | non-violation (operator statement; must be formalized in manifest) | valid disposal |
| IMG_5613 | non-violation (operator statement; must be formalized in manifest) | pre-existing/proximity attribution |
| IMG_5616 | non-violation (operator statement + forensic report) | temporary ground→recovery→disposal |

Before using IMG_5611/5613/5616 for promotion, an operator must sign the exact semantic intervals and file hashes.

## 6. Metrics

### Event semantics

- Clip-level precision, recall, F1.
- Event-level temporal precision/recall using frozen protocol Δt/Tmax.
- False-positive rate overall and per hard-negative class.
- False-negative rate overall and per positive class.
- Outcome confusion matrix: violation / valid disposal / recovered / no-event / unresolved.

### Identity and ownership

- Physical-object ID precision: fraction of stable UID links that join the same annotated object.
- Physical-object ID recall: fraction of annotated continuity preserved.
- Person ID split and false-merge rates.
- Actor attribution accuracy.
- Object attribution accuracy.
- UID transfer into simultaneously visible different object: **must be zero**.

### Lifecycle

- Release timing error.
- Ground timing error.
- Decision latency after semantic truth becomes observable.
- Causal-order violations: **must be zero**.
- Premature-finalization count.
- Recovery cancellation accuracy.
- Duplicate events per physical incident.

### Evidence

- Correct actor crop.
- Correct object crop.
- Carry/release/ground/recovery/disposal frames correspond to truth.
- Missing/corrupt artifact rate.
- “Verified” claims with missing artifact: **must be zero**.

### Reproducibility

- Three-run outcome equality.
- Event identity equivalence (excluding random UUID).
- Pin/model/config/code/source hashes.
- Learning-store mutation count during deterministic validation: **zero**.

## 7. Gates after each repair

A repair is rejected if any applies:

1. Focused test was not observed failing before code.
2. Any new unexplained unit/integration failure.
3. Targeted failure remains.
4. Previously passing safety scenario regresses.
5. Causal timestamp violation appears.
6. Evidence points to wrong actor/object.
7. Deterministic runs differ.
8. Production pin/model/config changes without explicit promotion.
9. Improvement depends on filename/color/timestamp special casing.
10. Real-video results are absent.

## 8. Final readiness gate

No fixed numeric precision/recall target may be invented during implementation. Before final promotion, the product owner must approve numeric thresholds based on deployment risk. Regardless of those thresholds, the following are absolute:

- zero known causal-order violations;
- zero UID transfers between simultaneously distinct physical objects in annotated validation;
- zero accusations for validated temporary-ground/recovery and valid-disposal cases;
- zero passer-by attribution on validated pre-existing-litter cases;
- zero duplicate events for annotated single incidents;
- 100% three-run reproducibility on CPU validation path;
- immutable pin/model/config provenance;
- all confirmed evidence linked to correct actor/object;
- no untested P0 root-cause class.

Only after the owner defines and the candidate meets the numeric P/R/F1/FPR gates may the status become `FUNCTIONALLY READY`. This is not equivalent to production deployment approval.

## 9. Required artifacts per run

```text
validation_runs/<candidate-id>/
  environment.json
  hashes.json
  git_status.txt
  command.txt
  stdout.log
  per_clip.json
  aggregate_metrics.json
  identity_metrics.json
  lifecycle_metrics.json
  evidence_audit.json
  reproducibility.json
  dashboard_checklist.md
  promotion_decision.md
```

Artifacts must be append-only/versioned. Never overwrite the baseline report with candidate results.
