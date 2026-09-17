# MOTARED — Change Ledger

**Rule:** append one repair group at a time. Never combine unrelated repairs. Never mark a group accepted from unit tests alone.

## Repository safety precondition

At plan creation, the working tree was already materially dirty, including tracked modifications in core inference, backend, evaluation, tests, and dashboard files plus many untracked artifacts. Therefore:

- **No core implementation may begin from the current tree without first identifying and preserving the owner’s existing work.**
- Do not create a misleading checkpoint that silently mixes unrelated changes.
- Before P0-1, inventory each tracked diff, classify it as intended baseline/work-in-progress/noise, and obtain a clean, named baseline by one of:
  1. owner-approved commit of the intended baseline;
  2. owner-approved dedicated worktree/branch from the intended commit with selected changes carried explicitly;
  3. a signed patch bundle plus file hashes when commit is not allowed.
- Never use `git reset --hard`, `git clean`, broad checkout, or destructive stash without explicit owner approval.

## Baseline record

| Field | Value |
|---|---|
| Branch observed | `main` |
| HEAD observed | `712133fd6f9a52edf5292701d61947e32a5b9ea2` |
| Working tree | Dirty; not safe for an automatic checkpoint |
| Production semantic baseline | Job #51 false positive documented in `IMG_5616_ROOT_CAUSE_ANALYSIS.md` |
| Learning mode for job #51 | deterministic, pinned, writes frozen |
| Implementation authorization | Not yet started; mandatory planning gate active |

## Per-repair entry template

Copy this section for each repair group. One entry corresponds to one coherent vertical slice.

---

### CHANGE-[NNN] — [Repair group / concise behavior]

**Root-cause matrix ID:** RC-__  
**Priority:** P0/P1/P2/P3  
**Owner:**  
**Status:** PROPOSED / RED / GREEN-FOCUSED / REGRESSION / REAL-VIDEO / FROZEN-GATE / ACCEPTED / ROLLED-BACK  
**Checkpoint before change:** branch + commit/worktree/patch SHA  
**Rollback target:**  

#### 1. Problem and proof

- First divergence:
- Evidence label/source:
- Why this is general, not video-specific:

#### 2. Preserved invariants

- [ ] Existing true-litter path preserved.
- [ ] Proposal-only detectors remain proposal-only.
- [ ] Actor/object ownership cannot be stolen.
- [ ] Production learning pin unchanged unless this is an approved promotion.
- [ ] Database/API/dashboard compatibility identified.
- [ ] Evidence remains tied to actual event actor/object.

#### 3. Desired behavior

A single observable contract, written before code:

> Given ..., when ..., then ...; and must not ...

#### 4. Files/modules allowed

- Allowed:
- Explicitly out of scope:
- Dependency/caller map:

#### 5. RED — failing test first

- Test file/name:
- Command:
- Expected failure:
- Actual failure:
- Failure artifact/log:
- Confirmation it fails for missing behavior, not test error:

#### 6. Minimal implementation

- Exact change:
- Why smallest safe change:
- No threshold-only/video-specific logic confirmation:

#### 7. GREEN — focused tests

- Command:
- Pass/fail/count/duration:
- Artifact:

#### 8. Regression suite

- Command:
- Baseline known failures:
- New failures:
- Resolution:

#### 9. Real-video validation

| Video | Ground truth/provenance | Before | After | Event time | Actor UID | Object UID | Evidence valid | Repeatability |
|---|---|---|---|---|---|---|---|---|
|  |  |  |  |  |  |  |  |  |

#### 10. Frozen evaluation

| Metric | Baseline | Candidate | Delta | Gate |
|---|---:|---:|---:|---|
| Precision |  |  |  |  |
| Recall |  |  |  |  |
| F1 |  |  |  |  |
| FPR |  |  |  |  |
| Temporal match |  |  |  |  |
| Identity correctness |  |  |  |  |
| Causal timestamp violations |  |  |  | must be 0 |
| Missing/incorrect evidence |  |  |  | must be 0 on confirmed set |
| Repeated-run equality |  |  |  | must be 100% CPU |

#### 11. Risk review

- New false-positive risk:
- New false-negative risk:
- Latency/memory impact:
- Privacy/security impact:
- Historical-data/API compatibility:
- Residual unknowns:

#### 12. Decision

- GO / NO-GO / ROLLBACK
- Approver:
- Reason:
- Next permitted repair group:

---

## Ledger entries

### CHANGE-000 — Pre-implementation forensic planning

**Root-cause matrix IDs:** RC-01 through RC-14  
**Status:** ACCEPTED AS PLANNING ARTIFACT ONLY  
**What changed:** Created/updated planning, validation, matrix, and ledger documents.  
**What did not change:** inference, FSM, identity, thresholds, models, database, API, and dashboard behavior.  
**Evidence:** `IMG_5616_ROOT_CAUSE_ANALYSIS.md`; `FINAL_REPAIR_MASTER_PLAN.md`; `FINAL_VALIDATION_PROTOCOL.md`; `ROOT_CAUSE_MATRIX.md`.  
**Implementation result:** none.  
**Safety finding:** repository is not currently clean enough for a safe core checkpoint; baseline isolation is the first execution prerequisite.
