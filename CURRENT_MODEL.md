# Current Recognizer Model

Updated: 2026-07-16

## Default Runtime Model

- Default model version: `v2_2_candidate`
- Default pointer: `recognizer/models/default_model.json`
- GUI display name: `V2.2（H3闭卷通过）`
- Architecture: V2.1 parent model + leanback two-stage fine classifier + multi-prototype matching + Boundary safety fallback.

Default launch:

```bash
python3 posture_csv_app.py
```

## Available Model Versions

### V1

- Model: `recognizer/models/rf_posture_v1.joblib`
- Metadata: `recognizer/models/rf_posture_v1.metadata.json`
- Prototype bank: `recognizer/models/prototype_bank_v1.json`
- Model SHA256: `fc899b62f1649d7febccd6922eb8c82b214912893e0f47fbea034f1bfd58bcaa`

Run explicitly:

```bash
python3 posture_csv_app.py --model-version v1
```

### V2 Candidate

- Model: `recognizer/models/rf_posture_v2_candidate.joblib`
- Metadata: `recognizer/models/rf_posture_v2_candidate.metadata.json`
- Prototype bank: `recognizer/models/prototype_bank_v2_candidate.json`
- Runtime config: `recognizer/models/rf_posture_v2_candidate.runtime_config.json`
- Model SHA256: `4efbfd4c833301384994c4c30b0c90b96797ebb65a35ea488a962523437acda4`

Run explicitly:

```bash
python3 posture_csv_app.py --model-version v2_candidate
```

### V2.1 Candidate

- Model: `recognizer/models/rf_posture_v2_1_candidate.joblib`
- Metadata: `recognizer/models/rf_posture_v2_1_candidate.metadata.json`
- Prototype bank: `recognizer/models/prototype_bank_v2_1_candidate.json`
- Runtime config: `recognizer/models/rf_posture_v2_1_candidate.runtime_config.json`
- Model SHA256: `7a30cdd3951a71400fc5c442124702233cdd7d6823ec76c2a681858f59651c39`

Run explicitly:

```bash
python3 posture_csv_app.py --model-version v2_1_candidate
```

### V2.2 Default

V2.2 keeps the V2.1 parent model unchanged, then runs a second-stage leanback
fine classifier only when the parent result is in the后靠 family. It was
promoted to the default runtime model after passing the H3 external holdout.

- Parent model: `recognizer/models/rf_posture_v2_1_candidate.joblib`
- Leanback submodel: `recognizer/models/leanback_subclassifier_v2_2_candidate.joblib`
- Leanback metadata: `recognizer/models/leanback_subclassifier_v2_2_candidate.metadata.json`
- Leanback prototype bank: `recognizer/models/leanback_prototype_bank_v2_2_candidate.json`
- Leanback runtime config: `recognizer/models/leanback_subclassifier_v2_2_candidate.runtime_config.json`
- Model bundle: `recognizer/models/v2_2_candidate.model_bundle.json`
- Submodel SHA256: `829e8204ca8c5b348de2310b209bba54a7b37c22c859336ccfcf3ef0fc73c75c`
- Bundle SHA256: `1f244b88db46840c3fad60f70148731c3c8193b60316de848e4282c75f9bdaf4`

Run explicitly:

```bash
python3 posture_csv_app.py --model-version v2_2_candidate
```

Development validation summary:

- LOFO H1/H2: 8/8 correct fine accept, wrong accept 0
- H1→H2: 4/4 correct fine accept, wrong accept 0
- H2→H1: 2/4 correct fine accept, 2 conservative fallback to `后靠坐姿`, wrong accept 0
- Object/empty stage2 triggers: 0

H1/H2 are development validation data and must not be counted as closed-book
holdout. H3 is the consumed promotion holdout for this frozen V2.2 artifact set.

H3 promotion evidence:

- correct_accept: 3/4
- correct_fallback: 1/4
- wrong_accept: 0/4
- gate_miss: 0/4
- safe_resolution: 4/4
- H3 is consumed external holdout data. It must not enter training, tuning,
  prototype construction, sample weighting, threshold selection, or any future
  unopened holdout claim.

## Holdout Batch 02 Metrics

Closed-book batch: `posture_dataset_v2/reports/external_holdout/holdout_batch_02/`

| Model | Boundary-aware file accuracy |
|---|---:|
| V1 | 16.67% |
| V2 candidate | 75.00% |
| V2.1 candidate | 83.33% |

V2.1 per-class file results:

- 端正坐姿: 2/3
- 前倾端坐: 3/3
- 标准靠背坐: 2/2
- 交叉腿靠背坐: 1/2
- 盘腿坐: 2/2
- wrong accepted files: 0
- object pressure entering posture model: 0

## Rollback

No model artifact was overwritten. To roll back at runtime, pass an explicit
model version:

```bash
python3 posture_csv_app.py --model-version v1
python3 posture_csv_app.py --model-version v2_candidate
python3 posture_csv_app.py --model-version v2_1_candidate
```

To change the default pointer later, edit only
`recognizer/models/default_model.json`; do not copy, rename, or overwrite
`.joblib`, metadata, prototype bank, or runtime config artifacts.

## Current Limitation

V2.2 has passed the H3 leanback fine-class external holdout, but any future
change to the V2.2 gate, boundary, prototypes, or classifier requires a new
unseen H4 or later holdout batch. H1/H2 are development data; H3 is consumed
for the promotion decision and cannot be reused as a future closed-book test.
