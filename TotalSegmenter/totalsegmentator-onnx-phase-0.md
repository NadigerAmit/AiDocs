### Rust Execution And Synthetic Agreement (2026-10-10)

The exported Dataset291 organ graph was executed in Rust, not only in Python. One zero-valued float32 patch with shape `[1,1,128,128,128]` was used; no CT or MRI data was accessed.

| Comparison | Result |
|---|---|
| PyTorch 2.5.1+cpu vs Python ONNX Runtime 1.20.1 | Maximum and mean absolute logit error: `0.0`; argmax agreement: `1.0` |
| Python ONNX Runtime vs Rust `ort` using ONNX Runtime 1.22.0 | Maximum and mean absolute logit error: `0.0` |
| Output | Finite float32 logits, shape `[1,25,128,128,128]` |
| One-run timing | PyTorch approximately 5.91 s; Python ORT approximately 5.92 s; Rust inference plus comparison approximately 6.84 s |

This establishes execution and exact agreement on **one synthetic zero patch only**. A zero patch is a degenerate input and is not representative of CT anatomy or intensity distributions. It does not establish real-input conversion correctness, segmentation accuracy, preprocessing parity, or clinical validity. The cleaned CT fixtures still require pixel review and suitability approval before the real-input tests.

The opt-in test verifies the graph/checkpoint/plan/label identity against the export sidecar, constructs a zero tensor directly in Rust, and compares against a private Python output file. It does not fabricate medical-fixture approvals or publish a passing Phase 0 report.

Re-run it after the private Python reference `synthetic/python-output.npy` has been produced in the existing experiment directory:

```bash
PHASE0_SYNTHETIC_RUN="$HOME/.local/share/radassist-model-factory/runs/totalseg-organ-export-20261010-r1" \
rtk proxy cargo test \
  --manifest-path tools/model-factory/totalseg-onnx/ort-smoke/Cargo.toml \
  --locked exported_organ_graph_synthetic_python_rust_parity \
  -- --ignored --nocapture
```

The test is ignored in the normal suite because the private model and reference are not checked into Git; it was explicitly run and passed. The normal Rust suite also passed seven tests after formatting. The synthetic agreement target (`1e-3`) is a diagnostic assertion, not a replacement for reviewer-approved real-input thresholds.
# TotalSegmentator ONNX Phase 0 Runbook

**Status:** The initial real-input PyTorch-vs-ONNX tensor-numeric gate did not meet its approved threshold. A later app-vs-reference follow-up passed full-volume segmentation parity after correcting the app-side preprocessing/inference pipeline, using the same pinned ONNX graph. These are distinct gates: the full-volume pass does not retroactively pass the tensor-numeric gate or prove conversion equivalence for every possible input. See the dated follow-up below.
**Owner:** AI/model-factory workstream.
**Purpose:** Prove that one pinned TotalSegmentator CT organ network can be exported to ONNX and run with numerically equivalent outputs before building application features around it.
**Related design:** [AI model integration architecture](../AI-MODEL-INTEGRATION-ARCHITECTURE.md), especially the model-factory and Phase 0 sections.
**Application full-range inference:** [Chunking, memory budgets, lifecycle, and validation status](../../radassist-multiplatform/docs/TOTALSEGMENTATOR-CHUNKED-INFERENCE.md).
**Engineer training:** [ONNX and Rust Engineering Guide](onnx-and-rust-engineering-guide.md) explains the model-format concepts, current pinned environment, Rust runtime boundary, commands, and evidence limitations.

## 1. What Phase 0 proves

The first goal is deliberately narrow: select the TotalSegmentator model component that predicts abdominal organs (often referred to as `part_organs` in the pinned release), export its neural-network forward pass, and run the exported graph with ONNX Runtime.

This is a technical feasibility experiment, not a clinical validation, product feature, or approval to distribute weights. It answers:

- Can the exact pinned PyTorch checkpoint be loaded and exported reproducibly?
- Can ONNX Runtime load the graph and execute its fixed-size 3D patch input?
- Do the PyTorch and ONNX outputs agree within pre-agreed numeric tolerances?
- Can the same case be processed by the reference pipeline and an ONNX-backed pipeline without orientation, label, or preprocessing drift?
- What are the model size, runtime, and peak-memory costs on target devices?

An ONNX file is only the neural-network graph and parameters. It does **not** include DICOM decoding, orientation conversion, resampling, normalization, sliding-window inference, overlap blending, softmax/argmax, label mapping, or conversion back to the original image grid. Those surrounding operations are part of the medical-imaging inference pipeline and need their own parity checks.

## 2. Scope and non-goals

**In scope**

- One CT organ model component and one pinned checkpoint/fold configuration.
- Linux CPU as the first reproducible development target; GPU may be measured but is not required to prove the baseline.
- PyTorch reference output versus ONNX Runtime output on identical, already-preprocessed 3D patches.
- A Rust/ONNX Runtime smoke test after the Python ONNX test passes.
- A small, approved, de-identified CT fixture set for an end-to-end pipeline comparison.
- Evidence capture for versions, hashes, tolerances, outputs, memory, and timing.

**Not in scope**

- Android application integration, a background service, UI, model download UX, or model registry implementation.
- MedGemma, MedSAM2, MedSigLIP, quantization, model fine-tuning, or custom training.
- Shipping Python, PyTorch, nnU-Net, or TotalSegmentator in RadAssist.
- Clinical performance claims, autonomous findings, or report insertion.
- Redistributing a checkpoint or ONNX bundle before its license and redistribution terms have been reviewed.

## 3. Important version rule: discover, do not assume

TotalSegmentator and nnU-Net model layouts, task IDs, checkpoint paths, class maps, and inference APIs can change. Do **not** copy a task ID, `Dataset291_*` path, fold number, checkpoint filename, or label index from an old note and assume it still identifies the desired model.

For the exact TotalSegmentator release selected by the team:

1. Record the immutable source tag/commit and package version.
2. Use that release's `totalseg_info` / task and class listing and inspect its model configuration.
3. Confirm that the checkpoint is the organ component you intend to test, and record its hash, plans/configuration, trainer, fold(s), and label metadata.
4. Confirm the applicable license for the software, checkpoint, and intended product/distribution separately.

The upstream project currently documents `total` as 117 CT classes and `--fast` as a lower-resolution 3 mm option, but these statements alone do not pin a checkpoint. Treat the chosen tagged release and downloaded weights as the source of truth. In particular, do not assume that selecting an organ pack in RadAssist implies that only those output channels were run: first establish the model's actual output and filter classes only after its normal prediction operation.

## 4. How the experiment is structured

Keep the factory environment and large assets outside the product build and outside Git. A suggested local layout is:

```text
secure-model-factory/                 # ignored/local location, not the app
  source/                              # pinned exporter and parity harness
  env/                                 # environment lock and package inventory
  checkpoints/                         # upstream weights; access controlled
  fixtures/                            # approved de-identified cases only
  runs/<experiment-id>/
    manifest.json
    reference/
    onnx/
    parity/
    benchmarks/
```

Use a separate immutable experiment ID for each combination of source release, checkpoint, exporter, ONNX opset, and runtime. Do not overwrite an earlier result when changing any of these.

### Evidence stages

| Stage | Comparison | What a pass proves |
|---|---|---|
| A. Source checkpoint | Pinned loader loads exact checkpoint and reports expected plan, patch size, channels, and labels | The experiment is using the intended model, not a similarly named or silently substituted one |
| B. ONNX structure | ONNX checker plus graph metadata inspection | The file is a structurally valid graph with the expected tensor interface |
| C. Tensor parity | PyTorch network versus ONNX Runtime on identical preprocessed patches | The exported forward pass is numerically equivalent within the agreed tolerance |
| D. Volume-pipeline parity | ONNX-backed preprocessing/inference/postprocessing versus pinned TotalSegmentator reference on the same volume | Surrounding imaging operations and label geometry have not drifted |
| E. Runtime viability | Repeated runs, resource measurements, then Rust and platform smoke tests | The graph can be hosted in the intended runtime and the device profile is viable |

Do not treat a pass at one stage as a pass at a later stage. In particular, `onnx.checker` does not establish medical-image correctness.

## 5. Step-by-step procedure

### Step 0: Name the experiment and set its gates

Before installing packages or downloading weights, create an experiment record containing:

- Experiment ID and date.
- Intended use: Phase 0 engineering feasibility only.
- Host OS/CPU/GPU, available RAM, driver/runtime versions, and whether measurements are cold or warm.
- Candidate TotalSegmentator release, nnU-Net version, PyTorch version, ONNX opset, ONNX Runtime version, and Rust `ort` crate version (when selected).
- Checkpoint source, model/task/component identity, SHA-256, plan/configuration file hashes, trainer and fold(s), and license review status.
- Fixture IDs that contain no patient name, MRN, birth date, accession number, or other direct identifiers.
- Numeric and end-to-end acceptance tolerances, agreed **before** examining test results.

If a required version or checkpoint identity is unknown, mark it `TBD` and resolve it before claiming reproducibility. Do not invent version pins in the report after the run.

### Step 1: Establish a trusted reference run

Use one approved de-identified CT NIfTI fixture for the first pass. Preserve its affine, spacing, origin, orientation, intensity values, and source hash. Keep the original DICOM-to-NIfTI conversion provenance if the fixture came from DICOM. Do not reorient, normalize, clip, or resample the fixture manually before the reference run.

Run the pinned TotalSegmentator release using the documented command-line interface. Confirm exact flags with that release's `TotalSegmentator --help` before running; the command below is a shape example, not a substitute for checking the pinned CLI:

```bash
TotalSegmentator \
  -i "$FIXTURE_NIFTI" \
  -o "$RUN_DIR/reference/total.nii.gz" \
  -ta total \
  --device cpu \
  --ml \
  --report "$RUN_DIR/reference/run-report.json"
```

The full `total` task can invoke multiple model components and may be expensive. It is useful as a reference artifact, but it is not required to test a single ONNX patch. If the release supports a documented way to run only the desired component, use it only after verifying that it is the same pinned checkpoint and configuration. Otherwise keep the CLI result for later pipeline parity and use the component's own nnU-Net loader for stages A-C.

Record the generated file hashes, command, stdout/stderr, report JSON, runtime, and peak resident memory. Inspect the output in an approved medical-image viewer and confirm orientation and obvious left/right anatomy before using it as a reference. A plausible image is not proof of accuracy.

### Step 2: Pin and inspect the model assets

1. Pin TotalSegmentator and nnU-Net source versions. Prefer immutable release tags/commits over `master` or unpinned `pip install`.
2. Set a private model-cache location outside the checkout. The upstream project supports `TOTALSEG_HOME_DIR` for its model cache; verify this behavior for the pinned release.
3. Download only through the upstream mechanism and terms permitted for the selected task. Do not put checkpoints in Git, application installers, or public logs.
4. Use the pinned task/class listing (`totalseg_info` where supported) and inspect the actual downloaded model folder. Record the mapping from model output index to class name from the pinned metadata, not from a current online page.
5. Inspect the nnU-Net plans/configuration and checkpoint metadata. Record patch dimensions and axis order, number of input channels, number of output classes, normalization scheme, target spacing, trainer, fold(s), and any postprocessing assumptions.
6. Hash the checkpoint and all sidecars. The model identity is the set of those hashes plus the exact upstream and tool versions.

Never load a checkpoint from an untrusted source. Never infer a license from the Python package's license alone; model weights, task-specific assets, and redistribution may have distinct terms.

### Step 3: Create a locked offline factory environment

Keep the Python environment separate from RadAssist and Rust application dependencies. Select Python, PyTorch, nnU-Net, TotalSegmentator, ONNX, ONNX Script/exporter, and ONNX Runtime versions as a compatible set. The upstream TotalSegmentator repository currently lists Python 3.10+ and PyTorch 2.0+ as minimums; these are not a tested compatibility matrix for this experiment.

The team should use a lock-producing tool such as `uv` or a fully pinned requirements file. Fill in and review actual versions before running an install; angle-bracket values below are placeholders, not installable pins:

```text
# requirements.in (example categories; exact versions must be selected and reviewed)
TotalSegmentator==<PIN>
nnunetv2==<PIN>
torch==<PIN>
onnx==<PIN>
onnxruntime==<PIN>
onnxscript==<PIN>
```

For the initial CPU baseline, install the CPU ONNX Runtime package only. Do not install both CPU and GPU runtime packages into the same environment unless the pinned runtime explicitly documents that combination. Save the resolved environment lock, package inventory, Python version, platform details, and install logs with the experiment. A `pip freeze` file is useful evidence, but it is not a replacement for a reproducible lock.

The export machine may need network access to acquire pinned code and weights. The inference harness should run from local, hash-verified inputs; do not send scans to online services. Disable optional usage reporting using the pinned project's supported setting where available, and record that it is disabled.

### Step 4: Export only the network forward pass

The exporter must load the exact network architecture and checkpoint through the pinned nnU-Net/TotalSegmentator code. Do not reconstruct the U-Net by hand and do not export the CLI as a whole Python program.

Implement the exporter as a small version-controlled script in the factory source directory. Its responsibilities are:

1. Load the pinned plans, dataset metadata, configuration, checkpoint, and intended fold(s) using the pinned loader API.
2. Put the network in evaluation mode and inference mode. Resolve deep-supervision behavior using the pinned model API; the exported wrapper must return the same final prediction tensor that the reference predictor uses, not an auxiliary training head.
3. Build a representative dummy tensor with the exact model input dtype, batch/channel dimensions, and fixed spatial patch dimensions from the model plan. nnU-Net commonly uses 3D tensors in `[N, C, D, H, W]` form, but confirm the actual plan and runtime interface rather than assuming the axis order.
4. Wrap the network if necessary so the exported graph has a simple tensor input and final-logit tensor output. Record whether output is logits or probabilities; do not apply softmax twice.
5. Export with a pinned PyTorch exporter mode and opset. Try the least-custom graph first. If an operator is unsupported, record the operator and assess an equivalent standard ONNX decomposition before considering custom operators. Custom ORT extensions are out of scope for this first proof.
6. Save an export sidecar with model/checkpoint hashes, plans and metadata hashes, patch size and axis order, input/output tensor names and dtypes, output meaning, opset, exporter mode, and exact package versions.

The exact `nnUNetPredictor` and PyTorch ONNX-export calls are version-sensitive. Write them against the selected locked versions and test the loader first; do not copy an old API snippet and silently adapt it until it appears to work. Ensure the exporter has no hidden network download, random augmentation, training-mode dropout, or nondeterministic preprocessing.

Suggested first export artifact names:

```text
model.fp32.onnx
export-sidecar.json
export-log.txt
```

Do not add FP16, INT8, dynamic quantization, graph surgery, or multiple execution providers until FP32 CPU parity passes.

### Step 5: Validate ONNX structure and execute one patch

Run the ONNX checker and inspect the model's graph inputs, outputs, opset, tensor element types, and static/dynamic dimensions. Confirm that there is one intended 3D patch input and the expected class-logit output. If shape inference cannot establish some dimensions, document that explicitly.

Next, run the exported graph with the pinned CPU ONNX Runtime using a single valid preprocessed patch. Check:

- The session opens without custom operators.
- The graph returns exactly the expected output tensor(s) and shape.
- Every output is finite; there are no NaN or infinity values.
- Repeated runs with the same input are stable within the agreed numeric tolerance.
- The output class count matches the pinned metadata.
- Runtime logs identify the actual execution provider; do not infer CPU/GPU use from the package installed.

Retain the ONNX checker output, graph summary, sample input hash, output hash, session settings, and logs. Do not use a synthetic random tensor as the only evidence; it can reveal shape or operator errors, but does not represent the model's real input distribution.

### Step 6: Run tensor-level numeric parity

Build a fixed patch corpus from approved de-identified CT cases. Use the pinned reference preprocessing to generate the input patches once, then feed the identical bytes to PyTorch and ONNX Runtime. Include more than one anatomy location and representative intensity patterns. Store patch hashes and extraction coordinates, not PHI.

Compare raw outputs before argmax. Report, per patch and across the corpus:

- Maximum and mean absolute error.
- Maximum and mean relative error, with a documented zero-denominator policy.
- Fraction of voxels with identical argmax class.
- Per-class Dice on patch argmax outputs, including small structures when present.
- Non-finite counts and output shape/dtype.

Set tolerances before running the holdout patches. The broader architecture draft proposes an initial FP32 target of maximum absolute difference below `1e-3` and argmax agreement of at least `99.99%`; these are **engineering targets to validate for the pinned exporter/runtime**, not universal guarantees and not clinical acceptance criteria. If they are changed, record the reason and rerun a clean holdout set. A high global argmax agreement can hide a small-organ regression, so also report per-class metrics and inspect disagreements.

Any failure must be diagnosed by layer: checkpoint/fold mismatch, eval/deep-supervision handling, input layout/dtype, exporter lowering, unsupported operator, or runtime kernel differences. Do not loosen thresholds solely to make a failed export pass.

### Step 7: Prove volume-pipeline parity

Tensor parity proves only the network forward pass. The next harness must use the same volume and compare the pinned TotalSegmentator reference against the candidate ONNX-backed path, including:

- DICOM/NIfTI geometry interpretation and coordinate frame.
- Axis permutation/flips and laterality.
- Resampling to the model spacing and interpolation order.
- Intensity rescale/normalization and padding.
- Sliding-window starts, patch order, overlap, and Gaussian blending.
- Probability activation, class selection, postprocessing, and inverse resampling.
- Final label IDs, affine, dimensions, spacing, and origin.

Start with one model component and compare only the labels produced by that component. Resolve the semantic label map from the pinned model metadata. Do not compare model-local indices as if they were stable anatomy IDs.

For each agreed organ class, calculate Dice and volume difference on the same native grid; also check label bounds, laterality, affine/spacing equivalence, border-touch/incomplete status, and empty masks. The architecture draft proposes per-class Dice `>= 0.97` and volume difference `<= 2%` for the initial enabled pack. Treat those as draft engineering gates requiring review against fixture variability and the intended use. A parity pass against TotalSegmentator is not proof of clinical accuracy because both paths share the same model.

Include at least one asymmetric laterality fixture or a controlled test volume with known orientation. Demonstrate that a deliberate left/right flip is detected by the geometry/laterality checks. If no independently reviewed reference masks are available, label the evidence as implementation parity only, not clinical validation.

### Step 8: Run the same graph from Rust

Only after Python ONNX Runtime passes should the team add a minimal Rust smoke harness using the selected, pinned `ort` crate and ONNX Runtime shared library. The existing Cargo workspace currently has no ONNX Runtime dependency, so keep this as an isolated Phase 0 harness until architecture and packaging decisions are made.

The Rust harness should:

1. Load the same `model.fp32.onnx` by its SHA-256 identity.
2. Load the same serialized preprocessed patch used in Step 6, without changing byte order or layout.
3. Create a CPU session and run one inference; record provider, session options, input/output metadata, and errors.
4. Compare Rust output to the Python ONNX Runtime output and the PyTorch output with the same parity code/tolerances.
5. Repeat the run and test a malformed shape/dtype to confirm errors are explicit and bounded.

This proves the chosen Rust binding can host the graph. It does not yet establish the app's worker protocol, memory coordination, platform packaging, or volume pipeline.

### Step 9: Measure resource use and reliability

Benchmark only after correctness checks pass. Record cold session creation separately from warm inference. Fix the input patch and thread/provider configuration across repeated runs. Report median and p95 latency, peak resident memory, model file size, session-load time, and whether the measurement includes preprocessing or disk I/O.

Run enough repeated inferences to reveal leaks or unstable failures. Note CPU model, core/thread settings, RAM, OS, runtime build, GPU model/driver/provider if applicable, and power mode. Keep GPU measurements separate from CPU baseline. A provider that accepts the model may still fall back to CPU for unsupported nodes; capture provider assignment/fallback information.

For the initial Phase 0 desktop result, Linux CPU is the required reproducible baseline. Then run a small smoke matrix on Windows CPU and Android arm64 CPU to expose packaging/operator/provider issues. Do not promise Android performance based on a Linux benchmark. If GPU is considered, test each provider separately and compare outputs again.

### Step 10: Make a go/no-go decision

**Go to Phase 1 design/implementation only when:**

- The source release, checkpoint, plans, class map, hashes, license review, and environment are pinned and recorded.
- The FP32 ONNX graph passes structural checks and runs with standard ONNX Runtime operators.
- Tensor-level FP32 parity passes the pre-agreed tolerances on a held-out, representative patch corpus.
- The one-component volume pipeline passes the reviewed per-class, geometry, and laterality gates, or any failures have a documented, accepted resolution.
- The Rust smoke harness produces equivalent output on the same patch.
- Resource measurements are complete for Linux CPU and the platform smoke results are documented.
- No fixture or patient data was committed, uploaded, or included in logs.

**No-go / investigate** if the checkpoint identity is ambiguous, export requires an undocumented custom operator, output parity fails, laterality is uncertain, or a target platform cannot execute the graph within the agreed budget. A no-go is a useful Phase 0 outcome: choose a different model/runtime/export path before building application infrastructure.

## 6. What to commit and what never to commit

Commit source scripts, environment lockfiles, schema definitions, non-sensitive test metadata, and a redacted evidence report. Do not commit weights, patient images, model outputs derived from patient images, access credentials, license keys, or large runtime binaries unless a separately approved asset policy explicitly allows it.

Suggested handoff artifacts:

```text
model-factory-source/                 # scripts and lockfile in version control
phase0-evidence/<experiment-id>/
  experiment.json                     # versions, hashes, config, evidence IDs
  export-report.json
  tensor-parity-summary.json
  volume-parity-summary.json
  benchmark-summary.json
  environment.txt
  logs-redacted/
```

Store large ONNX/checkpoint files in access-controlled artifact storage only after license review. The report should identify their hashes and restricted storage location, not embed them.

## 7. Common failure modes and responses

| Symptom | Likely cause | Response |
|---|---|---|
| ONNX export fails on an operator | Unsupported exporter lowering or an operator outside standard ONNX | Identify the exact op and source node; try a supported exporter/decomposition; do not add custom ORT kernels in the first spike |
| Output has wrong shape or multiple outputs | Deep supervision, wrong wrapper, or model config mismatch | Re-check loaded plan/checkpoint and make the wrapper return the reference final prediction only |
| Outputs differ substantially | Wrong checkpoint/fold, preprocessing mismatch, train mode, layout or dtype error | Compare hashes and intermediate tensors stage by stage; never tune tolerances first |
| Good patch parity but bad full-volume masks | Sliding windows, blending, orientation, resampling, activation, or inverse geometry mismatch | Compare intermediate geometry and probability volumes before final labels |
| Left/right labels are swapped | LPS/RAS conversion, affine interpretation, or axis flip mismatch | Stop publication; fix the single orientation transform and add a regression fixture |
| Runtime starts but is unexpectedly slow | CPU fallback, excessive threads, repeated session construction, or memory pressure | Capture provider assignment, reuse a session, measure cold/warm separately, tune only after correctness |
| Model cannot be redistributed | Weight/license terms do not permit the intended distribution | Stop bundle packaging; seek permission or select a model with acceptable terms |

## 8. Team knowledge checklist

Before calling the experiment complete, the engineer should be able to explain:

- The difference between a checkpoint, a network graph, and a complete inference pipeline.
- Why the model's preprocessing plan and label map are part of the model identity.
- Why patch-level parity and full-volume parity are separate gates.
- How DICOM LPS/NIfTI RAS and affine geometry can silently change laterality.
- Why all predicted classes should participate in the model's normal argmax before exposing a validated subset as an organ pack.
- Which operations run in the model factory and which would eventually be reimplemented in Rust.
- What the tested ONNX Runtime provider actually executed, and what hardware was measured.
- What the evidence does and does not establish; specifically, parity to a reference implementation is not independent clinical validation.

## Execution status (2026-10-10)

The isolated factory and Rust smoke harness are implemented and their local unit and configuration checks pass. The Python suite also exercises comparison behavior with synthetic and mocked inputs. These checks do not establish checkpoint compatibility, ONNX exportability, parity on real CT data, or inference performance.

**Verified environment:** Python 3.12.3; TotalSegmentator 2.5.0; nnU-Net 2.5.1; PyTorch 2.5.1+cpu (CUDA unavailable); ONNX 1.17.0; Python ONNX Runtime 1.20.1; Rust `ort` 2.0.0-rc.10. Python ONNX Runtime reports Azure and CPU providers as available; the experiment code explicitly requests and checks `CPUExecutionProvider`.

**Verified commands (from the repository root):**

| Command | Outcome |
|---|---|
| `uv lock --check --project tools/model-factory/totalseg-onnx` | Passed; 94 locked packages resolve without changing the lock. |
| `uv run --locked --project tools/model-factory/totalseg-onnx python -m unittest discover -s tools/model-factory/totalseg-onnx/tests -v` | Passed; 14 tests, including fixture approval/redaction, missing/hash-mismatched checkpoint rejection, mocked tensor parity thresholds, volume geometry, and deliberate laterality-flip detection. |
| `cargo test --manifest-path tools/model-factory/totalseg-onnx/ort-smoke/Cargo.toml --locked` | Passed; 6 tests, including approval-gate parity and runtime/tensor evidence fields. |
| `cargo fmt --manifest-path tools/model-factory/totalseg-onnx/ort-smoke/Cargo.toml -- --check` | Passed. |
| `uv run --locked --project tools/model-factory/totalseg-onnx phase0 --help` | Passed; all planned stage commands are registered. |

**Not run; hard prerequisites are missing:**

- The user-authorized `CT-Test.rar` was inspected locally; 74 DICOM entries were extracted temporarily with neutral filenames through the preinstalled libarchive library. A header-only scan found direct identifiers and private tags in every image; the archive contains three CT series (two 35-slice stacks and four localizers). No tag values or UIDs were logged, and no image data was passed to inference. The temporary extracted copies were deleted; the original archive was not modified. De-identification remains **unverified and failed for the provided archive as-is**.
- No private TotalSegmentator checkpoint cache was present. The selected checkpoint, plans, class map, hashes, and separate weight-license review are not verified.
- No experiment manifest records human approval of the fixture reviews or numeric thresholds. The existing gates correctly block patch extraction, export/parity runs that require approved input, volume comparison, benchmarking, and final validation.
- Consequently, ONNX export, real-patch Python parity, full-volume parity, Rust inference parity, and latency/memory benchmarks have no results. No ONNX model or accepted evidence report was produced.
- Windows and Android runtime smoke tests were not run; this host is Linux.

These are incomplete feasibility stages, not passing results. Continue only with a properly de-identified fixture (the provided archive does not pass that gate), recording reviewer/timestamp approvals, verifying the exact checkpoint and license, and approving thresholds before holdout evaluation.

## 9. Local CT/MR Metadata-Cleaning Procedure

This procedure creates **restricted metadata-de-identified working copies**, not anonymous or approved fixtures. The originals are read-only inputs. No medical data is uploaded, and the cleaner never invokes a model or changes an experiment's approval fields.

### Pinned Tooling And Profile

The factory lock pins `dicom-anonymizer==2.1.0`, `libarchive-c==5.3`, and the resolved `pydicom==3.0.2`. This Linux host uses native libarchive 3.7.2; the Python wrapper is locked, but the system library is recorded rather than bundled. The installed APIs were verified before use. The selected library rules are explicitly `initialize_actions_2024b`, based on DICOM PS3.15 2024b's Basic Application Level Confidentiality Profile.

The wrapper applies those rules at every nested sequence level, removes private tags recursively, replaces all remappable UIDs consistently within one run, and additionally clears identifying free text and code strings that are not on a finite technical-code allowlist. Dates and times become fixed neutral values. Class and transfer-syntax UIDs remain unchanged, including valid vendor class UIDs. Fresh file metadata and a zeroed preamble prevent header/preamble identifiers from being copied. No source-to-replacement UID or filename mapping is written.

This is a conservative library-profile application with documented extensions, **not a claim of complete DICOM IOD conformance or anonymity**. Standard technical rescale units are retained. Unsupported metadata is excluded with typed, value-free quarantine counts. Public overlay elements are removed; embedded overlay bits and burned-in text inside the unchanged pixel payload are not cleaned. Neither `BurnedInAnnotation=NO` nor OCR constitutes review evidence.

### Exact Local Commands

Run from the repository root. The output root must be absolute, outside the repository and source tree, owned by the current user, and mode `0700`. The following permission command was needed because the pre-existing fixture root was not owner-only:

```bash
rtk uv sync --locked --project tools/model-factory/totalseg-onnx
rtk proxy chmod 700 "$HOME/.local/share/radassist-model-factory/fixtures"
rtk uv run --locked --project tools/model-factory/totalseg-onnx phase0 deidentify \
  --source Medical_TestReports/CT-Test.rar \
  --output-root "$HOME/.local/share/radassist-model-factory/fixtures" \
  --run-id ct-metadata-20261010-r2
rtk uv run --locked --project tools/model-factory/totalseg-onnx phase0 deidentify \
  --source Medical_TestReports/RadAssist-Test \
  --output-root "$HOME/.local/share/radassist-model-factory/fixtures" \
  --run-id collection-metadata-20261010-r1
```

These run IDs now exist and **must not be reused**. Choose a new safe ID to repeat a run; existing destinations are never overwritten. Archive entries are streamed into memory, never extracted using their source names. Traversal paths, links, encryption, special entries, malformed DICOM, and bounded-input failures reject the run before publication. Limits are 64 MiB per instance, 4 GiB per collection/archive, and 10,000 entries. Directory sources also reject links and special files. Modality is read from DICOM headers, not filenames. Only supported single-frame CT/MR images are published; other modalities are counted without publishing them.

### Actual Outcomes (2026-10-10)

| Source | Header Modality | Input / Published | Series | Localizers | Inputs With Overlays | Outcome |
|---|---|---|---|---|---|---|
| Authorized CT archive | CT | 74 / 74 | 3 | 4 | 3 | `review_required`; no rejected instances |
| Existing local collection | MR | 120 / 120 | 11 | 0 declared in ImageType | 22 | `review_required`; no rejected instances |

Both successful runs are under `~/.local/share/radassist-model-factory/fixtures/`, using neutral `patient-0001/series-0001/image-000001.dcm`-style paths. Directories are `0700`; DICOM and audit files are `0600`. Each private `audit.json` records source hashes before and after, per-copy and pixel-payload hashes, tool/profile versions, counts, validation outcomes, limitations, and empty human-review fields. Source directory fingerprints include ordered relative filenames and file hashes **only inside the digest**, never the filenames themselves in the audit.

For every published copy, the saved file was re-read and compared against its source for exact pixel payload, transfer syntax, dimensions, positions, orientation, spacing, quantitative rescale/window fields, and numeric CT/MR acquisition fields. Nested UID contracts and file-meta SOP links were checked. A separate post-run check rehashed both sources and all 194 saved copies, repeated metadata validation, and confirmed permissions. All these checks passed; the originals are unchanged.

The CT input already references absent external images in 148 reference occurrences; the MR input has 125 such occurrences. Their replacement UIDs are consistent, but the absent targets remain **unverified**, as explicitly recorded in each audit. Supplied-reference targets lost through quarantine would reject publication. UID consistency does not prove that an incomplete source collection is a complete imaging study.

Earlier attempts remain accurately distinguished: `ct-metadata-20261010-r1` is a preserved audit-only quarantine run from the initial strict compatibility checks, with zero published images. The first `r2` attempt rejected existing external references and left no published run; after synthetic regression checks, `r2` was successfully created with the external-reference limitation recorded. No earlier evidence was overwritten.

### Focused Hardening And New Copies (2026-10-10)

The earlier runs remain historical evidence and were not overwritten. The corrected cleaner produced two new owner-only runs outside Git:

| New Run ID | Published | Rejected | Source / Copy Hashes And Permissions |
|---|---|---|---|
| `ct-metadata-20261010-r3` | 74 CT | 0 | Passed |
| `collection-metadata-20261010-r2` | 120 MR | 0 | Passed |

Both are under `/home/amit/.local/share/radassist-model-factory/fixtures/`. All 197 pre-existing run files were independently checked unchanged. The new audits continue to report 148 CT and 125 MR absent external reference occurrences as unverified. No pixel review, inference, or conversion was run.

Focused synthetic tests now total 30; the full factory suite totals 55, all passing. New coverage includes traversal errors in all source passes, consumed-file symlink/FIFO swaps, deflated preflight, library UID-cache cleanup, redacted resolution/cleanup failures, staging and atomic-publication failures, destination collisions, filtered supplied targets, reference-class mismatch, rejected-then-good duplicates, invalid geometry/native lengths, archive entry/expanded-byte quotas, and rejection of a pre-existing permissive output root.

Traversal errors now abort explicitly. File hashing and consumption use no-follow regular-file descriptors with `fstat`; archive reading uses the checked descriptor. Deflated transfer syntax is excluded before pydicom inflates dataset bytes. Profile processing never receives source file meta, and both UID dictionaries are cleared on run exits. Only accepted instances participate in duplicate suppression; supplied target IDs/classes are recorded before modality filtering and checked against references.

Completed staging directories are published in one Linux `renameat2(RENAME_NOREPLACE)` operation, never moved child-by-child or renamed over an existing run. Unavailable no-replace support fails closed. Only owned pending staging is cleaned, never an existing destination. Cleanup failure reports `staging_cleanup_failed` without paths and can leave restricted pending data requiring local operator attention. Atomic visibility is not a guarantee of crash durability; do not treat a pending directory as completed evidence.

The new per-instance `imaging_validation` value is **`payload_and_geometry_preserved`**, not image-validity approval. `imaging_checks` records bounded geometry and native pixel-length checks separately: positive dimensions and spacing, finite bounded numbers, approximately orthonormal orientation, supported monochrome bit layout, nonzero slope, bounded expected pixel bytes, and even-length native padding. Compressed codestream decoding is not performed. **Decoded validity, full DICOM IOD conformance, series completeness/consistency, and suitability remain pending**, even when every preservation check passes. Older audits' `imaging_validation=passed` means the older invariance check only, not retrospective suitability approval.

Use this as a **single-process local CLI on a stable, administrator-controlled input snapshot**. Stdout/stderr redirection and logging suppression are process-global, not thread-safe isolation; native/subprocess diagnostics are not guaranteed captured. No concurrent writers or ancestor-directory replacement are supported. Before/after hashes detect lasting source changes but cannot rule out transient modify-and-restore activity. No-follow/fstat leaf-file checks reduce consumed-file races; they do not create a filesystem snapshot or establish adversarial race immunity. Do not embed this cleaner into a concurrent service without separate isolation design.

The new run IDs are already occupied. To reproduce under fresh IDs, use the same authorized source arguments and a new safe ID:

```bash
rtk uv run --locked --project tools/model-factory/totalseg-onnx phase0 deidentify \
  --source Medical_TestReports/CT-Test.rar \
  --output-root "$HOME/.local/share/radassist-model-factory/fixtures" \
  --run-id <new-safe-ct-id>
rtk uv run --locked --project tools/model-factory/totalseg-onnx phase0 deidentify \
  --source Medical_TestReports/RadAssist-Test \
  --output-root "$HOME/.local/share/radassist-model-factory/fixtures" \
  --run-id <new-safe-collection-id>
```

### Mandatory Human Review And Eligibility

The earlier 194 copies and the new 194 copies all have `testing_eligible=false`. Their `pixel_review.status` is `required`; reviewer, timestamp, and outcome are unset. De-identification and suitability remain `pending`. No automated result is represented as human or clinical approval.

Privately inspect **every cleaned frame**, and inspect the corresponding original overlay content where overlays were removed. Check burned-in identifiers, embedded overlay text, and identifying anatomy such as facial structures. Record reviewer, review time, outcome, and the exact reviewed fixture SHA-256 in owner-only review evidence. Do not publish screenshots or identifying observations. Hash changes invalidate prior review evidence. Cases that fail or cannot be reviewed remain quarantined/restricted; do not automatically crop or paint anatomy.

Only reviewed, authorized, technically suitable CT can later be converted to an external NIfTI fixture for Phase 0, with separate recorded de-identification/suitability approvals and the existing model/license/threshold gates still satisfied. Conversion and manual approval are not performed by this command. MRI copies are **decoder/viewer-test input only after review**, never input to the CT-only TotalSegmentator checkpoint. The earlier inference-blocked status remains unchanged.

## 10. Upstream References

### Latest execution checkpoint (2026-10-10)

This checkpoint supersedes the earlier statement that no checkpoint or ONNX graph was available. Phase 0 remains incomplete.

- Downloaded only the official task-291 organs component into private storage outside Git. TotalSegmentator 2.5.0 references this asset under the v2.0.0 weights release; software and weight release versions are distinct.
- Checkpoint SHA-256: `98c429f89fd9b3b08890fbb2b4c08c365ad4655fdcadb45efaa29380129fd890`. Fold: `0`; trainer: `nnUNetTrainerNoMirroring`; patch: `128x128x128`; spacing: `1.5 mm`; labels: 25 including background.
- FP32 ONNX export passed structural validation: opset 17, 138 nodes, no custom operator domains. Graph SHA-256: `f1cca375f46852efe576d310326f82367b670ebf98493344a111479951eb2265`.
- Synthetic zero-patch CPU execution returned finite float32 logits with shape `[1,25,128,128,128]`. Session load: approximately 0.13 seconds; inference: 5.57 seconds; peak process RSS: approximately 1.59 GiB (1,671,084 KiB). These are single-run synthetic measurements, not a benchmark acceptance or real-input parity result.
- Local validation: 25 Python tests and 7 Rust tests passed; lock, Rust formatting, and whitespace checks passed. Rust tests include actual malformed-input rejection using a tiny synthetic ONNX graph.
- Added `setuptools==80.9.0` to the isolated environment: the pinned TotalSegmentator downloader requires its deprecated `pkg_resources` API.
- The provided CT archive contains identifying metadata and private tags. Its temporary extracted copies were deleted; the original archive is unchanged. No supplied CT data was used for inference.
- At this initial checkpoint, real-patch parity, full-volume parity, Rust parity against a trained-checkpoint graph, and repeated benchmarks had not run. See the follow-up checkpoint below. Internal upstream license inspection is not legal approval for distribution.

The exported graph and sidecar are stored locally under `~/.local/share/radassist-model-factory/runs/totalseg-organ-export-20261010-r1/onnx/`. They are not committed or included in the application build.

### Follow-up: reviewed CT and trained-checkpoint parity (2026-10-10)

The reviewer approved the private CT fixture and starter thresholds. A supplemental owner-only review record binds all 74 cleaned CT instance hashes to that review; the original cleaner audit remains unchanged. One reviewed 35-slice series was converted to private NIfTI with locked `dicom2nifti==2.6.2`, then checked for a 3D shape, finite voxel values, spatial affine, and owner-only permissions. No patient-derived content or paths are stored in this repository.

The first real-input attempt exposed an exporter defect: `nnUNetPredictor.initialize_from_trained_model_folder` builds the network and stores the selected fold weights in `list_of_parameters`, but does not load them into `predictor.network` until its inference loop. The initial export and direct PyTorch parity reference therefore used randomly initialized weights. Those graphs and results from runs `r1` and `r2` are invalid for the trained checkpoint and are preserved only as failed historical evidence. The exporter and direct tensor-parity reference now explicitly load the selected checkpoint state; both reject a multi-fold selection because this Phase 0 graph represents one fold only. Patch capture now uses a forward pre-hook so nnU-Net can continue reloading its state dict without wrapper key mismatches.

Corrected run `totalseg-organ-export-20261010-r3` exported the trained fold to a structurally valid 77-node FP32 graph with a deterministic nonzero synthetic trace input. Four fixed-size preprocessed patches were captured from the reviewed CT NIfTI. Python PyTorch-versus-ONNX parity **failed** the previously approved maximum-absolute-logit threshold of `0.001`: maximum error `0.0134063`, mean error `0.0005413`, argmax agreement `1.0`, with no non-finite outputs. Per-class Dice was unavailable (`null`) for every foreground class because no foreground voxels were predicted in either output on these patches; this does not establish class coverage. Do not relax thresholds after seeing this result. The report is recorded in the private r3 manifest.

Rust CPU inference on the corrected r3 graph was separately checked only on the synthetic zero patch: Rust ONNX Runtime 1.22.0 versus Python ONNX Runtime 1.20.1 had max absolute error `0.0000172` and mean error `0.000000816`. This is not Rust parity on the CT patches. Real-input Rust parity, full-volume comparison, benchmark, and final `validate` were not run because the Python tensor-parity gate failed. **Decision: no-go for the next implementation phase** until the logit discrepancy and absent foreground-class coverage are understood, and any proposed threshold change is approved before a new holdout run.

### Follow-up: RadAssist worker full-volume parity (2026-10-16)

This run used only the approved upstream `example_ct.nii.gz` fixture (SHA-256 `dbd3ae6d614d1d7ef3a46925c30c70038ed52da6b37fdd6afdc7b9b71387e3e1`), the pinned TotalSegmentator v2.5.0 / nnU-Net 2.5.1 `part_organs` model, and CPU execution with mirroring disabled in both paths. No user-provided CT or other local medical fixture was used. The model, plans, checkpoint, class-map, and approved thresholds were hash-checked; the thresholds were not changed.

The first app-vs-reference result had mean Dice `0.070358` across 19 present classes, colon Dice `0.149`, and failed liver/spleen laterality. The intermediate preprocessing correction raised mean Dice to `0.902353`, but only 4/19 classes passed Dice `>= 0.97`. The ONNX graph was not modified between the failed initial app comparison and the final pass; the approved graph SHA-256 remained `d5e4709a0db84d353b9c615acb518075971968c08b5047192dda15db43bbab2f`. The app-side preprocessing/inference pipeline was corrected to match nnU-Net's input orientation, crop, CT normalization, cubic order-3 input resampling, evenly distributed sliding-window starts, Gaussian overlap weights, and order-1 logit resampling before argmax. With that unchanged graph, the final run passed:

| Measure | Final result | Approved gate |
|---|---:|---:|
| Mean Dice, 19 nonempty classes | `0.999949` | All classes `>= 0.97` |
| Per-class Dice passes | `19/19` | `19/19` |
| Per-class volume-delta passes | `19/19` | `<= 2%` |
| Geometry | Exact shape, affine, and spacing match | Pass |
| Liver/spleen laterality | Both pass | Pass |
| Preprocessing tensor max absolute error | Crop `0`; normalized `1.19e-7`; resampled `3.58e-7` | Diagnostic tolerance `1e-4` |

The preprocessing tensor report is `~/.local/share/radassist-model-factory/runs/totalseg-app-preproc-parity-20261016-r4/preprocessing-parity.json`; the final automated full-volume report is `~/.local/share/radassist-model-factory/runs/totalseg-app-preproc-parity-20261016-r8/app-volume-parity.json`. Both run directories are `0700`, reports are `0600`. The r8 report was produced by `OfficialWorkerVolumeParityTests.test_approved_official_fixture_passes_rust_worker_reference_gates` with the private manifest, fixture, ONNX graph, and release worker supplied through explicit environment variables. It includes fixture, graph, checkpoint, plans, class-map, and worker hashes. Temporary tensors and labelmaps were removed after comparison.

The final r8 worker took approximately `306.0 s`; pinned nnU-Net took `133.1 s` on this CPU host. The worker compacts the cropped input in place, normalizes in place, consumes the source during cubic resampling, and checks combined source/spline and spline/target live buffers against the preprocessing budget. This does not establish a total process RSS budget on Android; that device profile remains deferred. The result is not clinical validation and provides no evidence about other fixtures or devices. The Python reference remains factory-only; the product worker remains Rust/ONNX Runtime.

The isolated factory exposes `phase0 preprocessing-parity` and `phase0 worker-volume-parity`; the latter is covered by an opt-in test requiring explicit private asset paths and a no-clobber report. Final verification on this checkout: 66 factory tests passed in the default suite (one private-asset integration test skipped there and passed explicitly on r8), 25 worker tests passed, release worker built without warnings, Rust formatting passed, and `git diff --check` passed. Existing frozen phase-0 and integration specs were not changed.

Verify these references again when selecting versions; upstream documentation is mutable.

- [TotalSegmentator repository and usage](https://github.com/wasserth/TotalSegmentator)
- [TotalSegmentator class details and label mappings](https://github.com/wasserth/TotalSegmentator/blob/master/resources/class_details.md)
- [nnU-Net v2 repository and documentation](https://github.com/MIC-DKFZ/nnUNet)
- [PyTorch ONNX export documentation](https://docs.pytorch.org/docs/stable/onnx.html)
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/)

The upstream documentation was consulted on 2026-10-10. The run's pinned source tag, package lock, checkpoint hashes, and license decision take precedence over this date-stamped summary.