# ONNX and Rust Engineering Guide

**Document version:** 1.0
**Last verified against this repository:** 2026-10-10
**Audience:** Engineers new to machine-learning deployment, ONNX, or Rust inference
**Status:** Training and engineering-feasibility guide; not a clinical deployment procedure

This guide explains the concepts behind the repository's TotalSegmentator export experiment and how to reproduce the checks that are currently safe to run. The companion [Phase 0 runbook](totalsegmentator-onnx-phase-0.md) is the operational record and source of truth for experiment gates and evidence. If this guide and the runbook differ, use the newest dated runbook entry and update this guide in the same change.

## 1. The Short Version

The project starts with a trained 3D segmentation network loaded by PyTorch and nnU-Net. The exporter traces the network's inference computation and saves the computation plus its learned parameters as an ONNX graph. Python ONNX Runtime and Rust ONNX Runtime then execute that graph on CPU. The Rust `ort` crate is a Rust API over ONNX Runtime; it does not translate a PyTorch model itself.

The artifact is not a complete CT application. Image decoding, coordinate interpretation, resampling, normalization, patch creation, sliding-window blending, label selection, and conversion back to the original image geometry live outside the exported network unless explicitly implemented inside its graph. Those steps need separate tests.

The current demonstrated result is deliberately narrow: a pinned model graph loads and produces finite output for one synthetic all-zero patch, with exact numerical agreement in the recorded Python/PyTorch/ONNX and Python/Rust comparisons. This is **not** a real-CT end-to-end test, segmentation accuracy result, clinical validation, or approval to distribute model weights.

## 2. A Beginner's Mental Model

### 2.1 What is a model?

A neural network is a parameterized computation. It accepts numbers arranged into tensors, performs operations such as convolutions and nonlinearities, and returns other tensors. Training adjusts the parameters from examples. Inference uses the trained parameters to make a prediction; it does not update them.

For an image-segmentation network:

- **Architecture** is the recipe for the operations and their connections, for example a 3D U-Net. It is implemented by code and configuration.
- **Weights** are the learned numeric parameters. A checkpoint file stores weights and may also contain trainer state or metadata.
- **Input contract** defines tensor rank, shape, axis order, dtype, intensity convention, and preprocessing assumptions.
- **Output contract** defines shape, class-channel meaning, and whether values are logits or probabilities.
- **Inference pipeline** prepares real images, calls the model, and turns its output into a result in the source image's coordinate system.

These parts are related but not interchangeable. The checkpoint alone is generally not enough to run a model. For this nnU-Net model, the exact architecture code, plans, dataset/class metadata, fold, checkpoint, and preprocessing implementation all matter.

### 2.2 Tensors, logits, and labels

A tensor is a multidimensional array. This experiment's network consumes one 3D CT patch as a `float32` tensor with axes `[N,C,D,H,W]`:

| Axis | Meaning | Current value |
|---|---|---:|
| `N` | Batch count | 1 |
| `C` | Image channels | 1 |
| `D,H,W` | Three spatial patch dimensions | 128 each |

The input shape is `[1,1,128,128,128]`. The output is 25 raw class-logit channels with shape `[1,25,128,128,128]`; one channel is background. A logit is an unnormalized score, not a probability. To get a hard label at a voxel, choose the channel with the highest score (`argmax` across `C`). Do not apply softmax unless the consuming algorithm explicitly requires probabilities; do not apply it twice. The Rust parity evidence also reports per-class Dice for these hard labels; when both outputs contain no voxels for a class, this harness records Dice as `1.0` and marks `both_empty`. Review that flag and class presence: empty-versus-empty agreement is not evidence that the model can detect that anatomy.

The model's `D,H,W` axes are tensor axes, not a promise about DICOM patient directions. Geometry and axis transforms must be carried by preprocessing and postprocessing, not inferred from the shape.

### 2.3 Why one successful inference is not enough

There are several independent questions:

1. Can the intended checkpoint be loaded?
2. Can it be represented by a structurally valid ONNX graph?
3. Does ONNX Runtime produce equivalent logits on the exact same already-prepared tensor?
4. Does the surrounding full-volume pipeline preserve spacing, orientation, laterality, and labels?
5. Is the output useful and clinically appropriate for an intended use?

A pass at an earlier level does not answer a later question. In particular, structural validation and numerical parity are not clinical validation.

## 3. Model Files and Formats

There is no fixed, universal count of model formats. Formats serve different purposes: saving a training checkpoint, serializing weights, representing a computation graph, packaging quantized language-model weights, or compiling a graph for one accelerator. A filename extension alone does not tell you that an artifact is portable, safe, complete, or compatible with a runtime.

| Format or family | What it normally contains | Typical use | Important limitation |
|---|---|---|---|
| PyTorch checkpoint (`.pt`, `.pth`) | Python/PyTorch tensors and often checkpoint metadata; may be a state dictionary or a larger training checkpoint | Continue training, research, or run with the matching Python architecture | Not a universal graph format. Requires compatible model code, package versions, and a trusted loader. |
| Safetensors (`.safetensors`) | Tensor values and tensor metadata in a safer weight-serialization format | Store/share weights, commonly for transformer models | Usually weights, not the full executable graph, preprocessing, or runtime. |
| ONNX (`.onnx`) | A typed computation graph, parameters/initializers, operator versions, and tensor interfaces | Interoperable inference across runtimes and languages, including C/C++/Rust via ONNX Runtime | Portability depends on supported operators, opset, data types, and provider kernels. Not every model exports cleanly. |
| GGUF | A self-describing container for GGML-family inference, with tensors and model/tokenizer metadata; often quantized | Local LLM inference with llama.cpp and compatible tools; also some speech/transformer projects in the GGML ecosystem | It is not a general interchange graph like ONNX and is not the normal target for a 3D U-Net. Model architecture/runtime support is specific. |
| TorchScript (`.pt`) | A serialized TorchScript representation of a PyTorch module | Legacy PyTorch deployment paths | PyTorch describes TorchScript as deprecated for new export work; use the current PyTorch export guidance for new projects. |
| TensorRT engine / plan | A graph optimized/compiled for a particular NVIDIA runtime, device, and often software stack | High-performance NVIDIA inference | Usually hardware/runtime-specific; not a general portable artifact. |
| LiteRT / TFLite (`.tflite`) | A graph for Google's edge/mobile runtime | Mobile and embedded inference | Operator support and conversion constraints differ from ONNX. |
| Core ML (`.mlmodel`, `.mlpackage`) | Apple's model representation/package | Apple platforms and Apple accelerators | Apple-platform deployment target; conversion and operator support must be checked. |
| OpenVINO IR | OpenVINO graph plus weights | Intel CPU/GPU/NPU inference | Runtime/provider ecosystem is OpenVINO-specific. |

This is a useful shortlist, not a complete taxonomy. Some ecosystems have additional graph formats, checkpoint containers, quantized encodings, and hardware-compiled forms. Also distinguish **model format** from **runtime**: ONNX is a graph representation; ONNX Runtime is software that executes compatible ONNX graphs. GGUF is a model container used by a particular family of inference engines; llama.cpp is one such engine.

### 3.1 How to choose

Choose a format based on the model architecture, target devices, operator support, performance, licensing, and the runtime the application can ship and maintain.

| If the workload is... | First format/runtime to evaluate | Why |
|---|---|---|
| 3D segmentation, detection, classification, or another tensor graph | ONNX + ONNX Runtime | The model is a graph of tensor operations; ONNX Runtime has APIs usable from native code, including through the Rust `ort` crate. |
| Local decoder-only LLM using llama.cpp-compatible architecture | GGUF + llama.cpp-compatible runtime | GGUF packages the model/tokenizer metadata and quantized tensor representation expected by that ecosystem. |
| Training or research in PyTorch | PyTorch checkpoint + matching Python code | Preserves the framework's training and model-loading workflow. Convert only when a deployment target is selected. |
| One specific accelerator needing maximum throughput | Its compiler/runtime format, such as TensorRT | Use after a portable/reference implementation passes and after accepting reduced portability. |
| Native Apple/mobile/Intel edge deployment | Core ML, LiteRT, or OpenVINO, respectively | Prefer the platform's supported runtime when its coverage and packaging meet requirements. |

For RadAssist, the current TotalSegmentator component is a 3D U-Net-like nnU-Net inference graph, so the Phase 0 deployment candidate is **FP32 ONNX executed by CPU ONNX Runtime**. The app's existing MedASR/whisper.cpp lineage uses GGML/GGUF-oriented tooling. That does not make GGUF the right format for TotalSegmentator. A future MedGemma GGUF conversion would be a separate model, converter, quantization, license, and validation decision; it is not part of this segmentation proof.

### 3.2 ONNX versus GGUF in plain language

ONNX says, in effect, "here is a graph of operations, their parameters, and the input/output tensor contract." A compatible ONNX Runtime backend evaluates those operations. GGUF says, in effect, "here are model tensors and metadata packaged for a supported GGML-family model implementation." The runtime already needs to know how that model architecture uses those tensors.

They are not two levels of the same universal standard, and they are not generally interchangeable by renaming a file. Converting to either format can alter numerical behavior, supported operations, precision, or quantization. Compare outputs against a pinned reference after every conversion.

## 4. The Current Repository Proof

The isolated factory is `tools/model-factory/totalseg-onnx/`. It is not a product dependency and does not add Python or PyTorch to the shipped RadAssist runtime. The standalone Rust smoke harness lives at `tools/model-factory/totalseg-onnx/ort-smoke/` and is intentionally not part of the application Cargo workspace.

| Component | Pinned value in the verified experiment |
|---|---|
| Python | 3.12.3; project constraint `==3.12.*` |
| TotalSegmentator package | 2.5.0 |
| nnU-Net v2 | 2.5.1 |
| PyTorch | 2.5.1+cpu |
| ONNX | 1.17.0 |
| Python ONNX Runtime | 1.20.1 |
| ONNX Script | 0.2.5 |
| Rust crate | `ort` 2.0.0-rc.10 |
| Rust harness ONNX Runtime build | 1.22.0 in the recorded run |
| ONNX opset | 17 |
| Device/provider | Linux x86-64 CPU; `CPUExecutionProvider` |
| Model component | TotalSegmentator 2.5.0 `part_organs`, task 291 weights |
| Checkpoint | fold 0, `nnUNetTrainerNoMirroring`, 128 x 128 x 128 patch, 1.5 mm configured spacing |
| Class output | 25 channels including background |

The TotalSegmentator Python package version and the weight-release version are different version labels. The runbook records the actual task/checkpoint identity, SHA-256 hashes, and license review rather than treating a package version as sufficient model identity. Do not assume an old task ID, local cache path, fold, or label order applies to a different release.

The Python and Rust ONNX Runtime versions differ in this experiment. `ort = 2.0.0-rc.10` is the **Rust crate version**, not the ONNX Runtime version. The crate's `download-binaries` feature supplies a runtime binary for this harness; `ort::info()` reports the runtime build. Record both values. Agreement between two runtime versions is useful, but it does not make them the same runtime or remove the need for target-platform testing.

### 4.1 Export contract

The exporter in `phase0/exporter.py` builds the network through the pinned nnU-Net predictor. A subtle nnU-Net behavior matters here: `initialize_from_trained_model_folder` stores fold weights in `predictor.list_of_parameters` but does not load them into `predictor.network` until prediction. The exporter must explicitly call `network.load_state_dict(predictor.list_of_parameters[0])` before tracing; otherwise it exports random initialization rather than the trained model. Phase 0 rejects multiple selected folds because one exported graph does not represent nnU-Net's fold ensemble. The exporter then switches to evaluation mode, disables deep-supervision output where supported, and wraps the final prediction tensor. It traces a deterministic seeded nonzero synthetic `float32` sample, then calls the PyTorch ONNX exporter with `dynamo=False` and opset 17. Graph names are `image` and `logits`.

The seeded trace sample is a reproducible way to exercise nonzero operator behavior during export; it is not medical data and does not establish useful model behavior or parity. The graph sidecar records the trace seed, package versions, tensor contracts, opset, and hashes for the graph and source model assets. `onnx.checker` checks graph structure; it does not check medical-image correctness.

### 4.2 What has and has not been demonstrated

The initial 2026-10-10 synthetic result was later found to have used a graph exported before the deferred nnU-Net checkpoint weights were loaded. It is historical evidence only and does not prove trained-checkpoint parity. The corrected 2026-10-10 runbook checkpoint records:

- The trained task-291 fold was exported to a structurally valid FP32 ONNX graph: opset 17, 77 nodes, no custom operator domains.
- On four approved CT patches, Python PyTorch-versus-ONNX parity failed the pre-approved max absolute error limit (`0.0134063` measured vs `0.001` allowed); mean error was `0.0005413`, argmax agreement was `1.0`, and no output values were non-finite.
- Foreground class Dice was unavailable because both outputs predicted no foreground voxels in these patches. This does not establish class coverage or segmentation quality.
- The corrected graph's synthetic zero-patch Rust/Python comparison passed: maximum error `0.0000172`, mean error `0.000000816`, output shape `[1,25,128,128,128]`. It remains synthetic-only evidence.
- Rust parity on the real CT patches, full-volume parity, and benchmarks were not run because the Python tensor-parity stage failed. Do not loosen the approved threshold post hoc.

Still **not demonstrated**: representative real-patch parity, real-volume pipeline parity, approved clinical performance, repeated benchmark stability, Windows/Android packaging or performance, and permission to redistribute weights/graph. The CT/MR metadata-cleaned working copies are not automatically eligible: human pixel review, suitability approval, and recorded fixture gates remain required. Do not run medical fixtures until the runbook's approvals are complete.

## 5. Environment Setup

Run commands from the repository root. Use the committed `uv.lock`; do not independently upgrade one dependency to "fix" export. The CPU PyTorch index is configured explicitly in `pyproject.toml`.

### 5.1 Prerequisites and locked environment

Install Python 3.12 and `uv` using the team's approved machine setup. Check the active tools and synchronize the locked factory environment:

```bash
rtk uv --version
rtk uv python list
rtk uv sync --locked --project tools/model-factory/totalseg-onnx
rtk uv run --locked --project tools/model-factory/totalseg-onnx python --version
rtk uv run --locked --project tools/model-factory/totalseg-onnx phase0 --help
```

The Python version command should report Python 3.12.x. If `uv` cannot find that interpreter, install Python 3.12 through the approved OS/toolchain process or `uv python install 3.12`, then repeat the sync. A successful sync means dependencies resolve; it does not download/approve the model checkpoint or authorize any fixture.

Useful environment checks and tests:

```bash
rtk uv lock --check --project tools/model-factory/totalseg-onnx
rtk uv run --locked --project tools/model-factory/totalseg-onnx python -m unittest discover \
  -s tools/model-factory/totalseg-onnx/tests -v
rtk cargo test --manifest-path tools/model-factory/totalseg-onnx/ort-smoke/Cargo.toml --locked
rtk cargo fmt --manifest-path tools/model-factory/totalseg-onnx/ort-smoke/Cargo.toml -- --check
```

The recorded Python test suite had 55 passing tests; the Rust suite had 7 passing tests at the latest checkpoint. These are tooling tests and synthetic/mocked checks, not proof of real CT inference. Re-run them on the current checkout and use the command output as the current result.

### 5.2 Private directories and environment variables

Keep checkpoints, fixtures, generated patches, ONNX graphs, outputs, and manifests outside the Git checkout. The CLI rejects experiment output destinations inside the repository. On Linux/macOS, this example establishes a private local root:

```bash
export RADASSIST_FACTORY_ROOT="$HOME/.local/share/radassist-model-factory"
umask 077
mkdir -p "$RADASSIST_FACTORY_ROOT"
chmod 700 "$RADASSIST_FACTORY_ROOT"
```

When a `phase0` command runs, `runtime_env.py` derives and sets the model/cache directories from the experiment manifest's `outputs_root`:

| Variable | Private directory | Purpose |
|---|---|---|
| `TOTALSEG_HOME_DIR` | `.../totalseg-home` | TotalSegmentator model/cache root |
| `nnUNet_raw` | `.../nnunet-raw` | nnU-Net raw-data environment |
| `nnUNet_preprocessed` | `.../nnunet-preprocessed` | nnU-Net preprocessing cache |
| `nnUNet_results` | `.../nnunet-results` | nnU-Net model results environment |
| `TOTALSEG_DISABLE_USAGE_REPORTING` | `1` by default if unset | Disables optional TotalSegmentator usage reporting in this process |

The factory creates these directories owner-only. Export and inference stages configure the cache variables from the experiment manifest; commands such as `init` do not need them. Manually exporting them is only necessary when running a separate upstream tool outside the factory. Do not point them into the repository or a shared/public directory. Do not treat disabling usage reporting as a privacy guarantee for other network or logging behavior; use the pinned local workflow and inspect its code.

The standalone Rust smoke test uses the runtime binary selected by the pinned `ort` dependency. It does not read the Python cache variables. Its model, input, reference output, sidecar, manifest, and evidence paths must be private and outside the repository.

## 6. Reproducing the Workflow Safely

The commands below describe the workflow's interfaces, not permission to skip its gates. The `init` command creates a manifest with `TBD`, pending, and unapproved fields. It is not ready for export. Have the responsible reviewer complete the manifest from verified local model/fixture evidence; never invent approval fields to make a command pass.

### 6.1 Create a private run skeleton

Choose a new experiment ID for every materially different model, exporter, opset, runtime, fixture corpus, or threshold set. Existing IDs and outputs are not overwritten.

```bash
export EXPERIMENT_ID="totalseg-organ-training-20261010-r1"
rtk uv run --locked --project tools/model-factory/totalseg-onnx phase0 init \
  --experiment "$EXPERIMENT_ID" \
  --output-root "$RADASSIST_FACTORY_ROOT"
export RUN_ROOT="$RADASSIST_FACTORY_ROOT/runs/$EXPERIMENT_ID"
```

The generated manifest is at `$RUN_ROOT/experiment.json`. Complete it only after confirming the checkpoint, plans, class map, folds, license, fixture approvals, de-identification/suitability review, and numeric thresholds. Keep this file private because it contains local paths and evidence identifiers. Hash identity fields must match the actual assets.

### 6.2 Inspect and export the pinned model

Locate the desired model directory using the pinned TotalSegmentator release's supported model/task discovery. Do not guess from an old path. Hash and inspect that folder, then inspect the report before using its values in the experiment manifest:

```bash
export MODEL_DIR="/absolute/private/path/to/verified/model-folder"
rtk uv run --locked --project tools/model-factory/totalseg-onnx phase0 inspect-model \
  --model-dir "$MODEL_DIR" \
  --output "$RUN_ROOT/model-inspection.json"
```

After the manifest's model identity and human review gates are complete, export:

```bash
rtk uv run --locked --project tools/model-factory/totalseg-onnx phase0 export \
  --experiment "$RUN_ROOT/experiment.json" \
  --output-dir "$RUN_ROOT/onnx"
```

Expected private artifacts include `model.fp32.onnx` and `export-sidecar.json`. Inspect tensor names/shapes, dtypes, hashes, opset, graph checker status, and package versions. Do not commit the graph or checkpoint.

### 6.3 Real-patch and volume stages (only after approval)

These stages require an authorized, human-reviewed suitable CT NIfTI fixture outside Git and thresholds approved before holdout evaluation. De-identification metadata cleaning alone is not enough. Use the exact command help and the companion runbook for manifest fields, generated paths, and stage evidence:

```bash
rtk uv run --locked --project tools/model-factory/totalseg-onnx phase0 extract-patches \
  --experiment "$RUN_ROOT/experiment.json" --limit 24
rtk uv run --locked --project tools/model-factory/totalseg-onnx phase0 tensor-parity \
  --experiment "$RUN_ROOT/experiment.json" \
  --onnx "$RUN_ROOT/onnx/model.fp32.onnx" \
  --output "$RUN_ROOT/evidence/python-parity.json"
```

Patch extraction uses the pinned nnU-Net preprocessing; it is not arbitrary NIfTI slicing. The factory records tensor hashes and rejects missing approval, mismatched model identity, or unapproved thresholds. Volume parity also exercises geometry, preprocessing, sliding windows, and label export, so it is a separate and more demanding stage. Read the current `phase0 volume-run --help` and runbook before running `volume-run`, `volume-compare`, `benchmark`, `report`, or final `validate`.

For reference, the exact CLI contracts are:

```bash
rtk uv run --locked --project tools/model-factory/totalseg-onnx phase0 volume-run \
  --experiment "$RUN_ROOT/experiment.json" \
  --onnx "$RUN_ROOT/onnx/model.fp32.onnx" \
  --output "$RUN_ROOT/evidence/volume-parity.json"

rtk uv run --locked --project tools/model-factory/totalseg-onnx phase0 volume-compare \
  --experiment "$RUN_ROOT/experiment.json" \
  --reference "/absolute/private/path/to/reference-labelmap.nii.gz" \
  --candidate "/absolute/private/path/to/candidate-labelmap.nii.gz" \
  --output "$RUN_ROOT/evidence/volume-compare.json"

rtk uv run --locked --project tools/model-factory/totalseg-onnx phase0 benchmark \
  --experiment "$RUN_ROOT/experiment.json" \
  --onnx "$RUN_ROOT/onnx/model.fp32.onnx" \
  --parity-report "$RUN_ROOT/evidence/python-parity.json" \
  --output "$RUN_ROOT/evidence/benchmark.json" \
  --repetitions 30

rtk uv run --locked --project tools/model-factory/totalseg-onnx phase0 report \
  --experiment "$RUN_ROOT/experiment.json" \
  --output "$RUN_ROOT/evidence/redacted-report.json"
rtk uv run --locked --project tools/model-factory/totalseg-onnx phase0 validate \
  --experiment "$RUN_ROOT/experiment.json"
```

These are syntax examples, not evidence that those stages passed. `report` writes an `incomplete` summary and exits with status 2 when required evidence is not passed; `validate` fails until all required stage reports, identities, fixture reviews, and approved thresholds agree. The report is redacted by the tool, but inspect it before sharing.

### 6.4 Run the recorded synthetic Rust parity test

The following opt-in test reproduces the narrow synthetic check against the private run from the 2026-10-10 evidence. It is ignored by the normal suite because the graph and Python reference output are local assets. Use the exact existing run if it is still present; otherwise follow the runbook's setup to create equivalent private artifacts and use that run's path.

```bash
PHASE0_SYNTHETIC_RUN="$HOME/.local/share/radassist-model-factory/runs/totalseg-organ-export-20261010-r1" \
rtk proxy cargo test \
  --manifest-path tools/model-factory/totalseg-onnx/ort-smoke/Cargo.toml \
  --locked exported_organ_graph_synthetic_python_rust_parity \
  -- --ignored --nocapture
```

The test checks experiment/model/graph identity, creates a zero-valued Rust tensor with shape `[1,1,128,128,128]`, runs CPU ONNX Runtime, and compares its logits to the private Python ONNX Runtime `.npy` result. Python tensor parity is a separate comparison of PyTorch and Python ONNX Runtime on the same approved patch; the Rust test does not load PyTorch or call it directly. The input and output hashes bind the separately produced Rust and Python evidence to the same artifacts. The documented diagnostic error bound is `1e-3`; the recorded error was `0.0`. This input is synthetic and degenerate. Passing does not mean CT data was read or that any organ segmentation is correct.

The Rust executable also has a direct evidence mode whose argument contract is:

```text
totalseg-onnx-ort-smoke <experiment.json> <model.onnx> <input.npy> \
  <python-output.npy> <export-sidecar.json> <python-parity.json> <evidence.json>
```

It rejects repository-local asset paths, checks model and graph SHA-256 identities and prior gates, requires the patch to be in the approved corpus, requests `CPUExecutionProvider`, checks finite output and tensor shape, compares Python/Rust logits and per-channel argmax metrics, and verifies malformed shape/dtype inputs are rejected. Use it only with reviewed evidence; do not manufacture an approved manifest for a demo.

## 7. What Rust Is Doing

Rust is the host language for this smoke harness, not a model converter. The harness calls the `ort` crate, which creates an ONNX Runtime session from `model.fp32.onnx`, binds the named `image` tensor, runs it, and extracts the named output as `float32`. `ndarray` represents arrays, `ndarray-npy` reads/writes `.npy` test tensors, `serde_json` reads evidence, and `sha2` verifies file identities.

Inputs and outputs must contain finite numbers. The harness rejects NaN and infinity rather than allowing them to propagate into an apparently valid mask or metric. If preprocessing creates a non-finite value, locate its source; do not silently replace or clip it unless that behavior is part of the pinned model's documented preprocessing contract.

Conceptually, the flow is:

```text
private manifest + ONNX file + fixed tensor
    -> verify approval and SHA-256 identities
    -> create CPU ONNX Runtime session
    -> bind image tensor (float32, [1,1,128,128,128])
    -> execute graph
    -> inspect logits (float32, [1,25,128,128,128])
    -> compare against Python output and write private evidence
```

Session creation can be expensive, so a product runtime would normally reuse a session instead of creating one per patch. Production integration also needs memory budgeting, cancellation, worker/process boundaries, runtime binary packaging, supported OS/CPU matrix, thread settings, and error handling. The current smoke harness does not decide those application architecture questions.

CPU execution is intentional for the baseline. A GPU provider is a different experiment: record the provider actually selected, runtime build, device and driver, unsupported-node fallback, and new parity results. A runtime accepting a graph is not proof every node ran on the requested accelerator.

The Python patch-parity stage queries `session.get_providers()` and rejects a non-CPU provider. The Rust harness configures only `CPUExecutionProvider` and records that configured provider in its evidence, but does not collect per-node placement telemetry. This harness is a CPU smoke test, not evidence of GPU provider selection or fallback behavior.

## 8. Evidence, Privacy, and Safety Rules

- Keep medical images, de-identified images, patches, model weights, model outputs, and local manifests outside Git unless a separate asset policy explicitly approves an artifact.
- Treat hashes as evidence identifiers, not de-identification. Store fixture hashes and paths in the private run record unless a redaction review approves publication.
- Metadata scrubbing does not remove burned-in text from pixels, validate image suitability, prove series completeness, or establish anonymity. Human pixel review is still required.
- Do not infer testing eligibility from a successful de-identification command. The command does not set fixture authorization, human review, clinical suitability, or numeric thresholds.
- Model software licensing and weight licensing can differ. Review the exact checkpoint and intended redistribution/use separately. A local export does not grant redistribution rights.
- Never describe numerical parity to the source implementation as clinical accuracy. Independent ground truth and intended-use validation are separate work.
- Do not put patient identifiers or image-derived content in terminal output, test names, issue text, screenshots, or committed evidence.

The current fixture status and precise gates are recorded in the runbook and its de-identification guide. If approvals are missing, stop before real-image inference and use only synthetic tensors or non-medical test graphs.

## 9. Troubleshooting Guide

| Symptom | Likely cause | First check |
|---|---|---|
| `uv sync --locked` rejects the environment | Wrong Python version, platform incompatibility, or changed lock | Verify Python is 3.12.x and the lock was not edited; use the pinned environment. |
| Export refuses model identity | Model path, checkpoint, plans, class map, fold, or package pins differ from the manifest | Re-run `inspect-model`; compare SHA-256 values and exact versions. Do not weaken identity checks. |
| Export fails on an ONNX operator | Exporter lowering or unsupported operation | Record the exact operator and inspect a standard ONNX decomposition; do not add custom runtime operators as a first response. |
| Runtime says input name/shape/dtype is wrong | Rust/Python tensor contract drift or wrong NPY file | Read the sidecar; confirm `image`, `float32`, `[1,1,D,H,W]`, contiguity, and hash. |
| Output shape differs | Wrong checkpoint/configuration, output head, patch, or graph | Compare model metadata and sidecar; verify final logits and output class count. |
| Python/Rust logits differ | Different graph, input bytes, runtime kernels/build, or serialization | Match graph and input hashes first; record both ORT versions and compare raw logits before argmax. |
| Patch parity passes but volume result differs | Preprocessing, geometry, resampling, mirroring, blending, postprocessing, or inverse transform | Compare intermediate geometry and probability tensors; include laterality checks. |
| No inference command proceeds | A gate is missing or unapproved | Read the error and complete real review evidence; do not set approval flags merely to proceed. |

Do not start by relaxing tolerances. Identify the failing layer and keep the original failing evidence in its immutable run.

## 10. Engineer Training Sequence

Use this sequence for onboarding. Engineers should be able to explain the result, not merely copy a command.

1. Identify architecture, checkpoint, preprocessing plan, class map, graph, and runtime in the repository and say what each contributes.
2. Explain the fixed input/output tensor contract and the meaning of logits versus class labels.
3. Compare ONNX and GGUF by workload and runtime, then explain why this U-Net uses ONNX while a llama.cpp-oriented language model may use GGUF.
4. Run the lock check, unit suites, and `phase0 --help`; verify the exact dependency pins and runtime versions.
5. Inspect the experiment manifest and explain why `init` creates a blocked template rather than a ready-to-run manifest.
6. Review the dated evidence and state precisely what the synthetic parity result proves and does not prove.
7. Trace how real patch parity differs from volume-pipeline parity, including geometry and laterality risks.
8. Explain why no CT/MRI inference is permitted until human fixture and threshold gates are approved.

Suggested discussion prompt: "If the Rust and Python outputs match exactly on the zero patch, what new evidence would we need before claiming the full CT pipeline matches?" A complete answer should mention representative approved inputs, pinned preprocessing, volume geometry, sliding windows/blending, class mapping, native-grid outputs, held-out thresholds, and human review.

## 11. Further Reading

- [ONNX documentation](https://onnx.ai/onnx/)
- [ONNX Runtime documentation](https://onnxruntime.ai/docs/)
- [PyTorch ONNX export](https://docs.pytorch.org/docs/stable/onnx.html)
- [GGUF specification in llama.cpp](https://github.com/ggml-org/ggml/blob/master/docs/gguf.md)
- [llama.cpp documentation](https://github.com/ggml-org/llama.cpp/tree/master/docs)
- [TotalSegmentator repository](https://github.com/wasserth/TotalSegmentator)
- [nnU-Net v2 repository](https://github.com/MIC-DKFZ/nnUNet)
- [RadAssist AI model integration architecture](../AI-MODEL-INTEGRATION-ARCHITECTURE.md)
- [MedGemma integration design](../medgemma-integration-design.md)
- [DICOM de-identification operator guide](../DICOM-DEIDENTIFICATION-USER-GUIDE.md)

## 12. Maintenance and Change Log

When the model, lockfile, exporter, opset, runtime, Rust binding, or evidence changes, update the relevant section and runbook together. Include the verification date, exact versions, hashes where appropriate, commands run, and what the new evidence does and does not prove. Keep previous experiment evidence immutable; add a new dated entry instead of rewriting history. Never promote synthetic results to real-input results by changing wording alone.

| Version | Date | Change |
|---|---|---|

## 13. Deeper Study: Runtime Contracts And Failure Analysis

After reproducing the existing evidence, use the following sequence to build a stronger mental model of native inference:

1. **Inspect the model contract.** Record each input/output name, rank, dynamic/fixed dimensions, element type, axis meaning, and whether outputs are logits, probabilities, or labels. Assert these at the Rust boundary.
2. **Trace preprocessing.** For a small synthetic input, save intermediate stages and verify units, layout, normalization, resampling, and coordinate transforms independently of the ONNX graph.
3. **Compare numerically before discretizing.** Compare raw outputs first (max absolute error, mean absolute error, and optionally relative error). Argmax can hide score drift; Dice can hide errors on absent/small classes.
4. **Test boundary behavior.** Include non-contiguous arrays where supported, invalid dimensions, NaN/Inf, empty masks, extreme values, cancellation, allocation failure, and output-write failure.
5. **Measure the whole lifecycle.** Separate model load, preprocessing, inference, postprocessing, disk I/O, and cold/warm behavior. Record p50/p95, peak process memory, thread count, execution provider, and exact device/runtime versions.
6. **Optimize only after a reference pass.** Compare execution providers, thread settings, graph optimizations, and reduced precision against a pinned baseline. Re-run representative parity and performance tests after every change.

### Common Numerical Metrics

For reference tensor `r` and candidate tensor `c`:

```text
max_abs_error  = max(abs(r - c))
mean_abs_error = mean(abs(r - c))
relative_error = abs(r - c) / max(abs(r), epsilon)
```

No single tolerance is universal. Choose tolerances before looking at held-out results, consider class/voxel-level effects, and report both numeric and task-level measures. Relative error is unstable near zero, so define its denominator policy explicitly.

### Portability Checklist

- Pin ONNX opset, exporter, runtime, Rust binding, and execution provider.
- Verify operator coverage and graph metadata on each target platform.
- Treat a successful CPU run as evidence only for that tested configuration.
- Distinguish graph compatibility from package/distribution and model-weight licensing.
- Preserve source-image affine, spacing, orientation, and label mapping through postprocessing.
- Keep an immutable experiment record; append new dated results instead of silently changing historical evidence.

Use the [AI Engineering Study Roadmap](STUDY_ROADMAP.md) to place this guide within the beginner-to-deployment sequence.
| 1.0 | 2026-10-10 | Initial engineer guide for the pinned TotalSegmentator ONNX export and isolated Rust CPU smoke proof. |