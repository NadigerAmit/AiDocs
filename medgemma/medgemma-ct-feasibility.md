# Offline MedGemma CT Feasibility: Prerequisite Audit

**Observed:** 2026-10-11
**Status:** Native bootstrap verified on Linux, with Android arm64 cross-build; synthetic vision smoke passed on 2026-10-12; CT inference remains unqualified.
**Scope:** First work package of the [feasibility plan](../../_bmad-output/specs/spec-ai-workflows/feasibility-plan.md). The [design](../medgemma-integration-design.md) and [implementation spec](../../_bmad-output/specs/spec-ai-workflows/SPEC.md) remain unchanged in intent.

**Newcomer entry point:** Start with the [MedGemma Local Vision Guide](medgemma-local-vision-guide.md); this file retains the detailed audit evidence and dated observations.

The audit sections below are historical observations before source acquisition. The appended Native Runtime Bootstrap section supersedes audit-era absence of a local runtime/reference; it does not close model/projector, CT parity, fixture or device qualification gates.

## Local Model Candidate

Read-only inspection found `~/Downloads/medgemma-1.5-4b-it-Q4_K_M.gguf`.

| Observed fact | Value |
| --- | --- |
| File size | 2,489,894,976 bytes, about 2.319 GiB |
| Fixed-header magic/version | GGUF / 3 |
| Declared tensor count | 444 |
| Declared metadata entry count | 52 |
| SHA-256 | b31becdf4f39561800505514cce67681604fe449d04dd35c8c92fd7848c6d7bd |

Same-descriptor recheck: 2026-10-11T14:38:33.983Z through 2026-10-11T14:38:37.455Z, Node v20.20.2. Size, mtime, ctime, inode and device were unchanged before/after the read, and the fingerprint matched the earlier sha256sum result. This is best-effort change detection, not an immutable snapshot or trusted artifact-signature verification.

The filename, fixed header and digest identify a candidate, not validated model identity or provenance. Tensor metadata, model vocabulary/template and payload integrity have not been validated. No runtime session was created, no weights were imported and no inference ran. Applicable terms acceptance and trusted model/projector pairing must be established before execution.

No matching `mmproj` artifact or official CT notebook was found by the bounded filename scan. This does not prove they are absent from the machine or unavailable elsewhere. A text-generation GGUF alone does not establish image-capable execution; the qualified runtime must provide the correctly paired vision components and processor.

## Available Hardware

| Target | Observed envelope | Qualification |
| --- | --- | --- |
| Linux host | x86_64; Intel Core Ultra 7 268V; 8 logical CPUs; 31.04 GiB total RAM; 9.58 GiB free at timestamped recheck | Not benchmarked. Free RAM is transient, not an admission threshold. |
| Connected Android | Moto G05; arm64-v8a; API 35; 7.54 GiB total RAM; 4.08 GiB available at timestamped recheck | Not benchmarked. These facts do not establish that CT inference fits or is responsive. |
| Windows/macOS | Not inspected | Unqualified; do not infer feasibility from Linux. |

`nvidia-smi` was unavailable; this is not evidence that the host has no GPU. Integrated GPU capabilities were not inspected. Android inference was not installed or started. Device serials are deliberately excluded from this report.

Hardware recheck timestamp: 2026-10-11T14:38:38.673Z. Linux facts use Node v20.20.2 `os` APIs. Android facts use ADB 1.0.41 / 36.0.0-13206524, `getprop` and `/proc/meminfo`: MemTotal 7,911,456 KiB, MemAvailable 4,282,552 KiB. Linux procfs labels these units kB but uses 1024-byte units; divide by 1024 squared for GiB. Earlier untimestamped free-memory observations differed, as expected.

None of `llama-cli`, `llama-mtmd-cli`, `llama-server`, or `llama-mtmd-debug` was found on `PATH`. Runtime builds outside `PATH` were not comprehensively inventoried. The existing segmentation worker is not a qualified MedGemma runtime.

## Fixture Gates

- Repository `mpr-phantom` contains three synthetic CT slices forming a 3 x 2 x 3 volume. These are suitable for geometry/snapshot regression, not anatomical interpretation, diagnostic accuracy or realistic memory/latency benchmarking.
- The existing [deidentification implementation record](../../_bmad-output/implementation-artifacts/spec-ct-mri-fixture-deidentification.md) documents real external CT copies with `review_required` and `testing_eligible=false`. No permission to run inference is inferred from metadata cleaning.
- The reviewed latest documented CT copies comprise two 35-slice stacks and four localizers. Current private manifest contents and images were not opened by this audit. Pixel/overlay review, decoded validity, reference integrity and series suitability still require approval/evidence.
- No real CT or MRI images were opened, converted, uploaded or analyzed. No fixture gate was changed. MRI remains outside the initial CT spike.

## Blocking Prerequisites

| Gate | Needed evidence |
| --- | --- |
| Model provenance and image components | Trusted identity/license acknowledgement for the local candidate and correctly paired vision/projector/tokenizer artifacts, or another qualified local bundle. |
| Official preprocessing | Pinned reference assets and tokenizer comparison are now recorded. Strict image parity is blocked: the notebook's fast processor differs from native GGUF preprocessing. |
| Runtime | Pinned Linux runtime passed generated one- and two-slice offline checks. Conversion/quantization parity and offline loading with caches absent remain open. |
| Evaluation inputs | Approved de-identified anatomical CT fixtures with independent reference observations, including distinct-lesion and cross-series duplicate/conflict examples. |
| Device budgets | Agreed measurable admission/latency/memory/cancellation criteria and platform benchmarks; Android RAM alone is not a pass/fail verdict. |

These are missing qualification inputs, not proof that MedGemma fails. No server fallback, automatic download or bypass of privacy review is authorized.

Closure requires recorded evidence, not an updated status by assertion. This qualification slice records exact reference text-token parity and successful synthetic host runtime checks, but the processor gate remains blocked on fast-resizer tensor differences. Trusted local bundle provenance/terms, approved anatomical fixtures, privacy/suitability review, conversion/quantization parity, and agreed per-platform benchmark criteria remain pending. These describe required evidence, not assigned people or completed approvals.

## Processor And Runtime Qualification (2026-10-12)

The synthetic-only harness is [processor_reference_parity.py](../../tools/medgemma-bootstrap/processor_reference_parity.py). It used the pinned Hugging Face repository [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it) at revision `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`. The exact ten-file processor/tokenizer/config allowlist and byte hashes are in the checked-in [processor-reference-manifest.json](../../tools/medgemma-bootstrap/processor-reference-manifest.json); the downloaded files themselves are in ignored `target/medgemma-bootstrap/processor-reference/`. The earlier documentation incorrectly placed the manifest under `target/`. The newcomer guide's [Reproduce The Qualification](medgemma-local-vision-guide.md#reproduce-the-qualification) section lists the exact `hf download` command, environment setup, per-step purpose, and runner invocations. No Transformers model weights were requested or fetched. The isolated environment used Python 3.12.3, Transformers 4.57.6, Tokenizers 0.22.2, Torch 2.5.1+cpu, torchvision 0.20.1+cpu, Pillow 12.3.0 and NumPy 1.26.4.

| Check | Observed result | Gate |
| --- | --- | --- |
| Same rendered chat-template text | HF/native token IDs exactly match, 15/15 | Passed for this deterministic prompt |
| Native vs reference fast processor, synthetic PNG | Same 896 x 896 x 3 tensor shape; max absolute difference `0.0078431368`, mean `0.0000024076`, 869,467 / 2,408,448 values differ | **Blocked** |
| Native vs reference fast processor, synthetic JPEG | Same shape; max absolute difference `0.0156863332`, mean `0.0005316096`, 968,785 / 2,408,448 values differ | **Blocked** |
| HF slow processor diagnostic vs native, synthetic PNG | Exact normalized-float match; max and mean difference 0 | Diagnostic only; does not replace the notebook's `use_fast=True` reference |
| HF fast vs native, deterministic synthetic RGB PNG 512 x 512 | Same 896 x 896 x 3 shape; max absolute difference `0.0078431964`, mean `0.0000886595`, 892,860 / 2,408,448 values differ | **Blocked**; input is upsampled to 896 x 896 |
| HF fast vs native, deterministic synthetic RGB PNG 1024 x 1024 | Same 896 x 896 x 3 shape; max absolute difference `0.0078431955`, mean `0.0000057424`, 881,260 / 2,408,448 values differ | **Blocked**; input is downsampled to 896 x 896 |
| Offline native runtime, one generated DICOM-derived slice | Exit 0; 1 image decoded and 1 visual-token group observed; 1-token bounded generation; 68.12 s | Passed for Linux synthetic runtime plumbing |
| Offline native runtime, two generated DICOM-derived slices | Exit 0; 2 images decoded and 2 visual-token groups observed; 1-token bounded generation; 132.34 s | Passed for Linux synthetic runtime plumbing |

The native runtime is llama.cpp revision `5c8854fd88319125125dc4b836c98b1b215b0f23` with the locally hash-verified GGUF pair. The final child-process peak-RSS high-water observations were 5,201,672 KiB for the single-slice case and 5,224,416 KiB for the two-slice case. These are exploratory host observations, not isolated per-case memory guarantees or approved budgets. `--offline` was used; network traffic was not independently monitored or denied.

The dimension tests isolate image resizing from DICOM/JPEG preparation by generating deterministic RGB pixels directly, then feeding the exact same PNG to HF `Gemma3ImageProcessorFast` and the native probe. The 512 input is enlarged (scale 1.75); the 1024 input is reduced (scale 0.875). A mismatch in both directions makes a size-specific explanation less likely, but does not yet identify the precise sampling or rounding operation responsible. The current evidence is consistent with implementation-level resize differences and is not proof of their sole cause.

The stage-level comparison now records the pre-rescale, pre-normalization `uint8` RGB buffers directly from the HF fast resize path and by reversing the native fixed-size path's normalization. The probe reads the normalization arrays from the actual loaded GGUF projector; native means and standard deviations are `[0.5, 0.5, 0.5]`, matching the HF fast processor. Before recovery, the inverse was checked across all 256 uint8 values per channel; each recovered native image also re-normalized bit-for-bit to the emitted native float tensor. Both outputs have shape 896 x 896 x 3. These hashes and byte mismatch counts locate the observed differences before normalization, in the resize output; they do not identify the precise implementation-level cause.

| Synthetic RGB input | Input SHA-256 | HF fast resized RGB SHA-256 | Native resized RGB SHA-256 | Different byte values | Max / mean absolute difference |
| --- | --- | --- | --- | ---: | ---: |
| 512 x 512 | `02949aa03e8f7d8bd634cbd951a69f7749937ada1c4231dd6f35edeee169a0a9` | `2daa0fec10f41320b6779306ebf33834c52d0ecd7624a7fb5fe786888b611784` | `5d90db4d31f137e9a8e0ad7f8ab135e91dc9b7b42dd0ed725361a9b7b6697477` | 27,222 / 2,408,448 | 1 / 0.0113027142 |
| 1024 x 1024 | `9d0bc5408e62e7a8f113c024c1d03fc0e81692ec5d13af342e5f6e9256b59052` | `084d1cfc9ba4ff3257e32871ece02cf2659b537113a1648318f20f3d9a675e80` | `3c111dad9f88b6d631f9fe0cd5252aa05f9350216933ac83b3af3b9b48f318fc` | 1,760 / 2,408,448 | 1 / 0.0007307610 |

An expanded synthetic comparison on 2026-10-13 included DICOM-derived PNG/JPEG inputs, square RGB PNGs, and non-square RGB PNGs. HF slow matched native for the DICOM-derived PNG and the square 512, 896, and 1024 inputs, but failed for the DICOM-derived JPEG (137,502 / 2,408,448 normalized values differ; max `0.0156863`) and both non-square inputs (over 2.08 million of 2.4 million RGB bytes differ; max difference 255). Fast/native show similarly large non-square differences. **HF slow is therefore not qualified as a general substitute.** The result keeps the reference unchanged and overall gate blocked; it does not yet isolate JPEG decoding versus resize or the exact non-square aspect-ratio behavior.

The latest full run at 2026-10-13T01:19:30Z again passed both synthetic offline runtime cases: one slice in 68.12 s (5,201,672 KiB peak child RSS) and two slices in 132.34 s (5,224,416 KiB). These remain exploratory observations without approved performance budgets. The full machine-readable result includes the fast/slow, JPEG, and non-square records; the `--skip-runtime` result remains separate and does not overwrite it.

Here, processor parity means equality of normalized image tensor values immediately before vision encoding, not equality of model answers. Since the GGUF route preprocesses the image itself, tensor differences mean we cannot claim it supplies the notebook's selected fast-processor inputs to the vision projector. The differences may affect vision features and downstream responses; their response-level or clinical impact was not tested or quantified. This is why integration remains blocked for an HF-equivalence claim. A different processor path or a numeric tolerance would require an explicit qualification decision and behavioral validation; the exact-match slow path remains diagnostic only.

Re-run using the already provisioned ignored environment and local artifacts:

```bash
rtk proxy target/medgemma-bootstrap/processor-venv/bin/python tools/medgemma-bootstrap/processor_reference_parity.py
```

The machine-readable output is `target/medgemma-bootstrap/processor-reference-result.json`. The harness exits with a blocked status when strict image parity fails. **Overall processor/runtime qualification remains blocked**; do not integrate on the claim of processor parity or silently substitute the diagnostic slow processor. The existing synthetic preprocessing and vision-pairing baselines passed separately (8/8 and 2/2 tests). No patient data, app code, Android runtime, or model weights were involved.

For a quick resize/tokenizer-only rerun without model generation, use the same command with `--skip-runtime`; it writes a separate `target/medgemma-bootstrap/processor-reference-resize-result.json` so the last full runtime result is preserved. Both the focused and full runs return status 2 while strict parity is blocked. The `--skip-runtime` result has no runtime evidence and must not be treated as a completed full qualification.

## Synthetic Geometry Verification

Executed from the existing workspace, without dependency downloads or lockfile changes:

```bash
rtk proxy cargo test --offline --locked --manifest-path "$HOME/MedicalWork/Cargo.toml" -p mpr-core --lib
```

Result: **27 passed, 0 failed, 2 ignored**. The ignored checks require a native GPU/software Vulkan adapter. Passing tests cover bounded-volume, source provenance, geometry, snapshot grid/voxel order and phantom MPR behavior. They do not execute the desktop DICOM worker, MedGemma preprocessing, anatomical inference or a GPU benchmark.

## Machine-Checkable Audit Summary

This summary contains no patient data and no device serial. It is a snapshot of the audit, not a live readiness check.

```json
{
  "schemaVersion": 1,
  "status": "blocked",
  "inferenceExecuted": false,
  "modelImported": false,
  "networkOperationRequested": false,
  "networkIsolationVerified": false,
  "candidate": {
    "filename": "medgemma-1.5-4b-it-Q4_K_M.gguf",
    "sizeBytes": 2489894976,
    "magic": "GGUF",
    "ggufVersion": 3,
    "tensorCount": 444,
    "metadataCount": 52,
    "sha256": "b31becdf4f39561800505514cce67681604fe449d04dd35c8c92fd7848c6d7bd",
    "compatibilityVerified": false,
    "statUnchangedAcrossRead": true,
    "matchesPriorFingerprint": true,
    "recheckedUtc": "2026-10-11T14:38:37.455Z"
  },
  "projectorFoundInScan": false,
  "ctReferenceFoundInScan": false,
  "runtimeFoundOnPath": false,
  "realCtTestingApproved": false,
  "linuxBenchmarked": false,
  "androidBenchmarked": false,
  "windowsBenchmarked": false,
  "macosBenchmarked": false,
  "geometryTests": { "passed": 27, "failed": 0, "ignored": 2 }
}
```

## Scan Scope And Next Action

Filename-only traversal examined `~/MedicalWork`, `~/.cache/huggingface/hub`, `~/.local/share/radassist-model-factory`, and `~/Downloads`. Root directories have depth zero; directories up to eight edges below a root are visited, including files directly in those directories. It skipped `.git`, `target`, `build`, `.cxx`, `node_modules`, `.venv`, and `.venv-1`, and did not follow directory symlinks. All four roots existed and traversal reported zero directory-read errors. Missing roots/error counts must be recorded in future scans, not interpreted as no matching artifacts.

Reproduce the filename-only scan with the same matching/depth rules; it prints metadata only, not file contents:

```bash
rtk proxy node <<'NODE'
const fs = require('fs'), path = require('path'), os = require('os');
const home = os.homedir();
const roots = ['MedicalWork', '.cache/huggingface/hub', '.local/share/radassist-model-factory', 'Downloads'];
const skip = new Set(['.git', 'target', 'build', '.cxx', 'node_modules', '.venv', '.venv-1']);
const pattern = /medgemma|mmproj|high_dimensional_ct|\.(gguf|safetensors)$/i;
for (const relative of roots) {
  const root = path.join(home, relative), result = { root: '~/' + relative, exists: fs.existsSync(root), matches: [], errors: 0 };
  const queue = result.exists ? [[root, 0]] : [];
  while (queue.length) {
    const [dir, depth] = queue.pop();
    let entries;
    try { entries = fs.readdirSync(dir, { withFileTypes: true }); } catch { result.errors++; continue; }
    for (const entry of entries) {
      const full = path.join(dir, entry.name);
      if (entry.isDirectory()) { if (depth < 8 && !skip.has(entry.name)) queue.push([full, depth + 1]); }
      else if (pattern.test(entry.name)) result.matches.push(path.relative(home, full));
    }
  }
  console.log(JSON.stringify(result));
}
NODE
```

Header and fingerprint recheck recipe; numeric fields are little-endian at offsets 4, 8 and 16 after the GGUF magic:

```bash
rtk proxy node <<'NODE'
const fs = require('fs'), crypto = require('crypto'), path = require('path'), os = require('os');
const file = path.join(os.homedir(), 'Downloads/medgemma-1.5-4b-it-Q4_K_M.gguf');
const fd = fs.openSync(file, 'r'), before = fs.fstatSync(fd), header = Buffer.alloc(24);
if (fs.readSync(fd, header, 0, 24, 0) !== 24) { fs.closeSync(fd); throw Error('Truncated GGUF header'); }
const digest = crypto.createHash('sha256'), startedUtc = new Date().toISOString();
const stream = fs.createReadStream(file, { fd, start: 0, autoClose: false });
stream.on('data', chunk => digest.update(chunk));
stream.on('error', () => { fs.closeSync(fd); console.error('Artifact read failed'); process.exitCode = 1; });
stream.on('end', () => {
  const after = fs.fstatSync(fd), unchanged = ['size', 'mtimeMs', 'ctimeMs', 'ino', 'dev'].every(key => before[key] === after[key]);
  fs.closeSync(fd);
  console.log(JSON.stringify({ startedUtc, finishedUtc: new Date().toISOString(), nodeVersion: process.version,
    sizeBytes: before.size, magic: header.toString('ascii', 0, 4), version: header.readUInt32LE(4),
    tensorCount: header.readBigUInt64LE(8).toString(), metadataCount: header.readBigUInt64LE(16).toString(),
    sha256: digest.digest('hex'), statUnchangedAcrossRead: unchanged }));
  if (!unchanged) process.exitCode = 1;
});
NODE
```

Hardware collection uses `os.platform()`, `os.arch()`, `os.cpus()`, `os.totalmem()` and `os.freemem()` alongside a UTC ISO timestamp. Android reproduction commands:

```bash
rtk proxy adb version
rtk proxy adb shell getprop ro.product.model
rtk proxy adb shell getprop ro.product.cpu.abi
rtk proxy adb shell getprop ro.build.version.sdk
rtk proxy adb shell cat /proc/meminfo
```

The executed audit commands requested no network activity or downloads; `cargo` used `--offline --locked`. Network traffic was not monitored or denied during this audit, so network isolation is unverified. These observations cannot qualify the runtime offline/privacy gate.

No intermediate audit files were written outside these Markdown records; command output is retained in tool-session history. Initial discovery output included account-qualified paths and an ADB serial, so session transcripts require redaction before external sharing. This report uses portable home paths and excludes the serial; future filename outputs must also be reviewed before export. Unrelated cached model weights and patient image contents were not opened.

Next: resolve the fast-resizer mismatch or obtain explicit approval to qualify a different reference path, then resolve remaining provenance, fixture-approval and platform-budget prerequisites. Synthetic geometry and host runtime checks do not establish image-analysis feasibility.

## Native Runtime Bootstrap

Implemented from the [frozen bootstrap spec](../../_bmad-output/implementation-artifacts/spec-medgemma-native-runtime-bootstrap.md) on 2026-10-11. This is isolated developer tooling, not an application worker or successful image read. No models, patient images, viewer, dictation or report code were changed. No notebook cells, model sessions, inference server or Android deployment were run.

### Pinned Sources And Outputs

The [source manifest](../../tools/medgemma-bootstrap/sources.json) is a source-revision record, not a model-distribution or model-pairing manifest.

| Source | Approved URL | Full verified revision |
| --- | --- | --- |
| llama.cpp | https://github.com/ggml-org/llama.cpp.git | `5c8854fd88319125125dc4b836c98b1b215b0f23` |
| Official MedGemma CT reference | https://github.com/google-health/medgemma.git | `a60a66024f6153496dfa9490dba12d4f1fbd092e` |

Acquired source and all outputs live under ignored `target/medgemma-bootstrap`: `sources/llama.cpp`, `sources/medgemma`, `build-host`, `build-android`, `reference/ct-code-cells.json`, `reference/ct-code-cells.py`, `manifest-host.json` and `manifest-android.json`. No vendor tree is committed. Evidence manifests include source identity, exact CMake argument arrays, tool versions, completion time, SHA-256 artifact fingerprints and whether acquisition/inference/reference execution occurred. The host manifest also stores real executable version/help output. JSON extraction retains exact code-cell content and original notebook indices; the Python inspection file contains notebook magics and is deliberately not an executable preprocessing implementation.

Notebook path: `notebooks/high_dimensional_ct_hugging_face.ipynb`. SHA-256 of its original UTF-8 bytes: `75f527afbe81c9641490a7b2223788309d37589315779129257d625ebcd7efb1`. Ten code cells were extracted, at original zero-based notebook indices `2, 5, 7, 9, 10, 12, 14, 15, 17, 19`. [Pinned upstream attribution](https://github.com/google-health/medgemma/blob/a60a66024f6153496dfa9490dba12d4f1fbd092e/notebooks/high_dimensional_ct_hugging_face.ipynb) is retained in both inspection outputs.

### Exact Verification Commands

Run from the repository root. Prerequisites are Node.js 20 or newer, Git, CMake, Ninja and a working local C/C++ toolchain. Windows requires an appropriate compiler environment with Ninja available; macOS requires its installed compiler toolchain. The optional Android target requires an explicit complete local NDK. No tool or privileged package installation is performed.

```bash
rtk proxy node --test tools/medgemma-bootstrap/bootstrap.test.mjs
rtk proxy node tools/medgemma-bootstrap/bootstrap.mjs --fetch --target host
rtk proxy node tools/medgemma-bootstrap/bootstrap.mjs --target host
rtk proxy node tools/medgemma-bootstrap/bootstrap.mjs --target android --ndk "$HOME/Android/Sdk/ndk/27.0.12077973"
rtk proxy env MEDGEMMA_BOOTSTRAP_NATIVE_EXECUTABLE="$PWD/target/medgemma-bootstrap/build-host/bin/llama-mtmd-cli" MEDGEMMA_BOOTSTRAP_ROOT="$PWD/target/medgemma-bootstrap" node --test tools/medgemma-bootstrap/bootstrap.test.mjs
```

The first synthetic-only test invocation passed before the native build; explicit integration checks are gated on the environment variables. The final full invocation ran **21 tests: 21 passed, 0 failed, 0 skipped**, including the actual built executable, exact ordered reference comparison and ELF64/AArch64 artifact/hash checks. Synthetic acquisition tests use temporary local Git repositories, not additional network requests. Tests cover full manifest validation, foreign/mismatched/attached/dirty/ignored/symlink checkouts, successful staged acquisition, first/second-source failure preservation, absent-source no-fetch behavior, missing tools/NDK, process nonzero/missing/timeout failures and structured host/Android command plans.

`--fetch` explicitly acquires only absent source objects using their full pinned revisions, then verifies detached HEAD, exact origin and a clean checkout before publishing staged sources. Existing unexpected checkouts are refused, not reset, cleaned or silently reused. No-fetch is the default; matching local source causes no Git acquisition. Configuration and build are local CMake/Ninja processes. A source-acquisition failure discards only this run's temporary staging directory. Concurrent bootstrap runs are refused by a directory lock; after an abnormal process exit, inspect for an active build before manually removing a stale `.bootstrap-lock`. No remote application execution or weight acquisition is implemented.

### Configured Options And Platform Evidence

Common CMake arguments use Ninja, `Release`, `BUILD_SHARED_LIBS=ON`, `GGML_NATIVE=OFF`, `GGML_CPU_ALL_VARIANTS=OFF`, `LLAMA_BUILD_MTMD=ON`, and disable CUDA, Metal, Vulkan, BLAS, Accelerate and OpenMP. `LLAMA_OPENSSL=OFF`, `LLAMA_SUBPROCESS=OFF`, `MTMD_VIDEO=OFF`, and tests/examples/app/server are disabled. Host common utilities/tools are enabled only to build target `llama-mtmd-cli`. Android common utilities/tools are disabled and target `mtmd` is built standalone. Both commands use `--parallel 4` by default on this host; `--jobs 1..64` is explicit and validated.

Android adds `CMAKE_TOOLCHAIN_FILE=<NDK>/build/cmake/android.toolchain.cmake`, `ANDROID_ABI=arm64-v8a`, `ANDROID_PLATFORM=android-26`, `ANDROID_STL=c++_shared`. Local NDK `27.0.12077973` (r27) supplied Clang 18.0.1. CMake detected baseline ARM FMA, not dot-product, SVE, FP16-vector or int8-matmul extensions; no host-native tuning was requested.

| Platform/check | Actual observed result | Not established |
| --- | --- | --- |
| Linux x86_64 host | Explicit fetch/build passed; final no-fetch rebuild passed at `2026-10-11T16:11:44.374Z`; real `--version` and `--help` exited successfully and help recognized `--model`, `--mmproj`, `--image` | MedGemma weight/projector compatibility, image inference, CT parity or diagnostic accuracy |
| Android arm64 cross-build on Linux | Built five ELF64 AArch64 shared libraries; final no-fetch build passed at `2026-10-11T16:11:51.930Z`; artifact hashes checked | Android loading, installation, on-device inference, memory/latency or viewer impact |
| Windows/macOS | Structured CMake argument plans tested; native hosts unavailable for this run | Native compilation, executable smoke or application feasibility |

Tools: Node v20.20.2, Git 2.43.0, CMake 3.28.3, Ninja 1.11.1, host GNU C/C++ 14.3.0. Runtime version output was `0.6.0-dev (build 1, commit 5c8854f)`, built with GNU 14.3.0 for Linux x86_64. Host executable SHA-256: `99b63fbcd400a4a806954c2d1aa95726e01afe1a0b52ddf21f3911998e89b914`; Android `libmtmd.so`: `9aa4975992af993378d6653b192197ab7433f314e666990634e6471eb7b09939`. Host evidence fingerprints the executable and shared-library files/links; Android evidence fingerprints `libmtmd.so`, `libllama.so`, `libggml.so`, `libggml-base.so`, `libggml-cpu.so`. Android packaging would additionally need the NDK's `libc++_shared.so`; packaging is outside this slice.

The first Linux build used the retired `LLAMA_CURL=OFF` option; inspection corrected it to the actual pinned upstream `LLAMA_OPENSSL=OFF`, disabled app/subprocess/video, and the corrected build and no-fetch rebuild passed. Android required the upstream standalone `LLAMA_BUILD_MTMD=ON` hook. No failed platform build was observed. Upstream emitted a non-fatal missing-ccache warning and two host disabled-video helper `noreturn` suggestions; no packages were installed to suppress them. The final Linux CPU backend uses upstream default SSE4.2/F16C/FMA/BMI2/AVX/AVX2, despite `GGML_NATIVE=OFF`: this is not an arbitrary-old-x86 redistribution guarantee. Other native hosts remain untested.

### Observed CT Reference, Not Chosen Presets

These details are inspection evidence from the pinned notebook, not a RadAssist CT processor or clinical recommendation:

| Original code cell | Observed behavior |
| --- | --- |
| 9 | Loads public DICOM metadata over authenticated HTTP, sorts by integer `InstanceNumber`, sets `MAX_SLICE=85`; when there are more slices, samples indices `int(round(i / MAX_SLICE * (N - 1)))` for `i=1..85` |
| 10 | Downloads selected DICOM instances using an ordered three-worker executor, decodes `pixel_array`, and calls `pydicom.pixels.apply_rescale` for HU |
| 12 | Clips/normalizes three RGB channels with ranges `(-1024,1024)`, `(-135,215)`, `(0,80)`; converts to float32, maps each range to 0..255, rounds then casts to uint8; constructs PIL images and a visualization GIF |
| 14 | JPEG-encodes each RGB slice with PIL defaults into a base64 image data URL; builds one user message with a medical-teaching instruction, alternating image entries and `SLICE 1..N` labels, then a liver-focused yes/no question |
| 17 | Uses `google/medgemma-1.5-4b-it`, `AutoProcessor.from_pretrained(..., use_fast=True)`, image-text model loading, bfloat16, automatic device mapping and offload buffers |
| 19 | Calls `apply_chat_template` with generation prompt enabled, no continuation of final message, tokenization/dictionary/PT tensors; moves inputs to model device/bfloat16 and generates with `do_sample=False`, `max_new_tokens=2000`; postprocesses output and removes echoed input where detected |

No explicit physical-geometry sorting or resampling is defined in these extracted cells. The notebook code does not itself specify resize dimensions, but the pinned processor config and runtime comparisons are recorded in the qualification section above. The sampling formula starts at `i=1`, not zero, and its actual rounding/source coverage must be reproduced rather than replaced with an assumed evenly-spaced endpoints policy. This liver-teaching prompt is not a whole-study reporting prompt, and the notebook's hard-coded RGB windows are not approved all-CT/anatomy presets. Authentication, pip installation, downloads, GIF display and model inference present in the cells were **not executed**.

### Bootstrap Review And Final Checks

Independent review of the bootstrap was completed. The final focused invocation with native/reference/Android artifact environment variables ran **28 tests: 28 passed, 0 failed, 0 skipped**. Hardened no-fetch Linux and Android arm64 builds also passed. Added regressions cover hidden Git index changes, configure/build/revision failures and retry, unsafe reused build contents, hard-linked inspection outputs, source-publication rollback, invalid Android outputs, atomic write cleanup and SIGTERM-resistant process timeout.

The bootstrap now rejects index flags that can hide source edits, foreign CMake source directories, escaping build symlinks and hard-linked build files. Atomic replacements avoid truncating external hard-linked inspection files. Failed source publication rolls back this run's verified published checkouts. Android success requires the complete core library set with ELF64/AArch64 validation before evidence publication.

Known alternate backends, system GGML, KleidiAI and llguidance are explicitly disabled; FetchContent disconnected controls are set. These preserve the intended local CPU baseline under ordinary cache reuse, not a sandbox against a malicious compiler or CMake environment. CMake reported the FetchContent controls unused on first configure because no enabled dependency-fetch path was reached. No monitored network isolation is claimed.

Per-stage bounded diagnostics are retained in ignored `host-configure.log`, `host-build.log`, `android-configure.log` and `android-build.log`, referenced by the evidence manifests. Post-lock failure records include UTC/stage/source/options/tool context. A failed pre-tool/pre-lock invocation leaves previous timestamped evidence unchanged; always use the process exit result and evidence timestamp, not existence of a prior passed manifest.

POSIX failed subprocess groups receive SIGKILL; explicit descendant quiescence and Windows process-tree cleanup still need qualification before production-worker reuse. Windows/macOS native builds remain unverified. None of these tooling checks establish model/projector compatibility or a CT read.

### Next Integration Work

#### Vision Pairing Candidate Identified

Local model metadata was inspected from `Model/medgemma-1.5-4b-it-Q4_K_M.gguf` with upstream GGUFReader, using the existing model-factory Python environment (NumPy 1.26.4). It reports `general.architecture=gemma3`, Unsloth quantization, Google's MedGemma 1.5 base repository, 34 blocks and embedding length 2560. Metadata inspection does not establish tensor correctness or image-runtime compatibility.

The public [Unsloth artifact repository](https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF/tree/3855f948626b7ae42bccd082757f15078c53e758), queried using metadata API only, identifies revision `3855f948626b7ae42bccd082757f15078c53e758`. On 2026-10-12, both local `Model/` artifacts were rehashed and matched the published sizes and SHA-256 values below; filenames and sizes alone were not used as pairing evidence.

| Candidate | Published bytes | Published SHA-256 |
| --- | --- | --- |
| medgemma-1.5-4b-it-Q4_K_M.gguf | 2489894976 | b31becdf4f39561800505514cce67681604fe449d04dd35c8c92fd7848c6d7bd |
| mmproj-F16.gguf | 851252224 | f45f0f750587494e8f976d952cb396dfaa3158662120e2f9c224fc11f3882c83 |

This is a locally hash-verified candidate pair, not yet a qualified CT pipeline. The public API says the repository is ungated; that does not waive applicable model terms. The user supplied the projector locally after acknowledging applicable terms; no automated weight download was performed.

#### Local Synthetic Image Smoke

On 2026-10-12, the built Linux `llama-mtmd-cli` was run with explicit local model, projector and temporary generated-image paths, `--offline`, a 1,024-token context and an 8-token generation limit. The 32 x 32 RGB checker/grid PNG was generated for this test and removed afterward. Both GGUF artifacts were rehashed immediately before execution and matched the values above.

The runtime exited successfully, encoded the image (`mtmd batch encoding done in 54408 ms`) and generated: `The image displays a grid of colored squares`. This confirms a bounded synthetic image can pass through this local runtime/model/projector setup. It does not establish CT preprocessing parity, patient-image behavior, clinical accuracy, network isolation beyond the runtime's `--offline` setting, or Android execution. The runtime also identified the CLI as experimental.

Read-only GGUF metadata inspection explains the warning: architecture is `gemma3`; token 212 is `</s>` with token type 4 (`USER_DEFINED`); token 106 is `<end_of_turn>` with type 3 (`CONTROL`) and is `tokenizer.ggml.eos_token_id`; `tokenizer.ggml.eot_token_id` is absent, `tokenizer.ggml.add_eos_token=false`, and `tokenizer.ggml.add_bos_token=true`. In pinned llama.cpp `src/llama-vocab.cpp`, the EOG-token scan recognizes the literal `</s>`, warns if it is not already a control token, then promotes its attribute. This explains the warning mechanism; it does not prove whether the GGUF conversion's tokenizer semantics match the upstream tokenizer. Neither model metadata nor llama.cpp source was changed.

#### Synthetic CT Preprocessing Harness

Added a local harness at `tools/medgemma-bootstrap/ct_preprocessing.py` with pinned test dependencies in `ct-parity-requirements.txt`. Eight `unittest` cases passed using Python 3.12, NumPy 1.26.4, pydicom 3.0.2, Pillow 12.3.0, and JPEG codec 6.2. Tests validate this environment, generate temporary explicit-VR-little-endian DICOM files, and exercise integer `InstanceNumber` ordering, both slice-sampling branches (including the reference's `i=1..85` formula), pixel decode/rescale, golden RGB values, JPEG output, and invalid inputs. Command:

```bash
rtk proxy /home/amit/MedicalWork/tools/model-factory/totalseg-onnx/.venv/bin/python -m unittest discover -s tools/medgemma-bootstrap -p 'test_ct_preprocessing.py'
```

This is synthetic stage-level verification through JPEG encoding, not CT model-input parity. The separate processor-reference harness compares the pinned Transformers tokenizer/processor with the native GGUF path; it records exact token equality for one rendered prompt and a blocking fast-resizer tensor mismatch (see [Processor And Runtime Qualification](#processor-and-runtime-qualification-2026-10-12)). No patient images were used.

The separate `tools/medgemma-bootstrap/test_vision_pairing.py` integration suite verifies both local artifact sizes/hashes on one open descriptor, then runs the built `llama-mtmd-cli` against a temporary generated RGB grid with explicit model/projector/image paths, `--offline`, a 1,024-token context, and an 8-token limit. On 2026-10-12, the two-test suite passed five executions (about 71-72 seconds each); the runtime encoded the image and exited successfully. It does not rely on exact generated wording. Each temporary image is removed. Run it only when the existing local artifacts and host runtime are present:

```bash
rtk proxy /home/amit/MedicalWork/tools/model-factory/totalseg-onnx/.venv/bin/python -m unittest discover -s tools/medgemma-bootstrap -p 'test_vision_pairing.py'
```

This test verifies the synthetic local runtime path, not network isolation, processor/tokenizer parity, CT behavior, or clinical accuracy.

Continue through the [vision pairing spec](../../_bmad-output/implementation-artifacts/spec-medgemma-vision-pairing.md).

Runtime source, inspectable CT reference, pinned processor assets and dependencies are now engineer-delivered prerequisites rather than user-owned setup blockers. Still required: trusted model provenance and full vision-projector/tokenizer qualification; conversion/quantization qualification; resolving the fast-resizer mismatch and broader preprocessing/source-coverage parity; approved anatomical fixtures; independent/guided evaluation and reconciliation; measured platform budgets and on-device load/cancellation/privacy checks. Windows/macOS remain unverified. The `--offline` smoke setting does not replace monitored network isolation. No patient-image read, full CT model-input parity, clinical accuracy, or offline product integration is claimed.