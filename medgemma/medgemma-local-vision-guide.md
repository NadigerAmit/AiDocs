# MedGemma Local Vision: Newcomer Guide

**Canonical guide for:** the local MedGemma 1.5 4B GGUF plus its matching vision projector, the pinned llama.cpp multimodal runtime, the synthetic-image smoke test, and the remaining CT qualification work.

**Evidence date:** 2026-10-14
**Project status:** Research/evaluation only. Linux synthetic runtime checks and text-token parity passed; strict image-processor parity is blocked by a fast-resizer mismatch. Clinical performance and Android inference are not qualified.

This guide is the newcomer entry point and current operational summary. The [feasibility audit](medgemma-ct-feasibility.md) remains the detailed historical evidence log; the [integration design](../medgemma-integration-design.md) records the proposed RadAssist workflow and architecture. Where a statement is marked *planned* or *not qualified*, do not treat it as implemented or proven.

## 1. At A Glance

Two GGUF files are used together by a multimodal runtime:

| Artifact | Role | Verified size | SHA-256 |
| --- | --- | ---: | --- |
| `medgemma-1.5-4b-it-Q4_K_M.gguf` | Main MedGemma language/decoder weights in a quantized GGUF representation | 2,489,894,976 bytes | `b31becdf4f39561800505514cce67681604fe449d04dd35c8c92fd7848c6d7bd` |
| `mmproj-F16.gguf` | Matching vision sidecar used by llama.cpp's multimodal image path | 851,252,224 bytes | `f45f0f750587494e8f976d952cb396dfaa3158662120e2f9c224fc11f3882c83` |

They are separate, complementary files, not alternate versions of the same model. The pinned runtime loads the main model with `-m` and the sidecar with `--mmproj`; it uses the sidecar to turn image content into visual representations the language model can consume. The model file alone is not the tested image-capable pairing.

Both files correspond to the same pinned [Unsloth GGUF repository revision](https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF/tree/3855f948626b7ae42bccd082757f15078c53e758). The language model was already local; the user manually acquired the matching projector. Both files matched the published sizes and hashes on 2026-10-12. This verifies byte identity with those published values; it is not a signed-provenance check, proof of full tensor correctness, clinical validation, or proof that every runtime/version combination is compatible.

## 2. What The Names Mean

| Term | Meaning here |
| --- | --- |
| MedGemma | Google's medical-domain model collection. The official overview describes MedGemma 1.5 4B as multimodal and built on Gemma 3. This project uses a local GGUF conversion, not the original Transformers checkpoint format. |
| 4B | Approximate parameter-count class, not the file size. |
| GGUF | A model container format used by llama.cpp. A GGUF file may contain model weights and metadata; the format name alone does not identify its architecture, training, license, or fitness. |
| Q4_K_M | A llama.cpp quantization variant for the main weights. Quantization reduces storage and compute cost relative to higher-precision weights, with possible numerical/quality trade-offs. The suffix is not a clinical quality grade. |
| `mmproj` | The separate multimodal vision sidecar expected by this llama.cpp build. Its image/vision features are projected into a representation usable by the language model. It is not a standalone diagnostic model. |
| F16 | The sidecar's advertised half-precision floating-point representation. It is independent of the Q4 quantization of the main model. |
| llama.cpp | The C/C++ inference project used here to load GGUF models locally. Its `mtmd` components and `llama-mtmd-cli` provide multimodal inputs. |
| CT preprocessing | The steps that turn DICOM pixel data and metadata into the ordered, rescaled, windowed, encoded image inputs seen by the model. It is part of the model behavior, not incidental file conversion. |
| Tokenizer/chat template | Model-specific encoding of text and conversation structure. A compatible image projector does not by itself prove these are correct. |

## 3. Sources, Revision Pins, And Terms

### Model artifacts

The selected distribution is an Unsloth GGUF conversion of MedGemma 1.5 4B. Both selected files are pinned to repository revision `3855f948626b7ae42bccd082757f15078c53e758`:

- [Download the Q4_K_M language model](https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF/resolve/3855f948626b7ae42bccd082757f15078c53e758/medgemma-1.5-4b-it-Q4_K_M.gguf?download=true)
- [Download the F16 vision projector](https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF/resolve/3855f948626b7ae42bccd082757f15078c53e758/mmproj-F16.gguf?download=true)
- [Pinned repository file listing and model card](https://huggingface.co/unsloth/medgemma-1.5-4b-it-GGUF/tree/3855f948626b7ae42bccd082757f15078c53e758)

Desktop Settings provides these same pinned download links and a link to browse alternate GGUF files. Android Settings supports separate app-private import and replacement of the MedGemma model and projector GGUF files, with per-artifact presence status. Import does not check model/projector compatibility, enable inference, or qualify Android CT input; Android MedGemma inference remains unavailable and unqualified. Downloads remain user-initiated in a browser.

Android copies each selected file into app-private storage; the browser-downloaded original remains where the user saved it. The import therefore requires additional free space for a second copy (up to 8 GiB for the model and 4 GiB for the projector).

Do not use `main` or an unpinned “latest” file for a reproducible pairing. Do not substitute `mmproj-BF16.gguf`, `mmproj-F32.gguf`, another quantization, or another repository revision without a separate compatibility check. A matching filename is not enough.

Before obtaining or using weights, each user must read and follow the current [Health AI Developer Foundations terms](https://developers.google.com/health-ai-developer-foundations/terms) and [prohibited use policy](https://developers.google.com/health-ai-developer-foundations/prohibited-use-policy). See Google's [MedGemma overview](https://developers.google.com/health-ai-developer-foundations/medgemma) and [model card](https://developers.google.com/health-ai-developer-foundations/medgemma/model-card). This guide does not accept terms on anyone's behalf or provide legal advice. The current work records that the project user acknowledged terms for local evaluation.

### Runtime and CT reference

The developer bootstrap pins these independent source repositories:

| Purpose | Repository and pinned revision |
| --- | --- |
| Local GGUF inference and `mtmd` multimodal runtime | [llama.cpp](https://github.com/ggml-org/llama.cpp/tree/5c8854fd88319125125dc4b836c98b1b215b0f23), commit `5c8854fd88319125125dc4b836c98b1b215b0f23` |
| CT example/reference behavior | [Google Health MedGemma notebook](https://github.com/google-health/medgemma/blob/a60a66024f6153496dfa9490dba12d4f1fbd092e/notebooks/high_dimensional_ct_hugging_face.ipynb), commit `a60a66024f6153496dfa9490dba12d4f1fbd092e` |

The CT notebook is a reference example, not a production processor or a whole-study reporting specification. Its cells were extracted for inspection and **were not executed**. The notebook includes authenticated network access, downloads, package setup, visualization, and inference; none of those notebook operations is part of the local smoke test.

## 4. Local Files And Integrity Checks

The verified project paths are:

```text
Model/medgemma-1.5-4b-it-Q4_K_M.gguf
Model/mmproj-F16.gguf
```

`Model/` and `target/` are ignored local-artifact locations in this repository. Do not add model weights or patient images to Git, issue attachments, or source-control history. Store and access them according to the applicable terms and the organization's data-handling rules.

From the repository root on Linux, check the files after every transfer or copy:

```bash
rtk proxy stat -c '%s %n' Model/medgemma-1.5-4b-it-Q4_K_M.gguf Model/mmproj-F16.gguf
rtk proxy sha256sum Model/medgemma-1.5-4b-it-Q4_K_M.gguf Model/mmproj-F16.gguf
```

Expected byte counts and SHA-256 values are in the table in Section 1. For reproducibility checks of this pinned reference pair, stop if either differs. Do not overwrite a mismatching artifact automatically. Hash agreement establishes equality to the pinned published bytes; it does not establish a cryptographic publisher signature, clinical suitability, or runtime correctness.

These byte counts and hashes describe the reference pair used for reproducible developer qualification; they are not an application gate for user-selected files. Desktop Settings accepts any readable, non-empty local model and projector files. Choose a pair intended to work together; the runtime can still reject an incompatible or invalid pair when analysis starts. A successful load does not establish processor parity or clinical suitability. The application does not download files automatically; use its Hugging Face links in a browser, then select the files locally.

## 5. Build The Pinned Local Runtime

### Prerequisites

For a host build, install/provide Node.js 20+, Git, CMake, Ninja, and a working local C/C++ compiler toolchain. The recorded Linux build used Node 20.20.2, Git 2.43.0, CMake 3.28.3, Ninja 1.11.1, and GNU C/C++ 14.3.0. Tool versions and host compiler affect reproducibility.

The build is CPU-only by configuration. No CUDA, Metal, Vulkan, BLAS, Accelerate, OpenMP, server, app, tests, or examples are built. Although `GGML_NATIVE=OFF`, the recorded Linux binary still uses the upstream CPU backend's SSE4.2/F16C/FMA/BMI2/AVX/AVX2 variants; this is not a promise of compatibility with arbitrary older x86 CPUs.

### Build steps

Run from the repository root. The bootstrap defaults to no source fetch and builds from already-present, exact, clean pinned checkouts:

```bash
rtk proxy node tools/medgemma-bootstrap/bootstrap.mjs --target host
```

On a fresh workspace, explicitly allow acquisition of the two pinned **source** repositories and then build:

```bash
rtk proxy node tools/medgemma-bootstrap/bootstrap.mjs --fetch --target host
```

`--fetch` fetches only the pinned llama.cpp and Google Health MedGemma source revisions. It does not download either model weight, install packages, run the notebook, or process medical images. The build and extracted source/evidence are under ignored `target/medgemma-bootstrap/`. Existing wrong, dirty, or unexpected checkouts are refused rather than reset or cleaned.

The expected Linux host executable is:

```text
target/medgemma-bootstrap/build-host/bin/llama-mtmd-cli
```

RadAssist Desktop also packages a CPU Rust worker that statically links `llama-cpp-2` 0.1.159 with its experimental `mtmd` API. Build it from the repository root with `cargo build -p radassist-medgemma-worker`; the desktop app selects it by default only on tested Linux x86-64 and retains the pinned CLI as the fallback on other hosts. The crate's bundled llama.cpp revision has not been proven identical to the CLI pin below, and a successful worker smoke does not establish HF-fast processor parity. A synthetic worker inference check is available after building the worker:

```bash
rtk proxy target/medgemma-bootstrap/processor-venv/bin/python tools/medgemma-bootstrap/test_vision_pairing.py VisionPairingTest.test_embedded_rust_worker_runs_pinned_pair_offline
```

### Desktop Rust Worker: MTMD, Inputs, And The CLI

The two desktop runtimes use the same local language-model/projector pair, but they are different programs and do not share a proven llama.cpp revision:

| | Rust MTMD worker | `llama-mtmd-cli` fallback |
| --- | --- | --- |
| How it is built | `radassist-medgemma-worker` links `llama-cpp-2` 0.1.159 and llama.cpp/MTMD into a Rust executable. | Separate C++ executable built from the pinned llama.cpp commit `5c8854fd88319125125dc4b836c98b1b215b0f23`. |
| How Kotlin calls it | Starts the worker as a child process with `--offline`, `--model`, `--projector`, `--image`, `--prompt`, and `--max-tokens`. Rust APIs call llama.cpp directly inside that process. | Starts the CLI as a child process with `-m`, `--mmproj`, `--image`, `-p`, `-n`, `-c`, and `--offline`. The CLI parses those flags itself. |
| Process boundary | Still out-of-process from Kotlin for cancellation and failure isolation; it does not run as an in-process Kotlin library or as a server. | Also out-of-process. |
| Engine identity | Crate 0.1.159's bundled llama.cpp revision has not been proven to match the pinned CLI source. The MTMD Rust API is experimental. | The source revision is pinned and identified. |

The Rust dependency enables MTMD at compile time; without this feature the crate does not expose its `mtmd` module or build the MTMD native target:

```toml
llama-cpp-2 = { version = "=0.1.159", default-features = false, features = ["mtmd"] }
```

This is the dependency declaration in the [worker manifest](../../radassist-multiplatform/native/radassist-medgemma-worker/Cargo.toml#L9). In `llama-cpp-2` 0.1.159, `mtmd` forwards to `llama-cpp-sys-2/mtmd`, compile-time-gates the Rust `mtmd` module, enables the CMake `LLAMA_BUILD_MTMD` target, and links the native MTMD library. Disabling default features keeps this worker's selected feature set explicit. The resulting Rust worker is statically linked, but Kotlin still launches it as a process.

#### HF Reference Contract Versus The Current Worker

The pinned HF config describes a **model/reference capability**, not the current desktop worker's limits. Here, “128K” means 131,072 context positions (`max_position_embeddings`). The pinned config and processor inputs record 256 visual positions per image and an 896 x 896 processor target. The requested HF-side generation expectation is up to 8,192 output tokens; that limit is not set by the pinned `generation_config.json` in this repository, so treat it as a caller/evaluation budget, not a verified runtime setting.

| Stage | HF reference expectation | Current RadAssist Rust worker |
| --- | --- | --- |
| Text | Text/question is rendered with the HF Gemma 3 chat template and tokenized by the HF tokenizer. | Kotlin passes a fixed research-only instruction plus the displayed frame's WW/WL. Rust wraps it in a user message and applies the chat template stored in the GGUF. |
| Image | HF `Gemma3ImageProcessorFast` converts RGB, resizes to 896 x 896, rescales by `1/255`, normalizes with channel mean/std `0.5`, and represents each image with 256 visual positions. | Kotlin supplies an explicitly selected contiguous range of 1-8 axial CT frames as locally rendered PNGs at one WW/WL. The worker checks each image against 32 MiB/16 million pixels and the batch against 64 MiB/64 million pixels. MTMD performs its own image processing; this worker does not explicitly resize PNGs to 896 x 896 or assert that each encoded image contributes exactly 256 positions. |
| Input budget | The pinned model config allows up to 131,072 positions across input context. | The worker processes at most two ordered images per batch, using a fresh 4,096-position context for each. It checks `image/text positions + requested output tokens <= 4,096`; oversized batches fail explicitly. |
| Generated response | The expected HF-side output budget supplied for this comparison is up to 8,192 generated tokens. | Each batch defaults to 128 tokens and rejects a request over 128. Rust emits one flushed JSON result event per batch; runtime errors go to stderr. |

Therefore, the app currently implements only a bounded 1-8-frame slice of the HF contract, not the full 128K-input/8192-output profile. More importantly, matching the HF shape/token metadata does not establish pixel parity: the existing strict comparison found that HF-fast normalized pixels differ from the native GGUF/MTMD path. Do not infer equivalent model input, answer quality, or clinical behavior from a successful Rust smoke.

#### Where The Current Call Happens

1. The desktop viewer lets the user choose an inclusive first/last frame range in the current indexed axial CT series. It shows the exact range and count; invalid, cross-series, or over-limit selections cannot start. The callback selects the Rust worker or CLI with `DesktopMedGemmaClient.selectRuntime` ([`Main.kt`](../../radassist-multiplatform/desktopApp/src/main/kotlin/com/nadigerventures/radassist/desktop/Main.kt#L733)).
2. `DesktopDicomImageSource.withManagedResearchFrameImages` locally re-decodes each selected frame at the same WW/WL, checks its study/series/SOP identity and aggregate size, then removes every managed PNG after success, failure, or cancellation ([`DesktopDicomImageSource.kt`](../../radassist-multiplatform/shared/src/jvmMain/kotlin/com/nadigerventures/radassist/platform/jvm/viewer/DesktopDicomImageSource.kt)).
3. `DesktopMedGemmaClient` passes the ordered image paths and visible selection scope to either runtime, with the same output/context bounds and timeout ([`DesktopMedGemmaClient.kt`](../../radassist-multiplatform/shared/src/jvmMain/kotlin/com/nadigerventures/radassist/platform/jvm/ai/DesktopMedGemmaClient.kt)). The pinned CLI inserts one matching default marker per repeated `--image`; the Rust worker constructs one marker alongside each ordered bitmap. Analysis is default-off and the Rust option is selectable only on the tested Linux x86-64 host.
4. In Rust, `analyze` loads the GGUF and projector, reads each bounded image buffer in argument order, creates one MTMD bitmap and one matching `mtmd_default_marker()` per frame, applies the GGUF chat template, and tokenizes the full list. `chunks.total_positions()` feeds the 2,048-position budget check; `eval_chunks` encodes the combined image/prompt before the greedy text-generation loop ([`main.rs`](../../radassist-multiplatform/native/radassist-medgemma-worker/src/main.rs)).
5. The separate HF parity harness uses `processor.apply_chat_template` and reads HF `pixel_values` for direct comparison ([`processor_reference_parity.py`](../../tools/medgemma-bootstrap/processor_reference_parity.py#L419)). That reference route is not called by the desktop worker, and parity remains blocked.

The bootstrap records source commits, options, tools, logs and artifact hashes in `target/medgemma-bootstrap/manifest-host.json`. A successful bootstrap's own smoke checks `--version` and `--help`; it does **not** load the MedGemma weights or prove image inference.

Optional Android arm64 cross-build, with an already installed complete local NDK:

```bash
rtk proxy node tools/medgemma-bootstrap/bootstrap.mjs --target android --ndk "$HOME/Android/Sdk/ndk/27.0.12077973"
```

This produces Android libraries; it does not install or run inference on a device. To package the app, bootstrap the pinned source/runtime outputs first, then build the APK:

```bash
rtk proxy node tools/medgemma-bootstrap/bootstrap.mjs --fetch --target android --ndk "$HOME/Android/Sdk/ndk/27.0.12077973"
rtk proxy ./gradlew :androidApp:assembleDebug
```

Gradle stages existing `target/medgemma-bootstrap/build-android/bin` libraries and builds the JNI bridge. It does not fetch or modify llama.cpp sources; the explicit bootstrap command fetches only pinned source repositories, never model weights. Preserve a dirty generated checkout and inspect it rather than cleaning it to make bootstrap pass. The resulting APK packages the arm64 CPU runtime and isolated `:medgemma` service, but has not been installed or exercised on an Android device.

Android admission rejects non-arm64/API-26 devices, low-RAM devices, missing runtime libraries, missing artifacts, and devices whose currently available memory is below twice the imported model-plus-projector file size. That multiplier is a conservative engineering heuristic, not a measured Android budget or qualification threshold; it can reject capable devices and does not guarantee that an admitted run will complete. Resource exhaustion is contained by the separate service process. Windows and macOS native builds have not been completed in this qualification record.

## 6. Reproduce The Synthetic-Image Smoke

This is a bounded **runtime plumbing** check, not a CT test. It uses a generated 32 x 32 RGB grid image, explicit local paths, an 8-token output cap, a 1,024-token context, and llama.cpp's `--offline` option. It does not use patient data. Run only after reading the applicable terms and verifying both hashes.

The following Bash block uses Node.js built-ins only to create a temporary PNG. It removes the generated image on exit:

```bash
set -euo pipefail
image="target/medgemma-bootstrap/synthetic-smoke-$$.png"
trap 'rtk proxy rm -f -- "$image"' EXIT

rtk proxy node - "$image" <<'NODE'
const fs = require('node:fs');
const zlib = require('node:zlib');
const output = process.argv[2];
function crc32(bytes) {
  let crc = 0xffffffff;
  for (const byte of bytes) {
    crc ^= byte;
    for (let bit = 0; bit < 8; bit++) crc = (crc >>> 1) ^ ((crc & 1) ? 0xedb88320 : 0);
  }
  return (crc ^ 0xffffffff) >>> 0;
}
function chunk(type, data) {
  const typeBytes = Buffer.from(type);
  const body = Buffer.concat([typeBytes, data]);
  const length = Buffer.alloc(4);
  const checksum = Buffer.alloc(4);
  length.writeUInt32BE(data.length);
  checksum.writeUInt32BE(crc32(body));
  return Buffer.concat([length, body, checksum]);
}
const width = 32;
const height = 32;
const header = Buffer.alloc(13);
header.writeUInt32BE(width, 0);
header.writeUInt32BE(height, 4);
header[8] = 8;
header[9] = 2;
const rows = [];
for (let y = 0; y < height; y++) {
  const row = Buffer.alloc(1 + width * 3);
  for (let x = 0; x < width; x++) {
    const offset = 1 + x * 3;
    row[offset] = x < 16 ? 220 : 30;
    row[offset + 1] = y < 16 ? 40 : 190;
    row[offset + 2] = ((x >> 2) ^ (y >> 2)) & 1 ? 220 : 60;
  }
  rows.push(row);
}
const png = Buffer.concat([
  Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]),
  chunk('IHDR', header),
  chunk('IDAT', zlib.deflateSync(Buffer.concat(rows))),
  chunk('IEND', Buffer.alloc(0)),
]);
fs.writeFileSync(output, png, { flag: 'wx' });
NODE

rtk proxy target/medgemma-bootstrap/build-host/bin/llama-mtmd-cli \
  -m Model/medgemma-1.5-4b-it-Q4_K_M.gguf \
  --mmproj Model/mmproj-F16.gguf \
  --image "$image" \
  -p "Describe the visible synthetic image in one short sentence." \
  -n 8 -c 1024 --offline
```

### What A Pass Means

Two runs on 2026-10-12 exited with status 0 and encoded the generated image in 54.408 and 54.573 seconds. They generated `The image displays a grid of colored squares` and `The image displays a four-color grid`, respectively. Generated text can vary even for the same synthetic input; check successful local model/projector loading, image encoding, bounded completion, and exit status rather than requiring an exact sentence.

The run emitted a warning for token 212 (`</s>`). Local GGUF metadata marks it `USER_DEFINED` (type 4), while token 106 `<end_of_turn>` is `CONTROL` (type 3) and is the declared EOS; `add_eos=false`, `add_bos=true`, and no EOT id is declared. The pinned llama.cpp source recognizes literal `</s>` as an EOG spelling and promotes its token type, which explains this warning. Whether that conversion matches the upstream tokenizer's intended semantics remains unverified. Do not edit the GGUF or suppress the warning. The CLI itself calls this an experimental multimodal test tool.

The `--offline` flag requests local-only operation from this executable; network traffic was not independently monitored or denied during the recorded run. Therefore this is not a formal network-isolation test.

### Synthetic CT Preprocessing Tests

The local harness tests the pinned notebook's deterministic preparation stages through JPEG encoding. It uses generated temporary DICOM files only; it neither invokes the notebook nor loads MedGemma. Activate a Python 3.12 environment with the versions in `tools/medgemma-bootstrap/ct-parity-requirements.txt` already installed, then run from the repository root. This command does not install packages:

```bash
rtk proxy python -m unittest discover -s tools/medgemma-bootstrap -p 'test_ct_preprocessing.py'
```

The 2026-10-12 run passed 8 tests with Python 3.12, NumPy 1.26.4, pydicom 3.0.2, Pillow 12.3.0 and JPEG codec 6.2. It covers environment pins, integer slice ordering, the `MAX_SLICE=85` sampling rule, signed-pixel decode/rescale, windowed RGB goldens, JPEG output and invalid inputs. The JPEG digest is qualified only for codec 6.2; another codec version is an unverified environment, not a reason to silently update the golden. The suite does not validate processor resize/tokenization, clinical inputs, or model accuracy.

The separate integration suite requires the two verified local model artifacts and the Linux runtime built in Section 5. It rechecks both artifact hashes before loading them, then runs only a generated image with `--offline`:

```bash
rtk proxy python -m unittest discover -s tools/medgemma-bootstrap -p 'test_vision_pairing.py'
```

The two-test integration suite passed five executions on 2026-10-12, each in about 71-72 seconds. It checks successful vision encoding and exit status, not exact generated wording. It uses no patient input or download. This still does not test formal network isolation or CT/processor parity.

### Reference Processor And DICOM-Derived Runtime Qualification

The pinned reference is the Hugging Face repository [google/medgemma-1.5-4b-it](https://huggingface.co/google/medgemma-1.5-4b-it), fixed at revision `91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b`. The checked-in [allowlist and SHA-256 manifest](../../tools/medgemma-bootstrap/processor-reference-manifest.json) identify the exact ten files acquired into the ignored `target/medgemma-bootstrap/processor-reference/` directory. The manifest is checked into `tools/`; only downloaded files and generated results are under ignored `target/`. No Transformers model weights were requested or fetched.

The ten files are deliberately limited to processor, tokenizer, and model configuration inputs:

| Files | Why they are needed |
| --- | --- |
| `preprocessor_config.json`, `processor_config.json`, `config.json` | Select the image processor and define model/image input configuration, including resize and normalization settings. |
| `tokenizer.json`, `tokenizer.model` | Provide the fast-tokenizer data and SentencePiece model used by the pinned processor/tokenizer comparison. |
| `tokenizer_config.json`, `chat_template.jinja`, `added_tokens.json`, `special_tokens_map.json` | Define tokenizer behavior, chat rendering, added tokens, and special-token identities so the text-side comparison uses the model's own conventions. |
| `generation_config.json` | Preserve the pinned repository's generation metadata; it is not used to claim image parity. |

The download does **not** include `model.safetensors`, `pytorch_model.bin`, or other Transformers checkpoint weights. The local `Model/*.gguf` files are a separate input: the Unsloth main GGUF and matching projector are listed in Sections 1 and 3. They are required by this harness to verify the local pair, compare token IDs with llama.cpp, and run the optional native runtime cases; they are not downloaded from the Google processor repository. They were already present in this workspace and were not fetched during this qualification. On a clean machine, obtaining either GGUF is a separate action governed by the source terms and approval; the current processor-reference scope does not authorize a weight download.

#### Reproduce The Qualification

Run these steps from the repository root on Linux. Prerequisites are Python 3.12.3, `uv` (0.12.9 was used for this run), Git/CMake/Ninja and a C++ compiler for the pinned llama.cpp build and preprocessing probe. The uv executable itself is not version-locked by the Python requirements file; install it using the [official uv instructions](https://docs.astral.sh/uv/getting-started/installation/) and check with `rtk proxy uv --version`. Read and follow the repository/model access terms first. If Hugging Face requires gated access, request it through the model page using your own account. The login command prompts for a token; do not put a token in a command, document, or source file.

1. Build the pinned local runtime as described in [Section 5](#5-build-the-pinned-local-runtime). Its source revision is `5c8854fd88319125125dc4b836c98b1b215b0f23`; that executable performs the native image path and supplies the native tokenizer. The parity runner also compiles `native_image_preprocess_probe.cpp` against this local build when its probe executable is absent or stale.
2. Put the hash-verified local main GGUF and matching projector at the paths in Section 4. These are required inputs for the existing harness; do not download or substitute a different pair during this qualification.
3. Create the isolated Python environment and synchronize the fully pinned, hash-checked Linux dependency set. The `.in` file documents top-level requirements; the `.txt` lock records the resolved dependency closure and hashes.

```bash
rtk proxy uv venv --python 3.12.3 target/medgemma-bootstrap/processor-venv
rtk proxy uv pip sync \
  --python target/medgemma-bootstrap/processor-venv/bin/python \
  --require-hashes --torch-backend cpu \
  tools/medgemma-bootstrap/processor-reference-requirements.txt
```

4. Authenticate interactively if required, then download only the ten manifest-listed files at the immutable commit. Listing files explicitly is intentional: it excludes the large Transformers model weights.

```bash
rtk proxy target/medgemma-bootstrap/processor-venv/bin/hf auth login
rtk proxy target/medgemma-bootstrap/processor-venv/bin/hf download \
  --revision 91850547d9f0b2fdd21aa7c5f4f3d1a8a52c243b \
  --local-dir target/medgemma-bootstrap/processor-reference \
  google/medgemma-1.5-4b-it \
  added_tokens.json chat_template.jinja config.json generation_config.json \
  preprocessor_config.json processor_config.json special_tokens_map.json \
  tokenizer.json tokenizer.model tokenizer_config.json
```

5. Run the complete comparison. Before comparing, the runner verifies every downloaded file against the checked-in manifest, verifies dependency versions, and verifies the local GGUF pair's byte counts and SHA-256 values. It compares one deterministic rendered text template, the selected HF fast image processor, the HF slow path as diagnostic evidence only, and the local native preprocessor. It then exercises one- and two-slice synthetic runtime cases using explicit local files and `--offline`.

```bash
rtk proxy target/medgemma-bootstrap/processor-venv/bin/python \
  tools/medgemma-bootstrap/processor_reference_parity.py
```

The full result is written to ignored `target/medgemma-bootstrap/processor-reference-result.json`. The overall gate requires exact text-token equality, exact HF-fast/native tensor equality for the DICOM-derived PNG and JPEG plus both representative synthetic dimensions, exact pre-normalization resized-RGB equality at both sizes, and successful non-empty runtime cases. A strict mismatch is an expected qualification outcome, not a runner crash: the current script exits with status 2 after writing JSON when the gate is blocked. Missing pinned assets or setup errors produce a blocked JSON result with a `failure.stage` and exact error; runtime launch failures and timeouts are recorded against the named runtime case and also block the result. To rerun only the processor/tokenizer/resize comparisons without model generation, append `--skip-runtime`; that writes to `processor-reference-resize-result.json` so it does not overwrite the last full runtime result. The native probe and tokenizer comparisons still require the local artifacts described above. This focused command also exits 2 while strict parity is blocked:

```bash
rtk proxy target/medgemma-bootstrap/processor-venv/bin/python \
  tools/medgemma-bootstrap/processor_reference_parity.py --skip-runtime
```

The checks answer separate questions. Text-token equality tests the rendered template/tokenizer path. Image parity here means equality of the normalized RGB image tensor immediately before the vision encoder/projector, not equality of generated answers. The native path currently receives image files and performs its own resize; therefore differing tensors mean the local GGUF route is not yet demonstrated to consume the same image input as the notebook's selected HF fast processor. Such an input difference can change vision features and downstream outputs, but this experiment does not measure response changes or clinical impact. It is not evidence that the model gives a particular wrong answer. Keep RadAssist integration blocked on a claim of HF processor equivalence until the native values match or an explicitly approved processor/tolerance qualification includes suitable behavioral validation. Do not silently substitute the slow processor or introduce a tolerance after seeing these results.

Reproduce on the same Linux host after setup, using the existing local model/projector and generated fixtures:

```bash
rtk proxy target/medgemma-bootstrap/processor-venv/bin/python tools/medgemma-bootstrap/processor_reference_parity.py
```

Observed results on 2026-10-12:

- HF/native token IDs exactly matched for the same rendered chat-template prompt (15/15 tokens).
- Against the notebook's selected HF fast processor, normalized tensor shapes matched at 896 x 896 x 3 but values did not. Synthetic PNG: max absolute difference `0.0078431368`, mean `0.0000024076`, 869,467 of 2,408,448 values differed. Synthetic JPEG: max `0.0156863332`, mean `0.0005316096`, 968,785 values differed.
- Deterministic resize characterization now tests synthetic RGB PNGs at both 512 x 512 (upsampled to 896 x 896) and 1024 x 1024 (downsampled to 896 x 896). HF fast and native output shapes match in both cases, but values remain different:

| Synthetic input | Max absolute difference | Mean absolute difference | Different tensor values |
| --- | ---: | ---: | ---: |
| 512 x 512 | `0.0078431964` | `0.0000886595` | 892,860 / 2,408,448 |
| 1024 x 1024 | `0.0078431955` | `0.0000057424` | 881,260 / 2,408,448 |

  The exact RGB input hashes and tensor hashes are recorded in `target/medgemma-bootstrap/processor-reference-resize-result.json`. The mismatch at both scale directions supports investigating resize sampling/rounding as the cause, but does not prove which implementation detail is responsible.
- HF slow processor vs native was exactly equal on the synthetic PNG. This is diagnostic only; do not treat it as acceptance of fast-path parity or switch the reference silently.
- With the verified local GGUF pair and `--offline`, one generated DICOM-derived slice decoded as one visual-token group and completed one-token generation with exit 0 in 68.12 s. The two-slice case decoded two groups and completed with exit 0 in 132.34 s.
- `RUSAGE_CHILDREN` peak-RSS high-water observations were 5,201,672 KiB and 5,224,416 KiB, respectively. They are exploratory, not per-case memory guarantees or an agreed budget.

The runner records a blocked overall gate because fast image parity fails. The successful runtime cases establish only synthetic Linux inference plumbing; they do not establish conversion/quantization equivalence, network isolation, clinical behavior, Android execution, or acceptable performance. Existing focused baselines also passed on this host: 8/8 synthetic preprocessing tests and 2/2 local vision-pairing tests.

The resize-only stage capture also compares the raw pre-rescale/pre-normalization `uint8` RGB images. The probe reads the actual normalization arrays from the loaded GGUF projector; both native mean and std are `[0.5, 0.5, 0.5]`, matching HF. The native normalized tensor is inverted only after all 256 uint8 channel values pass an exact round-trip check, and each recovered tensor re-normalizes exactly to the native output. Both sides are 896 x 896 x 3; byte equality still fails, locating the tested differences before rescale/normalize:

| Input | Input RGB SHA-256 | HF fast resized SHA-256 | Native resized SHA-256 | Different bytes | Max / mean absolute difference |
| --- | --- | --- | --- | ---: | ---: |
| 512 x 512 | `02949aa03e8f7d8bd634cbd951a69f7749937ada1c4231dd6f35edeee169a0a9` | `2daa0fec10f41320b6779306ebf33834c52d0ecd7624a7fb5fe786888b611784` | `5d90db4d31f137e9a8e0ad7f8ab135e91dc9b7b42dd0ed725361a9b7b6697477` | 27,222 / 2,408,448 | 1 / 0.0113027142 |
| 1024 x 1024 | `9d0bc5408e62e7a8f113c024c1d03fc0e81692ec5d13af342e5f6e9256b59052` | `084d1cfc9ba4ff3257e32871ece02cf2659b537113a1648318f20f3d9a675e80` | `3c111dad9f88b6d631f9fe0cd5252aa05f9350216933ac83b3af3b9b48f318fc` | 1,760 / 2,408,448 | 1 / 0.0007307610 |

#### Fast/Slow Candidate Matrix (2026-10-13)

The expanded comparison used the same generated RGB pixels for HF fast, HF slow, and native. Both HF processors had rescaling/normalization disabled for the raw RGB comparisons; native RGB was recovered only after the exact projector-mean/std round-trip check above. The candidate slow path was also compared on the existing synthetic DICOM-derived PNG and JPEG.

| Input | HF fast vs native | HF slow vs native |
| --- | --- | --- |
| DICOM-derived PNG | Existing comparison differs | Exact |
| DICOM-derived JPEG | Existing comparison differs | 137,502 / 2,408,448 normalized values differ; max `0.0156863`, mean `0.0004959` |
| 512 x 512 RGB PNG | 27,222 / 2,408,448 bytes differ; max 1 | Exact |
| 896 x 896 RGB PNG | Exact | Exact |
| 1024 x 1024 RGB PNG | 1,760 / 2,408,448 bytes differ; max 1 | Exact |
| 512 x 768 RGB PNG | 2,088,101 / 2,408,448 bytes differ; max 255 | 2,088,101 / 2,408,448 bytes differ; max 255 |
| 1024 x 512 RGB PNG | 2,137,924 / 2,408,448 bytes differ; max 255 | 2,138,100 / 2,408,448 bytes differ; max 255 |

All image tensors have the expected 896 x 896 x 3 shape, but HF slow fails exact parity for the DICOM-derived JPEG and both non-square cases. **Conclusion: HF slow is not a general substitute for the native preprocessing path.** Its exact match is limited to the tested square PNG inputs. These results still do not identify the exact aspect-ratio or JPEG decoding sub-operation; keep `use_fast=True` as the reference, and do not change the gate or integrate on HF-slow equivalence.

The exact stage records are in `target/medgemma-bootstrap/processor-reference-result.json` for a full run and `processor-reference-resize-result.json` for `--skip-runtime`. The latter contains no runtime evidence.

The latest full harness rerun at 2026-10-13T01:19:30Z passed the one-slice and two-slice offline cases in 68.12 s and 132.34 s. Peak child-RSS high-water readings were 5,201,672 KiB and 5,224,416 KiB; they are exploratory, not approved budgets.

Run the focused resize/gate regression suite in the locked environment (11 tests):

```bash
rtk proxy target/medgemma-bootstrap/processor-venv/bin/python -m unittest discover -s tools/medgemma-bootstrap -p 'test_processor_reference_parity.py'
```

## 7. What The CT Reference Does And Does Not Specify

The pinned notebook was inspected at the pinned revision. The following are observations, not approved RadAssist presets or clinical recommendations:

1. Its example sorts instances by integer `InstanceNumber` and defines `MAX_SLICE=85`.
2. If a series has more than 85 images, its code selects indices `int(round(i / 85 * (N - 1)))` for `i=1..85`. Preserve that exact formula when measuring reference parity; it is not an assumed endpoint-inclusive resampling scheme.
3. It decodes DICOM `pixel_array` and applies `pydicom.pixels.apply_rescale` to produce Hounsfield-unit values.
4. It creates three RGB channels using clip ranges `(-1024, 1024)`, `(-135, 215)`, and `(0, 80)`, which the notebook labels wide, mediastinum/chest, and brain windows. It maps each channel to 0..255, rounds, and casts to `uint8`. These are notebook labels, not approved all-CT presets.
5. It JPEG-encodes images with PIL defaults, labels images `SLICE 1..N`, and sends them in a liver-focused teaching prompt.
6. The Transformers example uses `AutoProcessor` and a particular model-loading/generation path. The pinned processor configuration and dependencies have now been compared with the local GGUF path: the tested rendered text template matches exactly, while image tensors from the selected fast processor differ. See Section 6 for the measured results.

The pinned processor config selects `Gemma3ImageProcessorFast`, 896 x 896 bilinear resize, RGB conversion, rescaling by `1/255`, normalization with mean/std `[0.5,0.5,0.5]`, and image sequence length 256. The executed comparison found that the native GGUF image path does not produce identical tensors to this fast processor; results and exact differences are in Section 6 above. A diagnostic HF slow processor matched the native tensor exactly, but it is not the notebook's selected path and does not close parity.

The notebook's example is not a whole-study report workflow. Its windows are not validated for all CT anatomies. The extracted cells do not define physical-geometry sorting or resampling. Screenshots and visual similarity alone are insufficient.

## 8. RadAssist Architecture: Planned Versus Built

### Desktop Research Output Flow (Updated 2026-10-16)

1. Enable MedGemma research in Settings and select/validate the local runtime, pinned model, and matching projector. The embedded worker remains CPU-only; analysis remains explicit, offline, and default-off.
2. Display an axial CT series at the desired window, choose an inclusive frame range of 1-8 consecutive frames, and queue **MedGemma Research**. The Viewer shows each job's progress and explicit Cancel action. Frames are rendered at the submitted WW/WL and analyzed in ordered pairs; completed findings appear before later pairs finish. The selectable output appears below Plane/Fit with a research warning and concise findings; source/runtime details are expandable.
3. Navigate or change the displayed frame while work continues in the app-level serial queue. Report review shows the same job progress/cancel action and partial or completed output separately from the clinical report. Navigation does not cancel jobs; use the job's explicit Cancel action. Research never enters raw/formatted clinical fields or radiologist report history.
4. **Copy Research** copies the warning, completion/partial status, timestamp, study/series identity, selected range and every submitted frame identity, displayed window, model/projector/runtime context, and batch findings. Clipboard failure is reported separately from clinical-report copying. A clipboard copy leaves the application boundary and must be handled according to local data policies.
5. **Save to Research History** explicitly saves one record per job to `~/.radassist/medgemma-research.json`. Saving a partial job and later saving its completion updates the same record. No autosave occurs, and no source images are persisted in this repository. Saved text and source context can contain sensitive information; the store is local JSON, not encrypted storage.
6. Open **History > MedGemma Research** for saved text, copy, or deletion. **Radiologist Reports** remains separate. Saved entries survive restart and are retained until explicit deletion or Reset App Data; they are never restored as the current displayed image's result. Read/save/delete errors remain visible, and corrupt history is not silently overwritten.
7. Submitted jobs retain their study/frame/window/runtime snapshot when the viewer or settings change. Reset explicitly cancels active/queued jobs and clears saved research; it reports failure if deletion fails and does not delete source DICOM or model files. Job state is in memory; temporary rendered PNGs are local files leased until processing/cancellation cleanup. Jobs are not restored after app restart. Normal completion/cancellation cleans temporary inputs; abrupt process termination is not proof that cleanup ran.

The concise fixed prompt requests brief findings with visible evidence and image references, plus one limitation sentence. If a batch reaches the 128-token cap before the model ends naturally, its event is marked truncated and the overall job is labeled partial, not complete. The selected-frame count is 1-8, batches contain two images, and each has a 4,096-position context. Images remain bounded to 32 MiB/16 million pixels each and 64 MiB/64 million pixels total. Rust loads the model/projector once per job, resets inference context between serial batches, and emits ordered events; the CLI fallback reloads model weights for each batch. The app limits active jobs to four and retains at most twenty job records in memory. Neither runtime silently drops selected frames.

Each batch prompt requests only concise **Findings** and **Limitations**, with uncertainty and no boilerplate interpretation section. The prompt forbids invented history, measurements, contrast status or other clinical facts, definitive diagnosis, treatment advice, and claims beyond the supplied images. These are instructions, not an output-safety guarantee. The 128-token output cap can truncate; it is not evidence of accuracy or processor/clinical qualification. The six-minute client timeout and explicit cancellation remain enforced. Any future budget increase requires coordinated Kotlin/Rust changes and evaluation; do not raise a limit as a substitute for evidence. The 8-token/1,024-context CLI example in Section 6 remains a deliberately smaller historical plumbing check.

### Performance And Capacity Consultation (2026-10-16)

**Problem statement:** Reduce offline MedGemma image-analysis latency while keeping DICOM viewing and dictation responsive. Extend explicitly selected CT samples from eight to approximately fifty frames without unbounded memory use, lost cancellation, misleading completion labels, or silent frame omission. Increasing capacity does not itself improve speed or establish whole-volume understanding.

**Status:** Background desktop jobs and sequential two-frame inference are implemented. The fifty-frame limit, per-batch input streaming, GPU/NPU acceleration, and embedding cache are proposals, not implemented or qualified capabilities. This section records the current implementation and questions for expert consultation; it does not supersede the broader planned independent/guided multi-series workflow.

#### Hardware And Evidence

| Candidate | Known configuration | Evidence and uncertainty |
| --- | --- | --- |
| Linux development laptop | Intel Core Ultra 7 268V; 8 cores/8 threads; Intel Lunar Lake integrated GPU and NPU; approximately 31 GiB RAM reported previously | CPU/GPU/NPU identified with `lscpu` and `lspci` on 2026-10-16. RAM is an earlier host reading, not a reserved inference budget. GPU driver/backend compatibility and usable GPU memory have not been qualified. |
| Android candidate, user-described "Moto Go5" | User reports 8 GB RAM and possible integrated GPU | Exact commercial name/model number, SoC, GPU, Android version, and available RAM are unverified. Do not assume a specific Motorola device or backend from this name. No inference measurement on this handset exists. |

Integrated graphics normally shares system RAM: it does not add dedicated memory to the inference budget. Android, the viewer, decoded frames, model weights, projector, KV cache, runtime buffers, and other applications must all fit within measured available resources. Neither total RAM nor GPU/NPU presence establishes feasibility.

#### Current Settings And Owning Code

| Surface | Current desktop implementation | Owner |
| --- | --- | --- |
| Model pair | MedGemma 1.5 4B IT `Q4_K_M` GGUF, approximately 2.49 GB; matching F16 projector, approximately 851 MB | Explicit local paths in Settings; pinned artifacts described earlier in this guide |
| Runtime | Isolated Rust subprocess; `llama-cpp-2 = 0.1.159`, default features disabled, `mtmd` enabled | [Worker dependencies](../../radassist-multiplatform/native/radassist-medgemma-worker/Cargo.toml) |
| Acceleration | `with_n_gpu_layers(0)` and `mtmd_params.use_gpu = false`; no GPU or NPU inference | [Rust worker](../../radassist-multiplatform/native/radassist-medgemma-worker/src/main.rs) |
| Threads and decoding | `RADASSIST_MEDGEMMA_THREADS` accepts 1-64; default available parallelism capped at 8; greedy sampling | Rust worker `thread_count` and sampling loop |
| Job lifecycle | Immutable study/series/frame/window/model/runtime snapshot; app-owned queue; explicit cancellation; navigation does not invalidate submitted work | [App orchestration](../../radassist-multiplatform/desktopApp/src/main/kotlin/com/nadigerventures/radassist/desktop/Main.kt), [job state and presentation](../../radassist-multiplatform/desktopApp/src/main/kotlin/com/nadigerventures/radassist/desktop/DesktopMedGemmaResearch.kt) |
| Inputs | 1-8 frames; same submitted WW/WL; job-owned PNGs rendered and retained before waiting for inference; identity checks and cleanup | [DICOM image adapter](../../radassist-multiplatform/shared/src/jvmMain/kotlin/com/nadigerventures/radassist/platform/jvm/viewer/DesktopDicomImageSource.kt) |
| Inference batches | Two frames; model/projector loaded once per Rust job; fresh 4,096-position context per batch; 128 output tokens; flushed ordered JSON events including truncation | Rust worker and [desktop process client](../../radassist-multiplatform/shared/src/jvmMain/kotlin/com/nadigerventures/radassist/platform/jvm/ai/DesktopMedGemmaClient.kt) |
| Resource/deadline bounds | 32 MiB/16 million pixels per image; 64 MiB/64 million pixels per request; six-minute Rust process deadline, not reset on batch progress | Desktop process client and DICOM adapter |
| UI and persistence | Four active jobs; twenty retained in-memory job records; concise findings and expandable provenance; manual copy/save; no automatic clinical-report insertion | Job state and [research record store](../../radassist-multiplatform/shared/src/jvmMain/kotlin/com/nadigerventures/radassist/platform/jvm/ai/DesktopMedGemmaResearchStore.kt) |
| Alternative runtime | Local llama.cpp CLI fallback runs two-frame batches serially but reloads weights per batch | Desktop process client |

The [completed background-job spec](../../_bmad-output/implementation-artifacts/spec-desktop-medgemma-background-batched-jobs.md) records acceptance criteria, tests, review findings, and remaining GUI qualification gaps. Runtime paths/settings are captured, but this does not make externally mutable source/model files immutable bytes; content-integrity policy remains a separate question.

#### Observed Latency And Benchmark Limits

The latest recorded four-image smoke used generated 512 x 512 RGB inputs, four CPU threads, and only eight requested output tokens per batch. It completed two ordered batches in approximately **259.7 seconds**. Earlier observations showed roughly one minute per synthetic image encode. These are development smoke results, not a controlled throughput benchmark, clinical-image evaluation, or CPU/GPU comparison. They suggest profiling image encoding first; they do not prove it dominates every request. The reproducible harness is [test_vision_pairing.py](../../tools/medgemma-bootstrap/test_vision_pairing.py).

For each candidate backend, record cold/warm model load, DICOM render, vision encoding, prompt evaluation, generation, time to first batch, total latency, peak process memory, thermal behavior, and viewer/dictation responsiveness. Hold model/projector, image pixels, prompt, batch size, and output budget fixed. Repeat runs and report variance; record backend/driver/build identity. Use synthetic inputs for development qualification. Do not log patient pixels, identifiers, prompts, or findings into performance diagnostics by default.

#### Proposed Fifty-Frame Design Conditions

- Keep two-frame inference contexts and bounded output; fifty selected frames means twenty-five serial batches, not one enlarged context.
- Separate the job selection limit from per-batch byte/pixel limits and define an explicit bounded staging/disk policy. Retaining all fifty rendered PNGs under the current aggregate budgets is not assumed safe or supported.
- Design a worker session/IPC boundary that receives batch paths while keeping model/projector loaded; merely launching the existing worker separately for each pair loses model reuse. Release batch-owned images only after native code stops using them.
- Use progress-aware startup/per-batch deadlines plus an overall job bound, with explicit cancellation and cleanup; the current whole-process six-minute deadline is insufficiently justified for fifty frames.
- Preserve original frame mapping and job identity, publish partial findings immediately, and identify processed versus unprocessed frames after failure/cancel. Do not silently sample or skip requested images.
- Keep inference concurrency bounded; measure the resource cost of staged inputs from multiple queued jobs. Streaming later inputs also requires a policy for source files changing after submission.

#### Speed Options And Expert Questions

1. **Vulkan GPU experiment:** Can this exact llama.cpp/MTMD revision offload both the language model and vision projector on Intel Lunar Lake? Verify driver support, build flags, operation coverage, actual GPU use, memory pressure, and CPU fallback. Benchmark before enabling; no speedup is promised. CUDA is not the backend for this Intel GPU.
2. **CPU tuning:** Benchmark 4, 6, and 8 threads on the laptop under sustained load, recording responsiveness and thermals as well as latency. More threads can worsen memory contention; the default already permits eight. Use AC power and record power mode when comparing runs.
3. **Reuse and caching:** Model reuse already exists within a job. Would bounded embedding reuse for overlapping requests help, and does this runtime expose a safe API? Cache keys must include exact rendered pixels, WW/WL/preprocessing, projector identity, and encoder configuration; define memory limits and deletion. This cache is not implemented.
4. **Precision/preprocessing changes:** Could projector quantization, another model quantization, or an explicitly selected processing resolution reduce cost? Such changes alter model inputs/numerics and require separate compatibility and quality evaluation; do not silently downsample CT frames.
5. **Android feasibility:** Confirm the exact "Moto Go5" model/SoC/GPU first. Is the 4B Q4 model plus F16 projector feasible within measured available memory on its 8 GB shared-RAM configuration? Which compatible backend supports the required vision operations, and what are peak RSS, crash/cancel behavior, latency, and sustained thermal limits? An NPU would require separate runtime support; presence alone is insufficient.
6. **Concurrency:** Is limited overlap of DICOM preparation and one inference useful? Avoid parallel model copies until measured throughput, memory-bandwidth, and viewer-impact evidence justifies them. Background execution improves responsiveness, not intrinsic inference speed.

**Android is a separate implementation:** It uses an AIDL service/JNI bridge rather than the desktop Rust batch protocol; current native limits are 2,048 context positions and up to 512 output tokens. Its arm64/API 26+ admission and approximate 2x model-plus-projector memory heuristic are not measured safe budgets. Clean pinned bootstrap provenance and on-device service/JNI, memory, latency, cancellation, and offline qualification remain open. No handset whitelist or guaranteed support for every 8 GB device is claimed.

**Non-negotiable constraints:** Keep all inputs/inference local and opt-in; retain explicit selected-frame coverage, uncertainty and partial/failure states; preserve full provenance and separate manual research saving; never automatically insert results into clinical reports. HF-fast/native processor parity and clinical accuracy remain unqualified. GPU execution, batching success, or faster output do not establish clinical reliability.

### Abnormality Evaluation, Not Validated Diagnosis

This mode produces testable, unverified predictions for independent radiologist assessment, not an autonomous final diagnosis or a clinical report. A radiology-style answer is not proof of accuracy. The mandatory warning, source provenance, separate copying, and explicit Save/history remain part of every result's workflow; no output is automatically promoted into clinical drafts.

For an authorized evaluation, use held-out, independently reviewed cases with reference findings that were not used to tune the prompt. Keep known reference diagnoses, reports, and annotations out of the model input during blinded testing; compare them only after predictions have been recorded. Record the exact prompt/version, model/projector/runtime identity, selected frame and WW/WL, and unedited output so predictions can be checked against the evidence actually supplied.

Track false-positive suspected abnormalities, missed reference findings visible in the supplied frame/window, and unsupported assertions such as invented history, measurements, contrast status, or whole-study claims. Separately record findings outside the selected frame/window as coverage limitations rather than treating them as demonstrated perception failures. Include inconclusive/nondiagnostic and negative cases, check that the model abstains when appropriate, and retain case-specific uncertainty and truncation/failure records. A radiologist should judge whether each tentative impression follows from its cited observations; do not replace this with model confidence scores. Use synthetic inputs only for development smoke tests. Patient-case evaluation requires separate authorization and data-handling controls and is not a development test fixture.

### Bounded Frame Sample Today; Whole-Volume Inference Is Not Qualified

MedGemma 1.5's model-level volume support is not qualification of this app, its GGUF conversion, or either local runtime for a volume workflow. The desktop action accepts an explicit inclusive range of 1-8 consecutive axial frames from the currently indexed series. The Kotlin path validates each source identity, renders the same WW/WL, and records every submitted SOP/frame identity. Both runtimes receive the ordered list; Rust constructs one MTMD marker per image. Count, per-image/aggregate bytes and pixels, and the existing combined image/text/output context budget are enforced. An oversized selection fails as a whole; it is never silently shortened.

This contiguous sample does not cover the whole series and does not establish model-level volumetric understanding. No coverage claim may be inferred from the selected range. Processor parity remains blocked: HF-fast and native image tensors differ, and neither local runtime is clinically qualified. Any future whole-volume workflow needs a separately approved coverage policy, geometry/provenance contract, resource and cancellation design, processor qualification, and independent evaluation. The existing GGUF chat template and greedy Rust sampling remain unchanged.

### Built and observed

- Pinned developer tooling builds llama.cpp and extracts the CT notebook without executing it.
- Linux host `llama-mtmd-cli` was built from commit `5c8854fd88319125125dc4b836c98b1b215b0f23`.
- A desktop Linux x86-64 research request launches a bundled `llama-cpp-2` MTMD worker on an explicitly selected bounded range of axial frames. The worker is statically linked and remains a child process for Kotlin-owned cancellation; its transient result is labeled unverified and is not written to the report.
- Android arm64 native libraries were cross-built, but not installed or run on a device.
- The Android APK now packages those native libraries and an isolated JNI inference service; APK assembly passed, but no device inference, memory/latency measurement, cancellation smoke, or offline network-denial check has run.
- Both local GGUF hashes were verified; generated one- and two-slice DICOM-derived inputs passed the bounded host multimodal runtime checks. The HF-fast/native image tensor comparison remains blocked as documented in Section 6.
- Eight synthetic DICOM preprocessing tests pass through JPEG encoding; no patient data is used.

### Full cross-platform workflow remains planned

The qualified implementation evidence is narrower than the broader design: the desktop default-off research action accepts an explicitly selected bounded axial-frame sample through either the bundled Rust MTMD worker or the pinned CLI fallback. Android inference code is implemented but on-device qualification remains open. The shared KMP adapter and multi-series workflow remain planned.

The product design places policy and orchestration in shared Kotlin Multiplatform code and runtime loading/inference in platform adapters. MedGemma is one adapter behind typed capability/model interfaces; it should not own viewer state or write the existing report. The intended workflow is explicit and default-off, with user-confirmed series, independent and guided modes, coverage accounting, cancellation, and a separate temporary AI report candidate. The normal DICOM viewer, dictation, and radiologist report remain unchanged.

A future workflow is expected to snapshot confirmed inputs, run qualified per-series preprocessing/inference, reconcile cross-series observations, compose one combined candidate with limitations, and publish only if all required checks pass. This is architecture intent, not a claim that those components or behaviors exist today. Review the [integration design](../medgemma-integration-design.md) for its full contracts and failure semantics.

## 9. Privacy, Safety, And Qualification Boundaries

- Use generated pixels for the checked-in smoke tests and processor/runtime qualification. The explicitly enabled desktop research action may process any user-selected patient image while all decoding, image handling, model loading, inference, and output remain local/offline. The transient result stays outside the report and is unverified; this permission is not clinical-use authorization.
- A de-identification label by itself does not approve patient data for automated test fixtures or publication. Existing external CT fixtures remain under review and are not approved by this guide.
- Do not send DICOM data to a server or allow model auto-download. Use explicit local model/projector/image paths and `--offline` for local experiments.
- A local CPU build, matching hashes, and one synthetic output do not establish diagnostic accuracy, sensitivity, specificity, robustness, CT preprocessing parity, or regulatory readiness.
- Google's official MedGemma page says the models are not clinical-grade and require validation for the developer's intended use. A qualified clinician and the organization's clinical, privacy, safety, and regulatory processes must govern any later evaluation or deployment decision.
- Keep the original files unchanged. Do not commit weights, patient data, PHI, or generated clinical candidates.

## 10. Qualification Checklist

| Gate | Current state | Evidence still needed |
| --- | --- | --- |
| Pinned artifact identity | **Passed for these local files** | Recheck hashes after any move/copy; provenance/signature and license obligations remain separate questions. |
| Host runtime build | **Passed on this Linux host** | Native Windows/macOS builds remain unverified. |
| Synthetic image path | **Single-slice and two-slice offline generated-DICOM runtime checks passed on Linux** | Network isolation, conversion/quantization parity and non-synthetic inputs remain unverified. |
| Text-token parity | **Passed for the tested rendered template (15/15 IDs)** | Broader prompt/template coverage is not implied by this one deterministic case. |
| Image-processor parity | **Blocked: HF fast vs native tensors differ on PNG and JPEG** | Resolve the resampler difference or explicitly re-qualify an accepted reference path; do not substitute HF slow silently. |
| Synthetic preprocessing | **Eight tests passed through JPEG encoding** | The strict fast-processor mismatch above prevents full model-input parity. |
| Performance budget | **Exploratory only** | Project owner must agree measurable limits; recorded Linux RSS/time are not pass/fail budgets. |
| Patient-fixture approval | **Not approved by this guide** | Authorized fixture review, privacy/suitability evidence, and independent reference annotations. |
| Android execution | **Inference service and arm64 runtime packaged; on-device inference not qualified** | This checkout's Android bootstrap manifest failed source preflight because its local llama.cpp checkout is dirty, so the current staged libraries lack successful bootstrap provenance. Rebuild from clean pinned sources, install on an arm64 device, measure memory/latency/cancellation/viewer impact, run synthetic and approved CT checks, and qualify CT input processing. |
| Product integration | **Partial desktop slice built** | The broader shared KMP adapter, multi-series coverage/reconciliation, report-candidate lifecycle, and end-to-end cross-platform privacy/lifecycle tests remain unimplemented. |
| Clinical use | **Not established** | Intended-use validation, risk management, human factors, and applicable regulatory assessment. |

## 11. Repository Evidence And Maintenance

This guide is the concise operational entry point; the following records retain source evidence and detail:

- [Feasibility audit and observed CT notebook behavior](medgemma-ct-feasibility.md)
- [Offline MedGemma integration design](../medgemma-integration-design.md)
- [Pinned source manifest](../../tools/medgemma-bootstrap/sources.json)
- [Bootstrap implementation](../../tools/medgemma-bootstrap/bootstrap.mjs)
- [Vision pairing and preprocessing-parity work specification](../../_bmad-output/implementation-artifacts/spec-medgemma-vision-pairing.md)
- [Native runtime bootstrap work specification](../../_bmad-output/implementation-artifacts/spec-medgemma-native-runtime-bootstrap.md)

When evidence changes, update this guide's evidence date, status, checklist and reproduction results alongside the detailed feasibility audit. Preserve dated historical observations instead of silently rewriting them. Do not promote a gate to passed based on plans, model-card claims, metadata strings, or a synthetic smoke test alone.

## 12. Study Companion: Follow The Evidence Chain

For a first pass, read Sections 1-3 to identify the exact model, projector, runtime, and CT reference. Then trace one input through Sections 5-7 and write down what changes at each boundary:

```text
DICOM pixels and metadata
  -> selected series/frames and physical ordering
  -> rescale/window/color conversion
  -> encoded image bytes
  -> runtime image resize/normalization
  -> vision features and language-model tokens
  -> generated text
```

Keep each step independently testable. A successful final response does not prove that every intermediate transformation matches the reference; a matching image tensor does not prove that the generated response is clinically correct.

### Evidence Ladder

Use these labels precisely when discussing results:

- **Artifact identity:** local bytes match a recorded hash/size; this is not proof of trusted origin or clinical fitness.
- **Runtime smoke:** the software loaded the selected artifacts and completed bounded inference on the tested synthetic input.
- **Preprocessing parity:** the candidate and reference produced matching/tolerated input tensors under a predeclared comparison. The current HF-fast gate is blocked as recorded above.
- **Input coverage:** the tested images/frames represent a declared coverage policy; a few selected frames do not become a whole-volume examination.
- **Predictive performance:** outputs are compared with independent, appropriate reference annotations on representative held-out cases.
- **Intended-use qualification:** performance, workflow, risks, privacy, platform behavior, and applicable regulatory requirements are assessed for the actual use.

Do not merge these into a single PASS label. Read the current status banner and qualification table before reusing any dated result. Follow the [AI Engineering Study Roadmap](../STUDY_ROADMAP.md) for the prerequisite concepts and companion runbooks.

