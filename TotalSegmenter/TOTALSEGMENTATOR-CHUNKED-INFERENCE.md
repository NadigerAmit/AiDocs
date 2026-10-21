# TotalSegmentator Full-Range Chunked Inference

**Status:** The desktop Rust worker's per-axis streaming cubic resampler passes the approved preprocessing-tensor and full-volume fixture gates (r10). Android now uses the same per-axis streaming design and its component tests pass; Android whole-volume fixture parity remains unmeasured. This is engineering parity only, not release approval or clinical validation.

**Model:** TotalSegmentator v2.5.0 `part_organs`, ONNX SHA-256 `d5e4709a0db84d353b9c615acb518075971968c08b5047192dda15db43bbab2f`.

## What Changed For Parity

The model conversion was not changed to obtain the passing result. The automated candidate used the same pinned ONNX graph SHA shown above; the correction was in the application-side volume pipeline. Its input orientation/canonicalization, nonzero crop, CT normalization, order-3 cubic resampling, global sliding-window placement, Gaussian overlap blending, and order-1 logit restoration were aligned with the pinned nnU-Net reference. The approved whole-volume comparison therefore isolates the improvement to preprocessing and surrounding inference/postprocessing behavior with the converted graph held fixed. This is evidence for this graph and fixture, not proof of conversion quality for every possible input or of clinical accuracy.

**Logits** are the model's raw per-class scores before probability conversion or class selection. The pipeline blends overlapping patches in logit space, restores the scores to the native grid, and then selects the class with the highest score (`argmax`). Interpolating already-selected class labels instead can change boundaries and reduce parity.

**Dice** measures overlap between the candidate label and reference label: $\mathrm{Dice}=\frac{2|A\cap B|}{|A|+|B|}$. It ranges from 0 (no overlap) to 1 (identical masks). The reported mean `0.999949` is the arithmetic mean over the 19 present/nonempty classes in this fixture; it is not a probability that a segmentation is correct and does not establish performance on other scans.

**Reference:** nnU-Net 2.5.1 CPU, 1.5 mm model spacing, 128-cube patches. Approved whole-volume gates remain geometry/laterality agreement, Dice >= 0.97 for every present class, and volume delta <= 2%.

## Design Contract

A selected range is one volume, not a collection of independently inferred subvolumes. Both platforms preserve the global model-grid coordinates and sliding-window schedule; output chunks only bound the temporary logit accumulator. Every native-grid output slice is produced from the complete set of model patches that can contribute to it. A patch crossing a chunk boundary is included on both sides of that boundary, and its contribution is blended before labels are finalized.

Only monotonic progress is published while work is running. No intermediate labelmaps or organ measurements are exposed. Cancellation, scratch failure, or a stale source revision prevents final publication; owned temporary files are cleaned up on success and failure.

## Memory And Chunk Budgets

| Resource | Rust desktop worker | Android inference |
|---|---:|---:|
| Source geometry cap | 128M voxels | 256M voxels |
| Model-grid cap | 64M voxels | 64M voxels |
| Preprocessing storage | Private mmap-backed source, per-axis stages, and target; at most 2 GiB for each simultaneously live adjacent-stage pair | Private mapped crop, per-axis stages, model, label, and output staging; total preflight capped at 2 GiB plus a 16 MiB reserve |
| Logit accumulator cap | 512 MiB | 32 MiB |
| Patch | 128 x 128 x 128 | 128 x 128 x 128 |
| Output behavior | Final native-grid labelmap only | Final native-grid labelmap only |

The Rust 64M-voxel cap still applies to the target model grid. Cubic resampling now processes one axis at a time into private mapped stages: source-to-X, X-to-Y, then Y-to-target-Z. Each predecessor is dropped after the next stage is complete, so the 2 GiB byte check bounds adjacent stage pairs rather than a full coefficient cube plus target. These are scratch-file size limits, not a claim that mmap pages can never enter the operating-system page cache.

The per-axis resampler keeps only a reusable 1D filter workspace in heap memory; each whole-axis output stage is file-backed. It does not independently filter fixed-halo slabs, which would change the recursive cubic boundary conditions. The streaming stages preserve the original 12-sample recursive boundary extension and four-tap interpolation support, then pass the same pinned tensor comparison.

Android derives slab depth from model-plane size and its 32 MiB accumulator budget. Its preflight separately estimates Java heap and ONNX Runtime native memory. Full-width example geometry of 255 x 255 x 299 (512 x 512 x 449 source at approximately 0.746 x 0.746 x 1.0 mm) uses 4 model-grid slices per accumulator chunk at that cap; the shared planner regression verifies 75 contiguous chunks, no gaps, and each slab within budget. For this geometry, source, X stage, Y stage, and target contain 117,702,656, 58,621,440, 29,196,225, and 19,442,475 voxels respectively. The estimated Rust peak adjacent-stage scratch falls from 953,266,496 bytes for source plus the old padded coefficient cube to 705,296,384 bytes for source plus the X stage (26.0% lower). Android's preflight estimate falls from 1,266,441,708 to 1,135,256,496 bytes (10.4% lower), including crop, stage, target, and native-output buffers. The target remains below the unchanged 64M model-grid cap.

A budget refusal is expected behavior. Do not bypass source/target caps, heap/native preflight, or available-storage checks to force a run.

## Processing Paths

### Rust desktop worker

1. Read and validate the geometry-stamped snapshot into a private mmap-backed source buffer.
2. Validate source orientation, reorient to the model axes, crop to the nonzero region, and apply the pinned CT HU normalization.
3. Resample to 1.5 mm with the parity-tested cubic B-spline implementation, streaming one axis per file-backed stage and releasing each predecessor; target-grid voxels remain capped at 64M.
4. Build one global 128-cube sliding-window schedule and Gaussian importance map. Process contiguous output slabs, retaining the patch halo/global offsets needed at each boundary; normalize accumulated logits and restore labels to the full native grid.
5. Write to a temporary result; the owning viewer/session revalidates the source revision and selected range before atomically publishing only the complete labelmap.

The chunk planner uses `(classCount + 1) * sizeof(float)` bytes per model voxel (class logits plus blend weight), capped by patch depth. Slab depth can be smaller than the patch depth. Patch context remains global; the chunk is never passed to the model as a separate patient volume.

The current slab strategy can evaluate a global patch again when it intersects multiple output slabs. Smaller budgets therefore reduce peak accumulator memory at the cost of repeated inference; profile the actual target device before making latency claims.

### Android

1. Validate the model hash and snapshot header, then map the source snapshot read-only. Reorient the supported LPS grid into model axes while finding the nonzero crop, and copy only that crop into private mapped scratch.
2. Apply pinned CT HU normalization before per-axis cubic B-spline resampling to the 1.5 mm model grid. Preflight model dimensions, both intermediate stages, total scratch including native-output staging, available app-cache space, estimated heap, and ONNX Runtime native memory.
3. Use nnU-Net's rounded global patch starts and Gaussian importance map. Partition only the output depth according to the accumulator budget; patches retain global coordinates and may be recomputed when they overlap multiple slabs.
4. Normalize and linearly resample logits to the cropped native grid before argmax, restore crop labels to LPS native coordinates, and write a temporary complete labelmap. Publish only after inference and cancellation checks succeed; delete owned scratch and failed output files in `finally` cleanup.

Android's per-axis resampler, patch schedule, Gaussian blending, and native-grid logit restoration are algorithmically aligned with Rust. Unit tests cover pinned 1D and nonconstant 3D cubic tensors, constant edge behavior, stage scratch sizing, window placement, Gaussian weights, cross-slab interpolation, and crop-to-native orientation. The identity-orientation guard still rejects other orientations rather than mapping them. Android whole-volume fixture parity has not been run; this is an Android validation gap, not a MedGemma dependency. Therefore, Android parity is implemented at component level but is not yet measured against the approved fixture.

## Progress, Cancellation, And Publication

- Progress is monotonic and remains below 100 until the complete output has been written and hashed.
- The UI retains only progress/cancel state during processing. Existing viewer/session code owns source-revision and selected-range validation before publication.
- Cancellation is checked during snapshot validation, preprocessing, patch inference, and output restoration.
- Scratch is private to the app/worker operation. Cleanup failure is surfaced; incomplete output is deleted rather than returned as a result.
- Do not expose partial masks or per-organ measurements from a slab. Measurements may be computed only from the final validated labelmap.
- Each nonempty organ measurement exposes `lengthMm`, `breadthMm`, and `depthMm` from the inclusive native-grid bounding box along X, Y, and Z, respectively; `dimensionAxisOrder` makes that mapping explicit in JSON. These are grid-aligned dimensions, not orientation-independent anatomical long axes. Empty labels keep dimensions and volume null.

## Case-Level Measurement History

The viewer can retain a validated summary for comparison across series in the same DICOM study. The shared history model requires consistent valid StudyInstanceUID and SeriesInstanceUID values, but persists only domain-separated SHA-256 digests, a local case UUID, series display metadata, run IDs, coverage, provenance, and measurement summaries. It does not persist raw DICOM UIDs, patient names, source paths, or SOP Instance UIDs. Records remain local to the app: Android uses app-private `filesDir` with `AtomicFile`; desktop uses `~/.radassist` with an atomic replacement and file lock.

History is saved only after the complete segmentation and measurement summary pass source revision, output hash, model/class-map hash, and geometry checks. It retains at most three studies, evicting the least recently updated when adding another. Full-series and ROI runs are separate coverage variants; only a successful rerun of the same series and exact coverage replaces an earlier run. A changed source revision is shown as historical. Users can delete one case, and app-data reset clears all case summaries. Failed or corrupt/future-schema data is not silently replaced. This history is neither synchronized nor inserted into a clinical report. See the [approved feature spec](../../_bmad-output/implementation-artifacts/spec-case-level-organ-measurements.md) for the complete behavior contract.

## Evidence And Reproduction

The only permitted model-parity fixture for this feature is the approved upstream `example_ct.nii.gz`, SHA-256 `dbd3ae6d614d1d7ef3a46925c30c70038ed52da6b37fdd6afdc7b9b71387e3e1`. Synthetic geometry is appropriate for planner and budget tests; do not use user CT/MRI data for model-parity runs.

A fresh run of the final axis-streamed Rust release worker passed the approved whole-volume gates: private report run `totalseg-app-preproc-parity-20261017-r10`, worker SHA-256 `69edce0621a1793a7b495309ce0da06f63589a4cb3401568a759a63db1e9a12c`, unchanged ONNX SHA-256 `d5e4709a0db84d353b9c615acb518075971968c08b5047192dda15db43bbab2f`, all approved nonempty-class gates passed, mean Dice `0.999949`, minimum Dice `0.999790`, maximum volume delta `0.000420`, no failed classes, matching shape/spacing/affine, and correct liver/spleen laterality. Worker runtime was `279.8 s` vs `114.5 s` for the pinned CPU reference, so no speed improvement is claimed. The approved fixture hash was verified before the run; reports and all fixture-derived outputs remain outside the repository.

A separate paired timing check around the 2026-10-18 review-UI change used the same release worker, ONNX graph, and approved fixture. Both runs passed all parity gates and produced the same candidate labelmap SHA-256 `be507a06683a0b867e68876fa6508b8b27f29955287889c7c3d730ea8b7c5359`. The pre-UI worker/reference times were `341.03 s` / `126.08 s`; post-UI they were `331.48 s` / `124.26 s`. This shows no observed worker-time degradation, but the `9.55 s` difference is not a speedup claim: same-worker runs have varied from `279.8 s` to `341.0 s`. These timings cover the isolated inference worker, not viewer rendering or Android analysis. Private reports: `totalseg-ui-workspace-baseline-20261018` and `totalseg-ui-workspace-after-20261018`.

The current Rust worker also passed pinned preprocessing-tensor parity in the same private run: crop tensors were exact; normalization maximum absolute error was `1.1920929e-7`; cubic-resampled maximum absolute error was `3.5762787e-7`, below the approved `1e-4` tolerance. The tested resampled shape was 224 x 202 x 244. This establishes preprocessing parity for the Rust worker, not the independent Android implementation.
The same r10 run passed pinned preprocessing-tensor parity: crop tensors were exact; normalization maximum absolute error was `1.1920929e-7`; cubic-resampled maximum absolute error was `3.5762787e-7`, below the approved `1e-4` tolerance. The tested resampled shape was 224 x 202 x 244. This establishes desktop Rust preprocessing parity, not whole-volume Android parity.

The r10 reference reported 18 nonempty anatomical labels: 9 masks touch the approved fixture's field-of-view boundary (colon, liver, left/right lower lung lobes, right middle lung lobe, left upper lung lobe, prostate, spleen, and stomach), and 9 do not. Boundary contact is a possible truncation signal. No boundary contact is not proof of anatomical completeness. The UI reports per-organ empty, possible-truncation, or no-boundary-contact-but-unverified status for the selected coverage; it never asserts anatomical completeness from this check.

Run the focused tests from the repository root:

```bash
rtk cargo test --manifest-path radassist-multiplatform/native/radassist-ai-worker/Cargo.toml
rtk proxy env -C radassist-multiplatform ./gradlew --no-configuration-cache :shared:allTests :androidApp:testDebugUnitTest :desktopApp:test
```

When the approved private fixture and model assets are available again, rebuild the worker and run the asset-gated official whole-volume comparison. Keep the manifest, fixture, graph, worker outputs, and report outside the repository; use a new immutable private output path for each run:

```bash
uv run --project tools/model-factory/totalseg-onnx python -m phase0.cli worker-volume-parity \
  --experiment "$EXPERIMENT/experiment.json" \
  --fixture "$APPROVED_FIXTURE" \
  --onnx "$APPROVED_ONNX" \
  --worker "$BUILT_WORKER" \
  --output "$PRIVATE_OUTPUT/worker-volume-parity.json"
```

The report must identify the worker hash and pass the already-approved Dice, volume, geometry, and laterality gates. Also add a corresponding Android-vs-reference comparison before declaring cross-platform parity; a Rust pass cannot certify the independent Android preprocessing path.

## Test Coverage

- Shared planner tests cover contiguous range coverage, context halos, a budget too small for one plane, and the 255 x 255 x 299 full-range Android slab budget.
- Rust worker tests cover scratch-size overflow/refusal, full-width staged scratch estimates, singleton and multiaxis cubic parity against the full-buffer reference, global chunk planning, and existing preprocessing/blending behavior; all 34 worker tests pass.
- Android inference tests cover snapshot validation, streamed-stage scratch estimates, pinned 1D and nonconstant 3D cubic tensors, constant edge behavior, chunk boundaries, the pinned window schedule and Gaussian map, cross-slab order-1 restoration, crop-to-native orientation, cancellation, and output cleanup. These component tests do not replace an Android whole-volume model run.
- The Rust official-fixture preprocessing and whole-volume gates pass on r10. Android whole-volume fixture parity remains required before claiming measured cross-platform parity; no MedGemma gate is involved.
