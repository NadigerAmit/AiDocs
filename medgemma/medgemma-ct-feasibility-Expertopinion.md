# MedGemma Qualification vs RadAssist Product Integration
## Final Recommendation and Conclusion

**Document Status:** Recommended Project Direction  
**Project:** RadAssist  
**Subject:** MedGemma Integration Qualification Review  
**Date:** October 2026

---

# Executive Summary

After completing the current qualification exercise, our conclusion is that **RadAssist should not block MedGemma integration while pursuing perfect preprocessing parity research**.

The team has already successfully demonstrated:

- MedGemma GGUF loading
- mmproj-F16.gguf integration
- Successful multimodal runtime execution
- Image ingestion
- Tokenizer consistency verification
- Single-image and multi-image inference
- Offline operation
- End-to-end image-to-text generation

These results establish that the core MedGemma deployment architecture is technically feasible and operational.

The remaining qualification gap concerns:

```text
Pixel-level preprocessing parity

between

Hugging Face Gemma3ImageProcessorFast

and

llama.cpp MTMD preprocessing
```

This is an important research topic but is no longer considered a blocker for beginning controlled RadAssist integration and radiologist evaluation.

---

# Key Observation

The current investigation has demonstrated:

```text
Runtime Integration:
PASS

Tokenizer Verification:
PASS

GGUF Runtime:
PASS

Projector Integration:
PASS

Image Input Processing:
PASS

Offline Inference:
PASS

Pixel-Level Equivalence:
NOT YET PROVEN
```

Therefore the remaining issue is not:

```text
Can MedGemma run?
```

The answer to that question is already:

```text
YES
```

The remaining question is:

```text
Can we prove exact preprocessing equivalence
with the Hugging Face Fast reference implementation?
```

---

# Important Distinction

There are two separate objectives.

## Objective A

Research Qualification

Goal:

```text
Prove exact equivalence with
Gemma3ImageProcessorFast.
```

This is primarily a scientific validation task.

---

## Objective B

RadAssist Product Validation

Goal:

```text
Determine whether MedGemma provides
useful clinical assistance
to radiologists.
```

This is a product validation task.

---

These two objectives should no longer be treated as the same workstream.

---

# Review of Current Findings

The investigation demonstrated:

## Proven

```text
✓ GGUF model loads correctly

✓ mmproj projector loads correctly

✓ Visual tokens are generated

✓ Image decoding functions correctly

✓ Prompt tokenization matches

✓ Inference completes successfully

✓ Offline execution is possible

✓ Multi-image workflows operate correctly
```

These are the most critical integration requirements for RadAssist.

---

## Not Yet Proven

```text
Exact equality between:

Hugging Face Fast Processor

and

llama.cpp MTMD Processor
```

particularly for:

```text
Non-square images

JPEG inputs

Resizing edge cases
```

---

# Analysis of the Mismatch

Current data strongly suggests that the primary source of divergence is:

```text
Image Resizing Strategy

Interpolation Method

Aspect Ratio Handling

Potential Cropping / Padding Policy
```

The evidence does NOT currently indicate problems with:

```text
Tokenizer

Normalization Parameters

Vision Projector

GGUF Runtime

Prompt Construction
```

This is important because it means the major architecture pieces are functioning correctly.

---

# Answer to Question 1
# What Should Be Considered the Reference?

Current proposal:

## Research Track

Continue to treat:

```text
Gemma3ImageProcessorFast
```

as the reference implementation.

This remains useful for scientific validation.

---

## Product Track

The product should not require bit-perfect tensor equivalence before field evaluation begins.

Instead we should establish:

```text
Behavioral Equivalence
```

rather than:

```text
Tensor Equivalence
```

---

In other words:

The critical question becomes:

```text
Does the model provide useful,
consistent clinical assistance?
```

rather than:

```text
Do all preprocessed pixels match exactly?
```

---

# Answer to Question 2
# How Should Native Matching Be Pursued?

The current evidence indicates that additional effort should focus specifically on:

```text
Image Decode
vs
Resize
```

as separate stages.

Recommended future research:

```text
Stage 1
Decoded RGB Comparison

Stage 2
Resize Comparison

Stage 3
Normalization Comparison

Stage 4
Vision Embedding Comparison
```

This work should continue as a dedicated research stream.

However:

```text
It should not block product integration.
```

---

# Answer to Question 3
# Should a Trusted External Preprocessor Be Used?

Technically:

```text
YES
```

A trusted preprocessing implementation could guarantee identical image inputs across environments.

Example:

```text
DICOM
  ↓
Trusted Preprocessor
  ↓
896×896 RGB
  ↓
llama.cpp + MedGemma
```

This is a valid future option.

---

However:

It introduces new challenges:

```text
Cross-platform maintenance

Desktop vs Android differences

Additional dependencies

More deployment complexity
```

For RadAssist Phase 1 this is not necessary.

---

# Answer to Question 4
# Should Tolerances Be Allowed?

For product validation:

```text
YES
```

but only if supported by downstream evaluation.

Recommended future metrics:

```text
Embedding Similarity

Clinical Finding Agreement

Response Consistency

Radiologist Evaluation Scores
```

The team should avoid defining arbitrary pixel tolerances without clinical validation.

The most meaningful validation is:

```text
Radiologist Review
```

rather than:

```text
Pixel Equality
```

---

# Answer to Question 5
# Can HF Slow Be Used As An Alternative?

Current evidence suggests:

```text
NO
```

at least not as a fully qualified substitute for:

```text
HF Fast
```

because current results show:

```text
Successful behavior on some inputs

Differences on JPEG inputs

Differences on non-square inputs
```

Therefore HF Slow should not currently be designated as a reference-equivalent implementation.

---

# Recommended Project Decision

## Decision

Proceed with MedGemma integration into RadAssist immediately.

Do not block integration on preprocessing parity research.

---

# Reasoning

The project has already proven:

```text
MedGemma can run.

Images can be processed.

Projector works.

GGUF deployment works.

Inference works.

Offline mode works.
```

These are the most important technical risks.

They have already been retired.

---

Meanwhile:

```text
Pixel parity research
```

remains important but does not prevent evaluation of actual user value.

---

# Recommended RadAssist Phase 1 Architecture

```text
CT / MRI Study
       |
       v

Rust DICOM Engine
       |
       v

Volume Construction
       |
       v

Representative Images
       |
       v

MedGemma GGUF
+
mmproj-F16.gguf
       |
       v

Clinical Question Answering

Report Assistance

Finding Summarization

Teaching Assistance

Observation Support
       |
       v

Radiologist Review
```

---

# Expected Near-Term Goals

The objective of RadAssist Phase 1 should NOT be:

```text
Automated Diagnosis
```

Instead:

```text
Radiologist Assistance
```

Examples:

```text
Summarize findings.

Draft preliminary report text.

Explain visual observations.

Provide educational context.

Answer clinical questions.

Support workflow efficiency.
```

---

# Why This Is The Correct Product Strategy

The value of MedGemma cannot be fully measured through preprocessing inspections alone.

The only way to determine whether MedGemma is useful for RadAssist is through:

```text
Real CT cases

Real MRI cases

Radiologist Feedback

Clinical Workflow Evaluation
```

Waiting for perfect preprocessing parity delays the collection of the most important data:

```text
User Feedback
```

---

# Deferred Research Activities

The following activities remain valuable but should be moved to a separate workstream:

```text
HF Fast vs MTMD parity research

Non-square resize investigations

JPEG decode investigations

Exact tensor comparison

Pixel-level regression framework

Vision embedding equivalence analysis

Cross-runtime reproducibility studies
```

These activities should continue in parallel without blocking application integration.

---

# Final Recommendation

The team should proceed with:

```text
MedGemma GGUF Integration

mmproj-F16.gguf Integration

Real CT Evaluation

Real MRI Evaluation

Radiologist Feedback Collection

Prompt Engineering

Workflow Validation
```

while continuing preprocessing parity investigation as an independent research effort.

In summary:

The recommendation is to continue bounded, default-off engineering integration and preprocessing research as separate workstreams. The evidence supports synthetic local runtime plumbing only; it does not establish processor equivalence, clinical performance, or approval for patient-case evaluation. Keep those claims and gates explicit in every project decision.

## Addendum: Scope Of The Recommendation

This opinion is a project-direction recommendation, not a qualification result or authorization to run clinical studies. Its recommendation to avoid blocking all engineering integration on exact HF-fast pixel equality should be read narrowly: isolated, default-off application plumbing and non-patient synthetic testing may proceed while the parity investigation continues.

It does **not** establish that the current native image path is behaviorally equivalent, that its response quality is acceptable, or that real CT/MRI evaluation can proceed. The current [Local Vision Guide](medgemma-local-vision-guide.md) records a blocked HF-fast tensor-parity gate, unqualified clinical performance, and explicit fixture/device constraints. A successful synthetic runtime test, matching token IDs, or a professional-looking answer cannot close those gates.

Exact tensor equality is not a universal scientific prerequisite for every possible model deployment. However, choosing a different processor or tolerance is a substantive model-input change, not a paperwork shortcut. Before making an equivalence or performance claim, define the accepted processor as part of the model system, predeclare numeric/behavioral criteria, and evaluate downstream behavior on independently reviewed, representative, authorized data. Until those conditions and the applicable privacy, safety, and governance approvals are met, keep patient-case evaluation and clinical use out of scope.

The original recommendation and dated evidence above are retained as historical analysis. Use the newest dated operational guide and audit for current implementation status, and use this document only for its decision rationale. For the full learning sequence, see the [AI Engineering Study Roadmap](../STUDY_ROADMAP.md).

We have already demonstrated that MedGemma and mmproj-F16.gguf are operational and technically viable within the target architecture. The remaining preprocessing parity work is valuable research, but it should no longer be considered a release blocker for controlled RadAssist evaluation. The project's next major milestone should be obtaining feedback from radiologists using real CT and MRI studies, because that feedback will provide significantly more product value than additional months of tensor-level preprocessing analysis.