# AI Engineering Study Roadmap

## Purpose

This is the entry point for the AI study material in this directory. It orders the existing guides from first principles through model architecture, deployment, and medical-imaging engineering. The source documents remain the detailed references; this page explains what to study first, what to build, and what evidence is needed before moving on.

The material is designed for a software engineer who is new to AI. You do not need to master every equation before writing code, but you should be able to explain the assumptions behind a model, measure its behavior, and distinguish a successful software run from evidence of model quality.

## How To Study

For each topic:

1. Read for the main idea, then write a short explanation without copying the wording.
2. Work through a small example by hand or with a short script.
3. Implement or run the smallest useful experiment.
4. Record the dataset, split, model/configuration, metric, and limitations.
5. Explain what your result proves and what it does not prove.

Keep experiments reproducible: pin dependencies and model revisions, seed randomness where practical, separate training/validation/test data, and do not put private data or model artifacts in source control.

## Learning Sequence

### Stage 0: AI And Mathematical Foundations

Start with [AI Basic Phase 0](Basics/AI_basic_phase0.md). Learn the vocabulary for data, features, labels, vectors, probability, loss, optimization, and generalization. Review the foundations in [AI Interview Questions](Ai%20interviewQuestions.md) after each chapter, not as a substitute for understanding.

**Move on when you can:** explain the difference between AI, ML, and deep learning; describe a model as a parameterized function; distinguish a prediction from a decision; and explain why a test set must remain separate from model selection.

### Stage 1: Classical Machine Learning And Evaluation

Use the supervised-learning, classification, regression, validation, metrics, and leakage sections in [AI Interview Questions](Ai%20interviewQuestions.md). Then study the production-oriented example in [Intent Classification](intent_classification_guide.md), beginning with TF-IDF and a linear classifier before embeddings.

**Practice:** build a small intent classifier with a fixed label set. Split by user/session where appropriate, report per-class precision/recall and a confusion matrix, test unsupported inputs, and choose a rejection policy using validation data only.

**Move on when you can:** explain why accuracy can mislead, why thresholds are not universal, how data leakage happens, and how an unknown or ambiguous input should be handled safely.

### Stage 2: Neural Networks And Model Architectures

Read the neural-network, CNN, U-Net, RNN, Transformer, attention, ViT, diffusion, and multimodal sections in [AI Architecture Study Guide](AI_Architecture_Study_Guide.md). Use [AI Interview Questions](Ai%20interviewQuestions.md) for retrieval practice and fill gaps in optimization, normalization, regularization, and model evaluation.

**Practice:** trace a tensor through a small classifier, then compare image classification with semantic segmentation. Draw the input/output tensor shapes for a U-Net and the Q/K/V attention operation. Do not assume that a model name alone defines preprocessing or output semantics.

**Move on when you can:** explain training versus inference, logits versus probabilities, the roles of convolution and attention, and why a model architecture does not specify the complete data pipeline.

### Stage 3: Local LLMs And AI-Assisted Development

Study [Aider + Ollama + Qwen Guide](Aider_Ollama_Qwen_Guide.md) to understand a local coding-assistant stack. Treat model tags, hardware estimates, and commands as version-sensitive; verify current runtime and model documentation before applying them to a new machine.

**Practice:** ask a local coding model to explain a small change, review its proposed diff, run the relevant tests yourself, and record where it was wrong. Keep approval and Git review in the loop; local execution is not by itself a security or privacy guarantee.

### Stage 4: Model Deployment And Systems Engineering

Read [ONNX and Rust Engineering](onnx-and-rust-engineering-guide.md), then the [TotalSegmentator ONNX Phase 0 Runbook](TotalSegmenter/totalsegmentator-onnx-phase-0.md). Learn the distinction between a checkpoint, computation graph, runtime, preprocessing pipeline, and application. Follow the runbook's approved evidence gates; its current evidence and limitations are more authoritative than generic examples in introductory material.

Study [TotalSegmentator Chunked Inference](TotalSegmenter/TOTALSEGMENTATOR-CHUNKED-INFERENCE.md) after understanding volume geometry and sliding-window inference. Focus on tensor parity, coordinate systems, memory budgets, cancellation, atomic publication, and platform-specific gaps.

**Practice:** create a synthetic tensor, compare reference and candidate outputs, record maximum and mean numerical error, and test shape/dtype failures. Do not substitute synthetic parity for representative-input or clinical validation.

### Stage 5: Medical Vision, Qualification, And Responsible Use

Read the [MedGemma Local Vision Guide](medgemma/medgemma-local-vision-guide.md) first. Use [MedGemma CT Feasibility](medgemma/medgemma-ct-feasibility.md) for detailed dated evidence and [MedGemma Qualification Opinion](medgemma/medgemma-ct-feasibility-Expertopinion.md) as a recommendation that must be interpreted against the measured gates, not as a replacement for them.

**Practice:** make an evidence table with separate rows for artifact identity, runtime loading, preprocessing parity, input coverage, independent accuracy, resource limits, and intended-use approval. Mark each as passed, blocked, not measured, or not applicable; do not promote a result from one row to another.

**Move on when you can:** explain the difference between a synthetic smoke test, numerical parity, segmentation agreement, clinical performance, and regulatory readiness. These claims require different evidence.

## Document Map

| Document | Best time to read | Main purpose |
|---|---|---|
| [AI Basic Phase 0](Basics/AI_basic_phase0.md) | First | Intuition and foundational vocabulary |
| [AI Interview Questions](Ai%20interviewQuestions.md) | Alongside each stage | Review questions, explanations, and retrieval practice |
| [AI Architecture Study Guide](AI_Architecture_Study_Guide.md) | After fundamentals | Model-family and architecture overview |
| [Intent Classification Guide](intent_classification_guide.md) | Classical ML stage | Embedded text classification and tool-routing design |
| [Aider, Ollama, Qwen Guide](Aider_Ollama_Qwen_Guide.md) | LLM tooling stage | Local coding-assistant setup and workflow |
| [ONNX and Rust Engineering Guide](onnx-and-rust-engineering-guide.md) | Deployment stage | Model export, runtime, tensor contracts, and evidence |
| [TotalSegmentator ONNX Phase 0](TotalSegmenter/totalsegmentator-onnx-phase-0.md) | Deployment stage | Export and parity experiment runbook |
| [Chunked TotalSegmentator Inference](TotalSegmenter/TOTALSEGMENTATOR-CHUNKED-INFERENCE.md) | Advanced deployment | Full-volume execution, bounded memory, and publication contracts |
| [MedGemma Local Vision](medgemma/medgemma-local-vision-guide.md) | Medical AI stage | Local multimodal runtime and current qualification status |
| [MedGemma CT Feasibility Audit](medgemma/medgemma-ct-feasibility.md) | Medical AI stage | Dated experiment evidence and unresolved qualification gates |
| [MedGemma Qualification Opinion](medgemma/medgemma-ct-feasibility-Expertopinion.md) | After the audit | Decision analysis and integration recommendation |

## Technical Reading Rules

- **Architecture is not a product pipeline.** A neural network may be only one stage among decoding, orientation, resampling, normalization, inference, postprocessing, and presentation.
- **A score is not automatically a probability.** Logits, similarity scores, classifier confidence, and calibrated probabilities are different quantities.
- **A successful run is not an accuracy result.** State the exact input, software/model identity, comparison, metric, and scope of every claim.
- **Evaluation must match deployment.** Use patient- or group-level splits for correlated records, time-aware splits for temporal prediction, and held-out data for final evaluation.
- **Medical AI needs independent evidence.** Agreement with another implementation of the same model can demonstrate engineering parity, not clinical correctness.
- **Human review is not a substitute for qualification.** Define intended use, failure handling, privacy controls, and independent evaluation before a high-stakes workflow.
- **Treat time-sensitive details as dated evidence.** Package versions, model availability, hardware performance, policy, and regulatory requirements change; re-check the cited source before relying on them.

## Suggested Capstone

Build a small, non-clinical prototype that classifies a fixed set of text intents and safely routes only allowlisted actions. Include a held-out test set, an unsupported-intent path, per-class metrics, schema validation, explicit confirmation for side effects, and a short model card describing intended use and limitations. Then write a separate design note explaining what would need to change before the same engineering approach could be evaluated in medical imaging.