# AI Architecture Study Guide
### Author: Amit Nadiger
### Purpose: Understanding AI Models from Basics to Advanced Architectures

---

# Table of Contents

1. What is an AI Model?
2. AI Model Families
3. Traditional Machine Learning
4. Neural Networks
5. CNN (Convolutional Neural Networks)
6. U-Net
7. RNN
8. LSTM
9. GRU
10. Transformer
11. Attention Mechanism
12. Large Language Models (LLMs)
13. Vision Transformers (ViT)
14. Conformer
15. Diffusion Models
16. Multimodal Models
17. Real-world Medical AI Pipeline
18. Models and Technologies Used in RadAssist
19. ONNX vs GGUF
20. TF-IDF vs MiniLM
21. Summary Cheat Sheet
22. Architecture Selection And System-Level Reasoning

---

# 1. What is an AI Model?

An AI model is simply a mathematical function that learns patterns from data.

```text
Input
 ↓
Model
 ↓
Output
```

Examples:

```text
CT Scan      → Segmentation Mask
Text         → Intent
Speech       → Transcript
Image        → Diagnosis
```

The model learns these mappings during training.

---

# 2. AI Model Families

```text
Artificial Intelligence
│
├── Traditional Machine Learning
│   ├── Logistic Regression
│   ├── SVM
│   ├── Random Forest
│   └── XGBoost
│
└── Deep Learning
    ├── CNN
    ├── U-Net
    ├── RNN
    ├── LSTM
    ├── GRU
    ├── Transformer
    ├── Conformer
    ├── Vision Transformer
    ├── Diffusion Models
    └── Multimodal Models
```

---

# 3. Traditional Machine Learning

Examples:

```text
TF-IDF + Logistic Regression
SVM
Naive Bayes
Random Forest
XGBoost
```

Advantages:

- Very fast
- Small memory usage
- Easy to train
- Excellent for intent classification

Disadvantages:

- Cannot understand images
- Limited language understanding
- Requires handcrafted features

Example:

```text
"Open Sony TV"

↓

TF-IDF Vector

↓

Classifier

↓

TUNE_CHANNEL
```

---

# 4. Neural Networks

Basic concept:

```text
Input
 ↓
Neuron
 ↓
Output
```

Many neurons together form:

```text
Input Layer
 ↓
Hidden Layers
 ↓
Output Layer
```

This becomes a Deep Neural Network.

---

# 5. CNN (Convolutional Neural Networks)

Best suited for images.

Uses small filters called kernels.

Example:

```text
3×3 kernel
```

CNN learns:

- Edges
- Shapes
- Textures
- Anatomical structures

Pipeline:

```text
Image
 ↓
Convolution
 ↓
Feature Maps
 ↓
Pooling
 ↓
Classification
```

Applications:

- CT analysis
- MRI analysis
- Face Recognition
- Object Detection

---

# 6. U-Net

Most important architecture in medical imaging.

Designed specifically for segmentation.

Structure:

```text
           Encoder
Input --------\
               \
             Bottleneck
               /
Output <------/
           Decoder
```

Looks like the letter "U".

Hence:

```text
U-Net
```

Example:

```text
Input:
CT Scan

Output:
Liver Mask
```

Applications:

- Organ Segmentation
- Tumor Segmentation
- Medical Measurements

---

# 7. RNN

RNN = Recurrent Neural Network.

Designed for sequence data.

Example:

```text
I
 ↓
love
 ↓
medical
 ↓
AI
```

Advantages:

- Understands sequences

Problems:

- Slow
- Cannot handle very long sequences

Known issue:

```text
Vanishing Gradient Problem
```

---

# 8. LSTM

Long Short-Term Memory.

Improved version of RNN.

Contains:

```text
Input Gate
Forget Gate
Output Gate
```

Capabilities:

- Longer memory
- Better sequence understanding

Applications:

- Speech Recognition
- NLP
- Time Series

---

# 9. GRU

GRU = Gated Recurrent Unit

A simplified LSTM.

Advantages:

- Faster
- Smaller
- Less memory

Used in edge devices and embedded AI.

---

# 10. Transformer

Revolutionary architecture published in:

```text
Attention Is All You Need
```

Google (2017)

Main innovation:

```text
Attention Mechanism
```

Instead of processing words one by one:

```text
Word1
Word2
Word3
```

Transformer processes everything simultaneously.

Advantages:

- Parallel processing
- Huge context windows
- Faster training
- Better accuracy

---

# 11. Attention Mechanism

Sentence:

```text
The liver is enlarged because it...
```

Question:

```text
What does "it" mean?
```

Transformer learns:

```text
it → liver
```

This process is called:

```text
Self Attention
```

Every word can attend to every other word.

---

# 12. Large Language Models (LLMs)

Examples:

```text
GPT
Llama
Gemini
Mistral
Qwen
MedGemma
```

Most LLMs are Transformers.

Pipeline:

```text
Text
 ↓
Tokenizer
 ↓
Transformer Layers
 ↓
Predicted Tokens
```

Example:

```text
User:
Describe liver anatomy

↓

Generated Answer
```

---

# 13. Vision Transformers (ViT)

Transformer adapted for images.

Instead of words:

```text
Image Patches
```

Example:

```text
512×512 Image

↓

16×16 patches

↓

Transformer
```

Advantages:

- Better global understanding
- Long-range relationships
- Strong medical imaging performance

---

# 14. Conformer

Conformer =

```text
CNN + Transformer
```

Combines:

### CNN Strength

```text
Local Features
```

### Transformer Strength

```text
Global Context
```

Applications:

```text
Speech Recognition
Voice Assistants
Medical Dictation
```

Pipeline:

```text
Input
 ↓
CNN Block
 ↓
Attention Block
 ↓
Feed Forward
 ↓
Output
```

---

# 15. Diffusion Models

Used for image generation.

Examples:

```text
Stable Diffusion
DALL-E
MidJourney
```

Concept:

```text
Image
 ↓
Add Noise
 ↓
Complete Noise
```

Training learns reverse process.

Generation:

```text
Noise
 ↓
Denoising
 ↓
Generated Image
```

---

# 16. Multimodal Models

Can process multiple modalities.

Examples:

```text
Image + Text
Speech + Text
Video + Text
```

Examples:

```text
GPT-4o
MedGemma
Gemini
LLaVA
Qwen-VL
```

Medical Example:

```text
CT Scan
+
Question

↓

AI Answer
```

---

# 17. Real-world Medical AI Pipeline

Example:

```text
CT Scan
 ↓
TotalSegmentator
 ↓
Organ Masks
 ↓
Measurements
 ↓
MedGemma
 ↓
Clinical Explanation
 ↓
Radiologist
```

This is very similar to RadAssist.

---

# 18. Models & Technologies Used in RadAssist

---

## TotalSegmentator

Purpose:

```text
Automatic organ segmentation
```

Detects:

- Liver
- Kidney
- Spleen
- Pancreas
- Vertebrae
- Aorta
- Hundreds of structures

Output:

```text
Segmentation Masks
```

---

## nnU-Net

Core technology used by TotalSegmentator.

Purpose:

```text
Medical image segmentation
```

Based on:

```text
U-Net
```

Automatically configures training and preprocessing.

---

## MedGemma

Google's medical multimodal model.

Inputs:

```text
Medical Image
+
Medical Question
```

Outputs:

```text
Medical Reasoning
Report Generation
Clinical Answers
```

Architecture:

```text
Transformer
+
Vision Encoder
```

Use Cases:

- Radiology QA
- Report Assistance
- Clinical Understanding

---

## MedSAM

Medical Segment Anything Model.

Purpose:

```text
Promptable Segmentation
```

Example:

```text
User clicks liver

↓

Model segments liver
```

Architecture:

```text
Vision Transformer
```

---

## MiniLM

Lightweight language model.

Purpose:

```text
Sentence Embeddings
```

Converts text into vectors.

Example:

```text
"Open Sony TV"

↓

384D embedding vector
```

Applications:

- Semantic Search
- Intent Matching
- Retrieval Systems

Advantages:

- Small
- Fast
- Offline capable

---

## TF-IDF

Term Frequency - Inverse Document Frequency.

Traditional NLP feature extraction.

Example:

```text
turn on tv

↓

Sparse Vector
```

Advantages:

- Extremely fast
- Tiny memory footprint
- Easy training

Recommended for:

```text
Intent Classification
```

---

## ONNX

Open Neural Network Exchange.

Purpose:

```text
Portable AI model format
```

Example:

```text
PyTorch

↓

ONNX

↓

Rust
C++
Java
Android
iOS
```

Benefits:

- Cross-platform
- Fast inference
- Runtime optimized

RadAssist currently uses:

```text
ONNX Runtime
```

for segmentation.

---

## GGUF

GPT-Generated Unified Format

Optimized model format for:

```text
llama.cpp
```

Purpose:

```text
Run LLMs locally
```

Advantages:

- Quantized
- Smaller
- Offline
- CPU Friendly

Examples:

```text
medgemma.gguf
llama.gguf
mistral.gguf
```

---

## mmproj-F16.gguf

Multimodal Projector model.

Purpose:

```text
Image Features
 ↓
Language Model Features
```

Acts as a bridge.

Without mmproj:

```text
LLM cannot understand images.
```

Pipeline:

```text
Image
 ↓
Vision Encoder
 ↓
mmproj
 ↓
LLM
 ↓
Answer
```

This is why MedGemma image analysis requires:

```text
MedGemma
+
mmproj-F16.gguf
```

---

# 19. ONNX vs GGUF

| Feature | ONNX | GGUF |
|----------|----------|----------|
| Purpose | General AI Runtime | Local LLM Runtime |
| Ideal For | CNN/U-Net/ViT | LLMs |
| Runtime | ONNX Runtime | llama.cpp |
| Medical Segmentation | Excellent | Not Used |
| Chat Models | Limited | Excellent |
| Offline | Yes | Yes |

---

# 20. TF-IDF vs MiniLM

| Feature | TF-IDF | MiniLM |
|----------|----------|----------|
| Speed | Very Fast | Fast |
| Memory | Tiny | Small |
| Semantic Understanding | Poor | Good |
| Synonym Understanding | No | Yes |
| Training | Easy | Usually Pretrained |
| Offline | Excellent | Excellent |

Examples:

```text
Open Sony TV
```

and

```text
Switch to Sony Channel
```

TF-IDF:

```text
Different
```

MiniLM:

```text
Very Similar
```

---

# 21. Summary Cheat Sheet

| Technology | Category |
|------------|-----------|
| TF-IDF | Traditional NLP |
| Logistic Regression | Traditional ML |
| CNN | Image Processing |
| U-Net | Medical Segmentation |
| RNN | Sequential Data |
| LSTM | Advanced RNN |
| GRU | Lightweight LSTM |
| Transformer | Modern AI Backbone |
| ViT | Vision Transformer |
| Conformer | CNN + Transformer |
| MedSAM | Medical Segmentation |
| TotalSegmentator | Organ Segmentation |
| MedGemma | Medical Multimodal LLM |
| MiniLM | Embedding Model |
| ONNX | General AI Model Format |
| GGUF | LLM Model Format |
| mmproj | Image-to-LLM Bridge |

---

# Final Advice for RadAssist

Current Stack:

```text
DICOM CT
 ↓
TotalSegmentator (ONNX)
 ↓
Organ Masks
 ↓
Measurements
 ↓
MedGemma + mmproj
 ↓
Medical Reasoning
 ↓
Radiologist
```

This is already aligned with how many modern medical AI assistants are built.

Recommended learning order:

1. TF-IDF
2. MiniLM
3. CNN
4. U-Net
5. Vision Transformer
6. Transformer
7. Attention
8. Multimodal Models
9. TotalSegmentator
10. MedGemma
11. RAG Systems
12. AI Agents

Mastering these topics will give you a strong foundation to design and extend enterprise-grade medical AI systems such as RadAssist.

---

# Architecture Selection And System-Level Reasoning

The architecture names above are starting points, not complete deployment specifications. A useful design describes the whole path from source data to a validated application result.

## Match The Model To The Task

| Task | Common output | Architecture families to consider | Important evaluation |
|---|---|---|---|
| Tabular classification | Class scores or probabilities | Logistic regression, trees, gradient boosting, MLP | Per-class precision/recall, calibration, subgroup performance |
| Text intent classification | Intent plus score or abstention | TF-IDF with linear model, encoder embeddings with classifier | Confusion matrix, out-of-domain rejection, latency on target hardware |
| Image classification | Class scores | CNN, ViT, hybrid | Sensitivity/specificity, calibration, site/device shift |
| Semantic segmentation | Per-voxel class scores or labels | U-Net/nnU-Net, transformer hybrids | Per-class overlap, geometry, laterality, boundary and empty-mask handling |
| Text generation | Token sequence | Autoregressive Transformer | Task-specific factuality, coverage, abstention, safety and human review |
| Image generation | Image or latent sample | Diffusion, GAN, VAE families | Fidelity, diversity, conditioning adherence, misuse risk |

Architecture selection should also consider available training data, inference memory, latency, supported operators, model/update licensing, device support, and the cost of failure. A larger or newer model is not automatically a better fit.

## The Full Inference System

For an imaging workflow, document each boundary explicitly:

```text
Source format
  -> decode and validate
  -> orientation / geometry handling
  -> model-specific preprocessing
  -> tensor contract (shape, dtype, axes, units)
  -> model runtime
  -> logits / probabilities / generated output
  -> postprocessing and label mapping
  -> source-grid restoration and provenance
  -> user-facing review and failure handling
```

Each arrow can introduce defects. Version and test preprocessing separately from the neural-network graph. Preserve source geometry and provenance, and distinguish incomplete coverage from negative findings. A plausible overlay or fluent explanation is not proof of correctness.

## Important Qualifications To Earlier Simplifications

- CNNs are effective for local spatial structure, but are not limited to images and do not guarantee translation invariance; convolution is translation-equivariant under idealized conditions, while padding, stride, and pooling affect that property.
- Transformers can process positions in parallel during training, but autoregressive generation still proceeds token by token. Standard self-attention also has sequence-length and memory costs; architecture and runtime optimizations change the practical trade-offs.
- U-Net is widely used in segmentation, not the only or universally best medical-imaging architecture. Model choice depends on task, labels, data, compute, and validation.
- A vision-language model output is generated text conditioned on inputs, not an independently verified diagnosis. A human review step does not convert an unvalidated model into a clinically validated device.
- ONNX is a graph interchange format and GGUF is an ecosystem-specific model container; neither format alone guarantees speed, portability, or accuracy.
- TF-IDF does not understand meaning by itself. MiniLM embeddings can encode semantic similarity, but similarity is not intent certainty, and application-specific evaluation and rejection logic remain necessary.

## Design Review Checklist

Before calling an architecture ready for an application, answer:

1. What is the intended use, user, input, output, and explicit out-of-scope behavior?
2. Which exact model weights, code, tokenizer, preprocessing, class map, and runtime are used?
3. What are the tensor shapes, axis order, numeric range, coordinate system, and output interpretation?
4. Which independent data represents the deployment population, and how are patient/site/time leakage prevented?
5. Which metrics and predeclared thresholds represent the important failure modes?
6. What happens on invalid inputs, unsupported cases, low confidence, timeout, cancellation, or resource exhaustion?
7. What evidence is engineering parity, what is predictive performance, and what remains unmeasured?

Use the [AI Engineering Study Roadmap](STUDY_ROADMAP.md) for the recommended order through the detailed inference and qualification guides.
