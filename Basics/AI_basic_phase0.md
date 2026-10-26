# AI Mastery Roadmap
# Phase 0 (Expanded Edition)
# Foundations of Artificial Intelligence, Machine Learning, and Deep Learning

Author: AI Study Roadmap for Amit Nadiger
Target Audience:
- Software Engineers
- Architects
- Medical Imaging Developers
- AI Beginners
- Engineers transitioning into AI

Prerequisites:
None

Difficulty:
Beginner

Estimated Study Time:
40-60 Hours

Primary Goal:
Build deep intuition for AI before studying any algorithms, neural networks, or transformers.

---

# Why Phase 0 Exists

Most AI courses start like this:

```python
from sklearn.linear_model import LogisticRegression
```

Students can train models.

But they do not understand:

- What a model actually is.
- Why learning occurs.
- Why data is needed.
- What probabilities mean.
- Why transformers work.
- Why embeddings work.
- Why LLMs hallucinate.

This phase solves that problem.

Think of Phase 0 as building the foundation of a skyscraper.

Without foundation:

Phase 5 (Transformers)
and
Phase 7 (LLMs)

become difficult.

With foundation:

Everything becomes straightforward.

---

# Chapter 1
# Understanding Intelligence

Before understanding Artificial Intelligence we must answer:

What is Intelligence?

Interestingly, scientists still disagree on a perfect definition.

However most definitions include:

1. Learning
2. Reasoning
3. Adaptation
4. Planning
5. Problem Solving
6. Decision Making

---

## Human Example

Imagine a child touches a hot stove.

Observation:
Stove is hot.

Experience:
Pain.

Learning:
Hot stove causes pain.

Future Decision:
Avoid touching stove.

This demonstrates intelligence.

The system:

Observe
→ Learn
→ Adapt

This is exactly what AI attempts to imitate.

---

## Intelligence Exists on a Spectrum

Examples:

Calculator
↓
Thermostat
↓
Dog
↓
Human
↓
Future AI Systems

A calculator follows rules.

A dog learns.

Humans reason.

Modern AI can perform some reasoning.

---

# Key Takeaway

Intelligence is not knowledge.

Intelligence is the ability to acquire and use knowledge.

---

# Chapter 2
# What Is Artificial Intelligence?

Artificial Intelligence (AI) is the science of creating machines capable of performing tasks that normally require human intelligence.

---

## Examples

### Speech Recognition

Input:

Audio

Output:

Text

Example:

"Open YouTube"

↓

Open YouTube

---

### Image Recognition

Input:

Image

Output:

Object Name

Example:

Cat Image

↓

Cat

---

### Medical Imaging

Input:

CT Scan

Output:

Tumor Detection

---

### Large Language Models

Input:

Question

Output:

Answer

Example:

ChatGPT
Gemini
Claude
Llama
MedGemma

---

# Important Understanding

AI is NOT one thing.

AI is a massive field.

Contains:

- Machine Learning
- Deep Learning
- Computer Vision
- Robotics
- Planning Systems
- Expert Systems
- Natural Language Processing

---

# Chapter 3
# The History of AI

---

## 1950

Alan Turing asks:

Can machines think?

Introduces:

Turing Test

---

## 1956

The term:

Artificial Intelligence

is officially created.

Birth of AI.

---

## 1980s

Expert Systems become popular.

Computers use hand-written rules.

Example:

IF fever AND cough
THEN flu

Problem:

Too many rules required.

---

## 1997

IBM Deep Blue defeats chess champion Garry Kasparov.

Huge milestone.

---

## 2012

Deep Learning Revolution

AlexNet wins ImageNet competition.

Error rate dramatically reduced.

Neural network era begins.

---

## 2017

Google introduces:

Transformer Architecture

Paper:

Attention Is All You Need

AI changes forever.

---

## 2022+

ChatGPT revolution.

AI becomes mainstream.

---

# Chapter 4
# Pattern Recognition: The Real Heart of AI

This chapter is perhaps the most important in Phase 0.

Everything in AI eventually becomes:

Pattern Recognition.

---

## Facial Recognition

Humans see:

Eyes
Nose
Mouth

Pattern detected:

Person

---

## Medical Example

Radiologist sees:

Lung CT

Identifies:

Ground Glass Opacity

Pattern recognized.

Diagnosis suggested.

---

## LLM Example

Input:

"The capital of France is"

Pattern learned:

Paris

Output:

Paris

---

# Ultimate Truth

AI does not think like humans.

AI recognizes patterns.

Very advanced patterns.

But still patterns.

---

# Chapter 5
# Data: The Fuel of AI

A car needs fuel.

AI needs data.

No exception.

---

# Data Types

## Structured Data

Database Tables

Examples:

Patient records

| Age | Weight | Disease |
|-------|---------|---------|
| 45 | 78 | Yes |

---

## Unstructured Data

Examples:

- CT scans
- MRI
- Images
- Audio
- Videos
- Text

Most modern AI works on unstructured data.

---

# Your RadAssist Example

Input:

DICOM CT

This is data.

Without this:

No AI.

---

# Rule

Better Data > Better Algorithm

Often collecting better data gives larger gains than changing algorithms.

---

# Chapter 6
# Information, Knowledge and Intelligence

These three terms confuse many people.

---

## Data

Raw Numbers

Example:

38.5

---

## Information

Meaning

Example:

Temperature = 38.5°C

---

## Knowledge

Understanding

Example:

Patient likely has fever.

---

## Intelligence

Action

Example:

Administer treatment.

---

# Hierarchy

Data
↓
Information
↓
Knowledge
↓
Intelligence

AI systems attempt to move upward.

---

# Chapter 7
# Mathematical Thinking

We now reach mathematics.

Do not fear mathematics.

AI uses mathematics because computers understand numbers.

Nothing more mystical than that.

---

# Example

Consider:

Height = 170 cm

Weight = 75 kg

Computer stores:

170
75

Everything becomes numbers.

---

# AI Principle

Reality
→ Numbers
→ Computation
→ Prediction

---

# Chapter 8
# Understanding Vectors

The single most important mathematical object in AI.

---

# What Is A Vector?

A vector is simply:

An ordered list of numbers.

Example:

[170, 75]

Could represent:

Height
Weight

---

# Patient Example

Patient:

Age = 50

Weight = 72

Heart Rate = 90

Vector:

[50,72,90]

---

# Why Vectors Matter

Every ML model consumes vectors.

Every Deep Learning model consumes vectors.

Every LLM consumes vectors.

Every Transformer consumes vectors.

Everything eventually becomes vectors.

---

# Chapter 9
# Dimensions

Each number inside a vector is called a dimension.

Example:

[1,2]

2 Dimensions

---

Example:

[1,2,3,4,5]

5 Dimensions

---

## Medical Example

Patient

Age
Height
Weight
Blood Pressure
Blood Sugar

Vector

[50,170,75,120,95]

5-dimensional data.

---

## AI Reality

Modern AI commonly uses:

512 dimensions

1024 dimensions

4096 dimensions

8192 dimensions

or more.

---

# Chapter 10
# Embeddings

Most important concept for future LLM learning.

---

## What Is An Embedding?

An embedding is a vector representing meaning.

Example:

King

↓

[0.82,1.2,-0.4...]

Queen

↓

[0.81,1.18,-0.5...]

Vectors become close because meanings are similar.

---

# Why This Matters

ChatGPT

Gemini

Claude

Llama

MedGemma

all operate on embeddings.

Not words.

Not sentences.

Embeddings.

---

# Chapter 11
# Distance and Similarity

How do we compare vectors?

Distance.

---

## Example

Dog

[1,2]

Wolf

[1.1,2.1]

Distance small.

Similarity high.

---

Cat

[100,200]

Distance large.

Similarity low.

---

This powers:

Semantic Search

RAG

Vector Databases

Recommendation Systems

---

# Chapter 12
# Probability

Real world contains uncertainty.

AI handles uncertainty through probability.

---

## Example

Weather

Tomorrow Rain?

Probability:

80%

Not certainty.

But likely.

---

## Medical Example

Cancer Probability

95%

Model highly confident.

Still not 100%.

---

# Important Principle

AI produces probabilities.

Humans make final decisions.

Especially in medicine.

---

# Chapter 13
# Statistics

Statistics is the science of learning from data.

Machine Learning is impossible without statistics.

---

# Mean

Average

Example

10 20 30

Mean:

20

---

# Median

Middle value.

Useful against outliers.

---

# Mode

Most frequent value.

---

# Variance

Measures spread.

Low Variance:

Values close together.

High Variance:

Values spread out.

---

# Standard Deviation

Most common measure of variability.

Important for:

- Data understanding
- Preprocessing
- Normalization

---

# Chapter 14
# Correlation

Variables that move together.

---

## Example

Height ↑

Weight ↑

Positive correlation.

---

Example

Outdoor Temperature ↑

Heater Usage ↓

Negative correlation.

---

# Golden Rule

Correlation

DOES NOT MEAN

Causation

---

Example

Ice Cream Sales ↑

Drowning Accidents ↑

Not causal.

Both increase because:

Summer increases.

---

# Chapter 15
# What Is A Model?

Most misunderstood AI term.

A model is simply:

A mathematical function that converts input into output.

---

Example

Input:

Patient Data

↓

Model

↓

Disease Prediction

---

Notation:

y=f(x)

That's a model.

Nothing more.

---

# Chapter 16
# Learning

What does it mean for AI to learn?

AI adjusts internal parameters using data.

---

Human:

Study
Practice
Improve

---

AI:

Train
Adjust Parameters
Improve

Same concept.

Different implementation.

---

# Chapter 17
# Error and Loss

How do we know model is wrong?

Measure error.

---

Example

Actual House Price

100

Predicted

90

Error

10

---

Loss Function

Converts error to score.

Training minimizes this score.

---

# Chapter 18
# Optimization

Optimization:

Process of reducing loss.

---

Mental Model

AI repeatedly performs:

Predict

↓

Measure Error

↓

Adjust Parameters

↓

Predict Again

↓

Repeat

Millions of times.

---

# Chapter 19
# Why Neural Networks Exist

Some problems are too complex.

Example:

Write rules for:

"What is a dog?"

Impossible.

Thousands of edge cases.

---

Neural Networks solve this by learning automatically.

Instead of programming rules.

We program learning.

---

# Chapter 20
# Deep Learning Intuition

Traditional Programming

Rules + Data

↓

Answer

---

Deep Learning

Data + Answer

↓

Learn Rules

---

This single change transformed AI.

---

# Chapter 21
# AI Through the Lens of RadAssist

You understand software architecture.

Let's map everything.

---

Input

CT Scan

↓

Vector Representation

↓

Neural Network

↓

Segmentation Model

↓

Organ Masks

↓

Measurements

↓

Clinical Insights

---

Everything we studied appears here.

Data

Vectors

Model

Probability

Optimization

Learning

Neural Networks

All connected.

---

# Phase 0 Final Mental Model

Reality
↓
Data
↓
Numbers
↓
Vectors
↓
Model
↓
Prediction
↓
Probability
↓
Decision

This single pipeline explains nearly all modern AI systems.

---

# Self Assessment

You have completed Phase 0 when you can explain:

1. What is AI?
2. What is ML?
3. What is a model?
4. Why data matters?
5. What is a vector?
6. What is an embedding?
7. What is probability?
8. What is statistics?
9. What is learning?
10. What is optimization?
11. Why neural networks exist?
12. How RadAssist fits into AI?

without looking at notes.

If yes,

You are ready for:

# Phase 1
# Python, NumPy, Pandas and Scientific Computing for AI

where we start touching code for the first time.

---

# Phase 0 Technical Notes And Practice

This section adds precision to the mental models above. Use it alongside the earlier chapters; it does not replace them.

## Important Refinements

### Models do not all consume the same kind of vector

Most machine-learning systems operate on numerical arrays called tensors, but the input structure depends on the task. A tabular model may consume a feature vector; a CNN consumes a tensor with spatial axes; a language model consumes token IDs and then learned embeddings; a segmentation model returns a spatial tensor of scores. The statement that "everything becomes vectors" is a useful first approximation, not a complete description of model interfaces.

An embedding is a learned representation, not a universal dictionary of meaning. Similarity is meaningful only relative to the embedding model, its training objective, the input preprocessing, and the chosen distance or similarity function. Two vectors from unrelated embedding models should not be compared as though they shared a coordinate system.

### A prediction is not a decision

A model may emit a score, logit, probability estimate, mask, or generated text. A separate decision rule turns that output into an action. For a binary classifier, for example, a threshold converts a probability estimate into a positive/negative decision. The threshold should reflect the task's costs and be selected on validation data, not assumed to be 0.5 or selected using the final test set.

Model confidence is not automatically calibrated probability. Calibration asks whether predictions assigned probability near 0.8 are correct about 80% of the time over a suitable evaluation population. Calibration can change under distribution shift and must be measured when probabilities drive decisions.

### Learning is an optimization procedure, not human-like understanding

Training adjusts parameters to improve an objective on observed examples. It does not prove that a model has learned a causal explanation or will behave correctly outside its training and evaluation conditions. A model can exploit shortcuts and still achieve a low loss. Hold-out evaluation, subgroup analysis, stress tests, and domain review are ways to look for those failures.

Likewise, "AI recognizes patterns" is a useful introduction but not a complete account of every AI system. Rule-based systems, search/planning, probabilistic models, and learned models use different mechanisms; modern applications often combine several of them.

## Worked Example: From Scores To A Safe Action

Suppose an intent model returns scores for `tune_channel`, `search_guide`, and `unsupported`:

```text
tune_channel: 0.47
search_guide: 0.45
unsupported: 0.08
```

The largest score is not enough to justify tuning a channel: the top two classes are close, and a score is not known to be calibrated. A production policy could ask a clarifying question when the top-two margin is small, reject low-confidence or out-of-domain text, validate the selected channel against an allowlist, and require confirmation before a consequential action. This is application policy around the model, not something the model guarantees by itself.

## Study Exercises

1. For a house-price example, identify one feature, one label, model parameters, one hyperparameter, the loss, and an evaluation metric.
2. Draw the data flow for image classification and image segmentation. Label tensor shapes and identify where preprocessing happens.
3. Explain why embeddings from two different models cannot automatically be compared by cosine similarity.
4. Given a confusion matrix, calculate precision and recall, then describe which error is more costly for the hypothetical application.
5. Describe a data split for multiple scans from the same patient and explain why a random image-level split can leak patient-specific information.
6. Give one example of correlation without causation and one reason a predictive model may still fail after that correlation changes.

## Readiness Check

Before moving to the coding phase, be able to distinguish: feature from label; parameter from hyperparameter; training loss from evaluation metric; logit from probability; model output from application decision; and engineering parity from real-world performance. If a term is unclear, return to the relevant chapter and make a one-sentence definition plus a concrete example.

Continue with the directory's [AI Engineering Study Roadmap](../STUDY_ROADMAP.md), then proceed to Python, NumPy, Pandas, plotting, and basic experimentation before training neural networks.
