# 40 Important AI and Machine Learning Interview Questions with Detailed Answers

A practical interview preparation guide covering AI/ML fundamentals, mathematics, classical algorithms, deep learning, Transformers, LLMs, RAG, fine-tuning, and production deployment.

## Table of Contents

1. [AI, ML, and Deep Learning](#1-what-are-artificial-intelligence-machine-learning-and-deep-learning)
2. [Supervised, Unsupervised, and Reinforcement Learning](#2-what-are-supervised-unsupervised-and-reinforcement-learning)
3. [Features, Labels, Parameters, and Hyperparameters](#3-what-are-features-labels-parameters-and-hyperparameters)
4. [Training, Validation, and Test Sets](#4-explain-training-validation-and-test-datasets)
5. [Overfitting and Underfitting](#5-what-are-overfitting-and-underfitting)
6. [Gradient Descent and Backpropagation](#6-explain-gradient-descent-and-backpropagation)
7. [Loss Functions, Optimizers, and Metrics](#7-what-is-the-difference-between-loss-functions-optimizers-and-metrics)
8. [Precision, Recall, F1, and Confusion Matrix](#8-explain-precision-recall-f1-score-and-the-confusion-matrix)
9. [Bias–Variance Trade-off](#9-explain-the-biasvariance-trade-off-and-regularization)
10. [Classification, Regression, Clustering, and Dimensionality Reduction](#10-compare-classification-regression-clustering-and-dimensionality-reduction)
11. [Artificial Neural Networks](#11-what-is-an-artificial-neural-network)
12. [CNNs, RNNs, and Transformers](#12-compare-cnns-rnns-and-transformers)
13. [Transformers and Self-Attention](#13-what-is-a-transformer-and-how-does-self-attention-work)
14. [Tokenization, Embeddings, and Positional Information](#14-what-are-tokenization-embeddings-and-positional-information)
15. [Pretraining, Fine-tuning, and Inference](#15-compare-pretraining-fine-tuning-and-inference)
16. [How LLMs Generate Text](#16-what-is-an-llm-and-how-does-it-generate-text)
17. [Retrieval-Augmented Generation (RAG)](#17-what-is-rag-and-how-is-it-different-from-fine-tuning)
18. [LoRA and QLoRA](#18-what-is-lora-and-how-does-parameter-efficient-fine-tuning-work)
19. [Vector Databases and Semantic Search](#19-what-are-vector-databases-and-embeddings)
20. [Evaluating and Deploying AI](#20-how-do-you-evaluate-and-deploy-an-ai-model)
21. [Linear and Logistic Regression](#21-what-is-the-difference-between-linear-and-logistic-regression)
22. [Decision Trees, Random Forests, and Boosting](#22-explain-decision-trees-random-forests-and-gradient-boosting)
23. [Support Vector Machines](#23-what-is-a-support-vector-machine)
24. [K-Means Clustering](#24-how-does-k-means-clustering-work)
25. [PCA](#25-what-is-principal-component-analysis-pca)
26. [Feature Engineering and Scaling](#26-what-are-feature-engineering-and-feature-scaling)
27. [Class Imbalance](#27-how-do-you-handle-class-imbalance)
28. [Cross-Validation and Data Leakage](#28-what-are-cross-validation-and-data-leakage)
29. [ROC-AUC and PR-AUC](#29-what-are-roc-auc-and-pr-auc)
30. [Vanishing and Exploding Gradients](#30-what-are-vanishing-and-exploding-gradients)
31. [Batch, Epoch, and Learning Rate](#31-explain-batch-size-epochs-and-learning-rate)
32. [Batch Normalization and Layer Normalization](#32-compare-batch-normalization-and-layer-normalization)
33. [Dropout and Weight Decay](#33-what-are-dropout-and-weight-decay)
34. [Generative vs. Discriminative Models](#34-compare-generative-and-discriminative-models)
35. [Diffusion Models and GANs](#35-what-are-diffusion-models-and-gans)
36. [Prompt Engineering and Structured Outputs](#36-what-is-prompt-engineering-and-how-do-you-get-structured-outputs)
37. [Hallucinations and Grounding](#37-why-do-llms-hallucinate-and-how-can-you-reduce-it)
38. [Quantization and Model Optimization](#38-what-is-model-quantization)
39. [MLOps and Model Monitoring](#39-what-is-mlops-and-why-is-it-important)
40. [Responsible AI, Privacy, and Safety](#40-what-are-the-key-responsible-ai-considerations)

---

## 1. What are Artificial Intelligence, Machine Learning, and Deep Learning?

**Answer**

Artificial Intelligence (AI) is the broad field of building systems that perform tasks associated with intelligence, such as reasoning, perception, language understanding, and decision-making.

Machine Learning (ML) is a subset of AI in which systems learn patterns from data instead of relying only on manually written rules.

Deep Learning (DL) is a subset of ML that uses neural networks with multiple layers to learn increasingly complex representations.

**Example: image analysis**
- Traditional programming: a developer writes explicit rules.
- Machine learning: a model learns patterns from labeled examples.
- Deep learning: a neural network learns features such as edges, textures, and higher-level structures.

**Key takeaway:** AI is the broad goal, ML learns from data, and DL uses multilayer neural networks.

## 2. What are supervised, unsupervised, and reinforcement learning?

**Answer**

- **Supervised learning:** Learns from input-output examples with labels. Examples include classification and regression.
- **Unsupervised learning:** Finds structure in data without explicit target labels. Examples include clustering and some dimensionality-reduction methods.
- **Reinforcement learning (RL):** An agent interacts with an environment, receives rewards, and learns a policy to maximize cumulative reward.
- **Self-supervised learning:** Creates training signals from the data itself, such as predicting the next token in a language model.

**Examples**
- Supervised: classify an X-ray using expert-labeled examples.
- Unsupervised: group similar images.
- RL: teach a robot to navigate an environment.
- Self-supervised: train a model to predict masked or subsequent tokens.

**Key takeaway:** These approaches differ mainly in the learning signal available to the model.

## 3. What are features, labels, parameters, and hyperparameters?

**Answer**

- **Features (`X`):** Inputs, such as pixels, text tokens, age, or temperature.
- **Labels (`y`):** Target answers for supervised learning.
- **Parameters:** Values learned during training, such as weights and biases.
- **Hyperparameters:** Settings chosen for training, such as learning rate, batch size, model depth, and regularization strength.

A linear model can be written as:

\[
\hat y = w_1x_1+w_2x_2+b
\]

Here, `x` values are features, `w` values and `b` are learned parameters, and the learning rate is a hyperparameter.

**Key takeaway:** Training learns model parameters; validation and experimentation help select hyperparameters.

## 4. Explain training, validation, and test datasets.

**Answer**

- **Training set:** Used to update model parameters.
- **Validation set:** Used to compare models, tune hyperparameters, and choose when to stop training.
- **Test set:** Kept separate for final evaluation after model choices are complete.

A possible split is 70% training, 15% validation, and 15% test, though the best split depends on data size and the problem.

Repeatedly tuning against the test set leaks information from the test set into the development process and makes the final score less trustworthy.

For medical imaging, split at the **patient level** when appropriate. Images from the same patient should not appear in both training and test sets because patient-specific characteristics can inflate measured performance.

**Key takeaway:** Train to learn, validate to select, and test to estimate generalization.

## 5. What are overfitting and underfitting?

**Answer**

**Overfitting** occurs when a model learns the training data too specifically, including noise or accidental correlations. It performs well on training data but poorly on unseen examples.

**Underfitting** occurs when the model is too simple, insufficiently trained, or poorly optimized to capture the underlying patterns.

| Observation | Likely issue |
|---|---|
| Low training error, high validation error | Overfitting |
| High training error, high validation error | Underfitting |
| Low training and validation error | Potentially good fit; test independently |

Ways to reduce overfitting:
1. Use regularization or weight decay.
2. Apply appropriate data augmentation.
3. Use dropout where suitable.
4. Apply early stopping based on validation performance.
5. Gather more representative data.
6. Reduce model complexity if necessary.

To address underfitting, consider a more expressive model, better features, longer training, or less aggressive regularization.

**Key takeaway:** The objective is generalization to unseen data, not memorization of training examples.

## 6. Explain gradient descent and backpropagation.

**Answer**

Gradient descent is an optimization method that adjusts model parameters to reduce a loss function. Backpropagation calculates how the loss changes with respect to those parameters.

A typical training iteration:
1. Run a forward pass to produce predictions.
2. Calculate loss against the target.
3. Use backpropagation and the chain rule to compute gradients.
4. Use an optimizer to update parameters.
5. Repeat over batches and epochs.

The gradient descent update is:

\[
\theta_{t+1}=\theta_t-\eta\nabla_\theta L(\theta_t)
\]

Where `θ` represents parameters, `L` is the loss, `∇L` is the gradient, and `η` is the learning rate.

If `w = 2`, the gradient is `4`, and the learning rate is `0.1`, then:

\[
w_{\mathrm{new}}=2-0.1(4)=1.6
\]

Common optimizers include SGD, SGD with momentum, RMSProp, and Adam.

**Key takeaway:** Backpropagation computes gradients; the optimizer uses them to update parameters.

## 7. What is the difference between loss functions, optimizers, and metrics?

**Answer**

- **Loss function:** The objective the model is trained to minimize.
- **Optimizer:** The algorithm that updates model parameters.
- **Evaluation metric:** A measure used to assess whether the model performs well for the intended task.

| Task | Common loss | Common metrics |
|---|---|---|
| Binary classification | Binary cross-entropy | Precision, recall, F1, ROC-AUC |
| Multiclass classification | Cross-entropy | Accuracy, macro-F1 |
| Regression | MSE or MAE | RMSE, MAE, R² |
| Segmentation | Dice loss or cross-entropy | Dice, IoU |

Mean squared error is:

\[
\mathrm{MSE}=\frac{1}{N}\sum_{i=1}^{N}(y_i-\hat y_i)^2
\]

Binary cross-entropy is:

\[
L=-[y\log(p)+(1-y)\log(1-p)]
\]

A model can have low training loss and still generalize poorly. Select metrics that reflect real-world error costs.

**Key takeaway:** Loss guides learning; an optimizer changes parameters; metrics help judge usefulness.

## 8. Explain precision, recall, F1-score, and the confusion matrix.

**Answer**

A binary confusion matrix contains:
- **True positive (TP):** Positive correctly identified.
- **False positive (FP):** Negative incorrectly identified as positive.
- **False negative (FN):** Positive missed.
- **True negative (TN):** Negative correctly identified.

Formulas:

\[
\mathrm{Precision}=\frac{TP}{TP+FP}
\]

\[
\mathrm{Recall}=\frac{TP}{TP+FN}
\]

\[
F1=2\frac{\mathrm{Precision}\times\mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}
\]

If a model predicts 100 positive cases, 80 are truly positive, and there are 100 actual positive cases in total, then precision is 80%, recall is 80%, and F1 is 80%.

Accuracy can be misleading on imbalanced data. If only 1% of cases are positive, predicting every case as negative achieves 99% accuracy but detects no positives.

**Key takeaway:** Precision measures reliability of positive predictions; recall measures the proportion of actual positives detected.

## 9. Explain the bias–variance trade-off and regularization.

**Answer**

- **High bias:** The model is too simple to capture important patterns.
- **High variance:** The model is overly sensitive to training data and does not generalize consistently.

A conceptual decomposition of expected squared prediction error is:

\[
\mathrm{Expected\ error}\approx
\mathrm{Bias}^2+\mathrm{Variance}+\mathrm{Irreducible\ noise}
\]

This decomposition applies under specific assumptions but is useful for understanding generalization.

L2 regularization adds a penalty to large parameter values:

\[
L_{\mathrm{total}}=L_{\mathrm{data}}+\lambda\sum_j w_j^2
\]

The hyperparameter `λ` controls the penalty. Too little regularization may encourage overfitting; too much can cause underfitting.

**Key takeaway:** Balance model flexibility against its ability to generalize.

## 10. Compare classification, regression, clustering, and dimensionality reduction.

**Answer**

- **Classification:** Predicts a category, such as spam/not spam. Algorithms include logistic regression, decision trees, SVMs, and neural networks.
- **Regression:** Predicts a numeric value, such as price or temperature. Algorithms include linear regression and regression trees.
- **Clustering:** Groups similar data without predefined target labels. Examples include K-means and DBSCAN.
- **Dimensionality reduction:** Represents data using fewer dimensions. Examples include PCA and learned embeddings.

PCA finds directions of high variance in numerical data. Those directions are not necessarily the most relevant to a supervised target.

**Key takeaway:** Choose the task according to the required output: category, number, group, or compact representation.

## 11. What is an artificial neural network?

**Answer**

An artificial neural network (ANN) consists of interconnected computational units called neurons. It learns by adjusting weights and biases.

A neuron computes:

\[
z=\sum_{i=1}^{n}w_ix_i+b,\qquad a=f(z)
\]

A network typically contains:
- **Input layer:** Receives features.
- **Hidden layers:** Transform inputs into useful representations.
- **Output layer:** Produces predictions.
- **Weights and biases:** Learnable parameters.
- **Activation functions:** Add nonlinearity.

Common activation functions:
- **ReLU:** `max(0, x)`; common in hidden layers.
- **Sigmoid:** Maps a value to (0, 1); often used for binary probabilities.
- **Tanh:** Maps a value to (−1, 1).
- **Softmax:** Converts class logits into a probability distribution.

Without nonlinear activations, stacking ordinary linear layers still results in a linear transformation.

**Key takeaway:** Neural networks learn complex functions by optimizing weights and biases across layers.

## 12. Compare CNNs, RNNs, and Transformers.

**Answer**

- **CNN (Convolutional Neural Network):** Learns local spatial patterns using convolutional filters. Used for image classification, detection, and segmentation.
- **RNN (Recurrent Neural Network):** Processes sequences with a hidden state passed from one step to the next. LSTM and GRU variants help with some long-range dependencies.
- **Transformer:** Uses attention to model relationships between tokens or input elements. It supports parallel processing of positions during training and underlies most modern LLMs.

Examples:
- CNN: ResNet, U-Net.
- RNN: sequence and time-series models.
- Transformer: BERT, T5, and GPT-style models.

These categories can overlap: vision Transformers process images, and modern systems can combine convolutional and attention-based components.

**Key takeaway:** CNNs emphasize spatial structure, RNNs process sequences recurrently, and Transformers use attention.

## 13. What is a Transformer and how does self-attention work?

**Answer**

A Transformer is a neural network architecture introduced in the 2017 paper *Attention Is All You Need*. Self-attention allows each token representation to incorporate information from other tokens according to learned relevance.

Each input is projected into:
- **Query (Q):** Represents what information a token is seeking.
- **Key (K):** Represents what a token offers for matching.
- **Value (V):** Represents the content that can be combined.

Scaled dot-product attention is:

\[
\mathrm{Attention}(Q,K,V)=
\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

Steps:
1. Compute query-key similarity scores.
2. Scale by the square root of the key dimension.
3. Apply softmax to obtain attention weights.
4. Use the weights to combine value vectors.

**Multi-head attention** performs several attention operations in parallel, allowing different heads to learn different relationships.

Transformers also use positional information, feedforward layers, residual connections, and normalization. Decoder-only language models use a causal mask to prevent looking at future tokens during next-token prediction.

**Key takeaway:** Self-attention forms context-aware representations by weighting information from other tokens.

## 14. What are tokenization, embeddings, and positional information?

**Answer**

1. **Tokenization:** Divides text into tokens, which may be words, subwords, punctuation, or other units. Tokens are mapped to integer IDs.
2. **Embeddings:** Map token IDs to dense learned vectors.
3. **Positional information:** Represents token order or relative position, using methods such as positional embeddings or RoPE.

For example, an embedding matrix for a vocabulary of 50,000 tokens and embedding dimension 768 has shape:

\[
50{,}000\times768
\]

The initial token embedding is retrieved from this matrix. Subsequent Transformer layers produce contextual representations that depend on surrounding tokens.

**Key takeaway:** Tokenization converts text to IDs; embeddings convert IDs to vectors; positional information helps encode order.

## 15. Compare pretraining, fine-tuning, and inference.

**Answer**

- **Pretraining:** Learns broad representations from a large dataset. An autoregressive LLM commonly learns by predicting the next token.
- **Fine-tuning:** Further trains a pretrained model on targeted examples to improve task performance, behavior, or output format.
- **Inference:** Uses the trained model to generate predictions or responses; standard inference does not update learned parameters.

**Instruction tuning** fine-tunes a model on instruction-and-response examples. Preference optimization and reinforcement-learning-based techniques can further shape model behavior.

**Key takeaway:** Pretraining develops general capabilities, fine-tuning adapts a model, and inference uses it.

## 16. What is an LLM, and how does it generate text?

**Answer**

A Large Language Model (LLM) is trained on large amounts of text or multimodal data to learn patterns and produce useful outputs. Most modern generative LLMs use Transformers.

An autoregressive model predicts the next token conditioned on previous tokens. Its training objective is commonly expressed as:

\[
L=-\sum_{t=1}^{T}\log P(x_t\mid x_{<t})
\]

Generation typically works as follows:
1. Tokenize the prompt.
2. Process tokens through the model.
3. Compute logits for the next token.
4. Convert logits to probabilities.
5. Select a token using greedy decoding or sampling.
6. Append it and repeat until a stopping condition is reached.

**Temperature** controls the sharpness of the probability distribution. **Top-p sampling** restricts sampling to a set of high-probability tokens whose cumulative probability reaches a threshold.

Fluent output is not proof of factual correctness. LLMs can hallucinate.

**Key takeaway:** An autoregressive LLM generates text through repeated next-token prediction.

## 17. What is RAG, and how is it different from fine-tuning?

**Answer**

Retrieval-Augmented Generation (RAG) combines retrieval with generation. It retrieves relevant information from an external knowledge source and provides it to an LLM as context.

Typical pipeline:
1. Collect and clean documents.
2. Split documents into chunks.
3. Embed chunks and store them in a searchable index.
4. Embed or otherwise process the user's query.
5. Retrieve relevant chunks.
6. Optionally rerank results.
7. Add the retrieved context to the prompt.
8. Generate an answer, ideally with source references.

| Aspect | RAG | Fine-tuning |
|---|---|---|
| Main purpose | Supply external knowledge | Adapt model behavior or task performance |
| Updating knowledge | Update the corpus/index | Usually requires additional training |
| Retrieval at inference | Yes | Not inherently |
| Citations | Can be tied to retrieved sources | Not inherently provided |

RAG can fail if retrieval is poor, documents are missing, or the model misinterprets context.

**Key takeaway:** RAG supplies relevant knowledge at inference time; fine-tuning changes model parameters to adapt behavior.

## 18. What is LoRA, and how does parameter-efficient fine-tuning work?

**Answer**

LoRA (Low-Rank Adaptation) adapts a pretrained model by training small additional matrices while keeping the original weights frozen.

The weight update is represented as:

\[
W=W_0+\Delta W,\qquad \Delta W=BA
\]

For a weight matrix of shape \(d\times k\), with rank `r`, the trainable matrices can have shapes:

\[
B\in\mathbb{R}^{d\times r},\qquad
A\in\mathbb{R}^{r\times k}
\]

This uses `r(d + k)` trainable parameters instead of `dk` parameters for the adapted matrix, excluding biases. A common scaling form is:

\[
W=W_0+\frac{\alpha}{r}BA
\]

Important configuration options include:
- `r`: rank.
- `lora_alpha`: scaling factor.
- `lora_dropout`: dropout applied to adapter inputs.
- `target_modules`: layers to adapt.
- `bias`: whether selected biases are trained.
- `task_type`: task-specific configuration.

**QLoRA** combines a quantized frozen base model with trainable LoRA adapters to reduce memory needs.

LoRA is useful for specialized terminology, structured outputs, or domain-specific instructions. It does not automatically guarantee new visual capabilities or correctness.

**Key takeaway:** LoRA reduces trainable parameters by learning a low-rank update to frozen weights.

## 19. What are vector databases and embeddings?

**Answer**

An embedding is a numerical vector that represents learned characteristics of text, images, audio, or other data. A vector database or vector index supports efficient similarity search over these representations.

A semantic search system can retrieve a document with different wording but similar meaning. For example, a query about "reducing overfitting" may retrieve a document about "regularization in deep learning."

Typical workflow:
1. Split documents into chunks.
2. Create embeddings for each chunk.
3. Store vectors with text and metadata.
4. Embed the query in a compatible vector space.
5. Retrieve nearest vectors.
6. Optionally filter or rerank results.

Cosine similarity is:

\[
\cos(\theta)=\frac{A\cdot B}{\|A\|\|B\|}
\]

Exact nearest-neighbor search compares against all vectors. Approximate nearest-neighbor indexes accelerate search but may miss some exact neighbors. Hybrid search combines semantic similarity with keyword search such as BM25.

Examples of tools include FAISS, Qdrant, Milvus, and pgvector.

**Key takeaway:** Embeddings represent data as vectors, and similarity search retrieves semantically related items.

## 20. How do you evaluate and deploy an AI model?

**Answer**

A real AI system needs more than a strong benchmark score. It needs reliable data, appropriate evaluation, robust integration, and monitoring.

A practical workflow:
1. **Define intended use:** Specify inputs, outputs, users, and acceptable failure modes.
2. **Prepare data:** Check quality, labels, duplicates, imbalance, bias, and leakage.
3. **Establish a baseline:** Start with a reproducible model and protocol.
4. **Evaluate:** Choose metrics appropriate to the task.
5. **Test robustness:** Use independent data and relevant edge cases.
6. **Optimize deployment:** Measure latency, memory, throughput, startup time, and hardware use.
7. **Integrate safely:** Validate inputs and outputs, handle errors, and preserve traceability.
8. **Monitor:** Track performance, failures, latency, and distribution shifts.

For medical imaging, evaluation may include sensitivity, specificity, calibration, segmentation metrics, and expert review of generated content. The dataset should reflect the intended clinical use and relevant scanner/acquisition variation.

Offline inference is possible if model weights, dependencies, compatible runtimes, and sufficient compute are available locally. Offline operation does not by itself guarantee security or clinical reliability.

**Key takeaway:** Evaluate the whole system: quality, robustness, latency, memory, security, usability, and safety.

---

# 20 Additional AI/ML Interview Questions

## 21. What is the difference between linear regression and logistic regression?

**Answer**

**Linear regression** predicts a continuous numeric value. It models the target as a linear combination of features:

\[
\hat y=w^\top x+b
\]

It is often trained by minimizing mean squared error or another regression loss.

**Logistic regression** is primarily used for classification. For binary classification, it applies the sigmoid function to a linear score:

\[
p(y=1\mid x)=\sigma(w^\top x+b)
\]

where:

\[
\sigma(z)=\frac{1}{1+e^{-z}}
\]

The resulting value is between 0 and 1 and can be interpreted as a modeled probability under the fitted model. A decision threshold converts it into a class prediction.

Logistic regression is commonly trained using binary cross-entropy.

**Important distinction:** Despite its name, logistic regression is a classification algorithm in its usual application.

**Key takeaway:** Linear regression predicts numeric values; logistic regression models class probabilities.

## 22. Explain decision trees, random forests, and gradient boosting.

**Answer**

- **Decision tree:** Repeatedly splits data based on feature conditions. It is easy to interpret but can overfit.
- **Random forest:** Trains many trees on randomized samples and feature subsets, then combines their predictions. This generally reduces variance compared with a single tree.
- **Gradient boosting:** Builds an ensemble sequentially, with each new learner improving the current ensemble's errors according to an optimization objective.

Common boosting libraries include XGBoost, LightGBM, and CatBoost.

| Algorithm | Main idea | Typical advantage |
|---|---|---|
| Decision tree | Recursive feature splits | Interpretability |
| Random forest | Average many randomized trees | Robustness and reduced variance |
| Gradient boosting | Sequentially improve an ensemble | Strong predictive performance on many tabular tasks |

Boosting can overfit if poorly configured. Learning rate, tree depth, number of estimators, and regularization matter.

**Key takeaway:** Random forests primarily use parallel randomized trees; boosting builds an ensemble sequentially.

## 23. What is a Support Vector Machine (SVM)?

**Answer**

A Support Vector Machine is a supervised algorithm commonly used for classification and, in a modified form, regression.

For classification, it seeks a decision boundary with a large margin between classes. The closest training examples that constrain this boundary are called **support vectors**.

For a linear classifier, the decision boundary is:

\[
w^\top x+b=0
\]

The SVM objective balances a large margin against classification errors, using a regularization parameter commonly written as `C`.

**What is the kernel trick?**

A kernel allows the model to represent nonlinear decision boundaries by calculating similarities that correspond to a higher-dimensional feature space, without explicitly constructing every transformed feature.

Common kernels include linear, polynomial, and radial basis function (RBF).

**Advantages:** Effective on many small-to-medium datasets and can model nonlinear boundaries with kernels.

**Limitations:** Kernel SVMs can become expensive on very large datasets, and scaling and parameter selection are important.

**Key takeaway:** SVMs seek a margin-maximizing decision boundary, with kernels enabling nonlinear classification.

## 24. How does K-means clustering work?

**Answer**

K-means partitions data into `K` clusters by assigning each point to its nearest centroid and updating the centroids.

Algorithm:
1. Choose the number of clusters `K`.
2. Initialize `K` centroids.
3. Assign each point to the closest centroid.
4. Recompute each centroid as the mean of its assigned points.
5. Repeat assignment and update until convergence or a stopping condition.

Its objective is to minimize within-cluster squared distances:

\[
J=\sum_{i=1}^{N}\|x_i-\mu_{c_i}\|^2
\]

where `μ` is a cluster centroid and `cᵢ` is the assigned cluster.

**Limitations:**
- Requires choosing `K`.
- Sensitive to initialization and feature scaling.
- Can be affected by outliers.
- Works best for roughly compact, separated clusters; it struggles with some irregular shapes.

K-means++ initialization, feature scaling, multiple restarts, and methods such as the silhouette score can help.

**Key takeaway:** K-means alternates between nearest-centroid assignment and centroid updates.

## 25. What is Principal Component Analysis (PCA)?

**Answer**

PCA is a linear dimensionality-reduction technique that finds orthogonal directions capturing the greatest variance in numerical data.

A simplified workflow:
1. Center the features, and often scale them when units differ substantially.
2. Compute the covariance matrix or use singular value decomposition (SVD).
3. Find principal directions.
4. Sort directions by explained variance.
5. Project the data onto the selected components.

The first principal component captures the largest possible variance among unit-length linear directions. Subsequent components capture the largest remaining variance while being orthogonal to earlier components.

**Why use PCA?**
- Reduce feature dimensionality.
- Remove some redundancy.
- Visualize high-dimensional data in two or three dimensions.
- Sometimes improve computational efficiency.

**Limitations:** PCA is linear and unsupervised. A low-variance feature can still be highly predictive of a target, so dimensionality reduction should be evaluated against the actual task.

**Key takeaway:** PCA compresses data by projecting it onto directions of high variance.

## 26. What are feature engineering and feature scaling?

**Answer**

**Feature engineering** creates or transforms input variables to make useful patterns easier for a model to learn. Examples include extracting date components, combining related measurements, encoding categories, and deriving domain-specific statistics.

**Feature scaling** adjusts numeric features to comparable ranges.

Common techniques:
- **Standardization:** Subtract the mean and divide by standard deviation.
- **Min-max scaling:** Map values into a selected range, often [0, 1].
- **Robust scaling:** Use statistics such as the median and interquartile range.

Scaling is often important for distance-based algorithms, gradient-based models, and SVMs. Tree-based methods are generally less sensitive to monotonic feature scaling.

**Avoid leakage:** Fit scaling parameters only on the training set, then apply the same transformation to validation, test, and production data.

**Key takeaway:** Feature engineering improves representations; scaling controls numeric ranges and must be fitted without leaking evaluation data.

## 27. How do you handle class imbalance?

**Answer**

Class imbalance occurs when some classes have far fewer examples than others. A model may achieve high accuracy by mostly predicting the majority class.

Possible approaches:
1. Use suitable metrics such as recall, precision, F1, PR-AUC, or class-specific sensitivity.
2. Apply class-weighted losses.
3. Oversample minority examples or undersample majority examples where appropriate.
4. Use valid augmentation to improve minority-class diversity.
5. Tune the decision threshold on validation data.
6. Collect more representative minority-class data where possible.
7. Inspect calibration and performance across relevant subgroups.

Resampling should be performed within the training process, not before splitting all data, because duplicates or synthetic samples could leak into evaluation sets.

In high-stakes applications, threshold choice should reflect the costs of false positives and false negatives, not simply maximize accuracy.

**Key takeaway:** Treat imbalance as both a data problem and an evaluation problem; no single technique works for every dataset.

## 28. What are cross-validation and data leakage?

**Answer**

**Cross-validation** estimates generalization by repeatedly training and evaluating a model on different partitions of the available development data.

In `K`-fold cross-validation:
1. Divide data into `K` folds.
2. Train on `K−1` folds.
3. Evaluate on the remaining fold.
4. Repeat so each fold serves as validation.
5. Aggregate the results.

Cross-validation helps compare models when data is limited. It does not eliminate the need for a final independent test set when one is required.

**Data leakage** occurs when training or model selection uses information that would not legitimately be available at prediction time.

Examples:
- Normalizing all data before splitting.
- Allowing the same patient's scans into both training and test sets.
- Using future information to predict a past event.
- Including a feature that directly encodes the target.

For time-series problems, use time-aware splits. For grouped data, use group-aware splits.

**Key takeaway:** Cross-validation improves estimation; leakage makes evaluation unrealistically optimistic.

## 29. What are ROC-AUC and PR-AUC?

**Answer**

The **ROC curve** plots true positive rate (recall) against false positive rate at different classification thresholds.

\[
\mathrm{TPR}=\frac{TP}{TP+FN}
\]

\[
\mathrm{FPR}=\frac{FP}{FP+TN}
\]

ROC-AUC is the area under the ROC curve. It summarizes ranking discrimination across thresholds.

The **precision-recall curve** plots precision against recall at different thresholds. PR-AUC summarizes the trade-off between finding positives and limiting false positives.

**When should you use each?**
- ROC-AUC is useful for assessing ranking discrimination across thresholds.
- PR-AUC is often especially informative when the positive class is rare, because precision explicitly reflects false positive predictions relative to positive predictions.

Neither metric by itself selects the best operational threshold, guarantees calibration, or proves clinical usefulness.

**Key takeaway:** ROC-AUC measures discrimination through TPR and FPR; PR-AUC focuses on precision and recall.

## 30. What are vanishing and exploding gradients?

**Answer**

During backpropagation, gradients pass through many layers. Repeated multiplication by derivatives can cause gradients to become extremely small or extremely large.

- **Vanishing gradients:** Earlier layers receive tiny gradients and learn very slowly.
- **Exploding gradients:** Gradients grow excessively, leading to unstable parameter updates or numerical problems.

Mitigation strategies include:
- Appropriate initialization.
- Residual or skip connections.
- Normalization layers.
- Suitable activation functions.
- Gradient clipping for exploding gradients.
- Architectures designed to improve gradient flow.

LSTM and GRU networks were designed to address some challenges of learning long-range dependencies in recurrent networks. Transformers also use residual connections and normalization.

**Key takeaway:** Stable gradient flow is essential for effective optimization of deep networks.

## 31. Explain batch size, epochs, and learning rate.

**Answer**

- **Batch size:** Number of examples processed before an optimizer update.
- **Epoch:** One pass through the training dataset.
- **Learning rate:** Controls the scale of parameter updates.

If a training dataset has 10,000 examples and the batch size is 100, one epoch involves approximately 100 batches, assuming all examples are used and no special batching behavior changes this count.

**Trade-offs**
- Larger batches can improve hardware utilization but consume more memory and change gradient noise.
- Smaller batches may use less memory but produce noisier gradient estimates.
- A learning rate that is too high can destabilize training.
- A learning rate that is too low can make training slow or stall.

Learning-rate schedules, warmup, and adaptive optimizers can improve training behavior.

**Key takeaway:** Batch size determines examples per update, an epoch covers the dataset once, and learning rate controls update magnitude.

## 32. Compare batch normalization and layer normalization.

**Answer**

Both normalization methods can improve optimization, but they normalize different dimensions.

**Batch normalization** typically normalizes activations using statistics across the batch for each feature or channel. During inference, it commonly uses running estimates accumulated during training.

**Layer normalization** normalizes across the feature dimensions of an individual example. It does not depend on statistics across the batch in the same way.

| Aspect | Batch normalization | Layer normalization |
|---|---|---|
| Normalization axes | Usually batch and spatial positions, per channel | Features within an example |
| Dependence on batch statistics | Yes, especially during training | No cross-example batch statistics required |
| Common use | CNNs and some feedforward networks | Transformers and sequence models |

The exact axes depend on the implementation and tensor layout.

**Key takeaway:** Batch normalization uses batch-related statistics; layer normalization operates within each example and is well suited to many Transformer architectures.

## 33. What are dropout and weight decay?

**Answer**

Both are techniques commonly used to improve generalization, but they work differently.

**Dropout** randomly sets a proportion of activations to zero during training. This discourages excessive reliance on particular activations. It is generally disabled during ordinary inference, with framework-specific scaling behavior.

**Weight decay** discourages large parameter values. In a simple L2-regularized objective, a penalty proportional to the squared weights is added to the loss. Decoupled weight decay, as used in AdamW, applies the decay separately from the adaptive gradient update.

These techniques can be combined, but neither is always beneficial at every strength or for every architecture.

**Key takeaway:** Dropout regularizes activations during training; weight decay regularizes parameter magnitude.

## 34. Compare generative and discriminative models.

**Answer**

**Discriminative models** learn to predict labels or targets from inputs, often modeling \(P(y\mid x)\) or directly learning a decision function.

Examples:
- Logistic regression for classification.
- A classifier predicting an image category.
- A model estimating whether an email is spam.

**Generative models** learn a data distribution or a process for producing new samples. Depending on the model, they may learn \(P(x)\), \(P(x,y)\), or a conditional distribution such as \(P(y\mid x)\).

Examples:
- Autoregressive language models.
- Diffusion models for image generation.
- Variational autoencoders.
- Generative adversarial networks.

The distinction is about the modeling objective, not simply whether the model uses a neural network. Some generative models can also perform classification or conditional prediction.

**Key takeaway:** Discriminative models focus on predicting targets; generative models learn to represent or sample from data distributions.

## 35. What are diffusion models and GANs?

**Answer**

Both are generative model families, but they learn to generate data differently.

**Diffusion models**
1. During training, progressively add noise to data according to a defined noise schedule.
2. Train a model to predict noise or another parameterization of the reverse process.
3. During generation, iteratively denoise a random sample into a plausible output.

Diffusion models are widely used for image generation and other modalities. Generation can require multiple denoising steps, although accelerated methods exist.

**Generative Adversarial Networks (GANs)** use two networks:
- **Generator:** Produces synthetic samples.
- **Discriminator:** Attempts to distinguish real samples from generated samples.

The two networks train in an adversarial process. GANs can generate sharp images, but training may be unstable and suffer from mode collapse, where the generator produces insufficient diversity.

**Key takeaway:** Diffusion models learn iterative denoising; GANs train a generator against a discriminator.

## 36. What is prompt engineering, and how do you obtain structured outputs?

**Answer**

Prompt engineering is the practice of designing instructions and context to improve a model's responses for a specific task.

Useful techniques include:
- Clearly defining the task and intended audience.
- Providing necessary context and constraints.
- Giving representative examples (few-shot prompting).
- Specifying the required output format.
- Separating trusted instructions from untrusted input.
- Asking the model to identify uncertainty or missing information.
- Testing prompts against representative and adversarial examples.

**Structured output** means requesting data in a predictable schema, such as JSON with required fields and defined types. Where supported, constrained decoding or schema-based output modes can help enforce formatting.

The result should still be parsed and validated by application code. Valid JSON can contain incorrect facts, missing context, or values that violate business rules.

**Key takeaway:** Prompting improves task specification; schema constraints and application validation improve output reliability.

## 37. Why do LLMs hallucinate, and how can you reduce hallucinations?

**Answer**

Hallucination refers to generated content that is unsupported, fabricated, or factually incorrect. An LLM is trained to predict plausible continuations, and its output is not automatically verified against reality.

Potential causes include:
- Missing or outdated knowledge.
- Ambiguous prompts.
- Poor retrieval context.
- Weak representation of rare facts.
- Incorrect reasoning or calculation.
- Excessive confidence in uncertain outputs.

Mitigation techniques:
1. Use RAG with trustworthy, relevant sources.
2. Require evidence or citations for claims that depend on retrieved documents.
3. Allow the system to say that information is unavailable.
4. Use tools or deterministic code for calculations and database lookups.
5. Validate structured outputs.
6. Evaluate factuality and failure cases systematically.
7. Use human review in high-stakes workflows.

Lower temperature may reduce variation but does not guarantee factuality. Fine-tuning can help some behaviors but cannot guarantee that all answers are true.

**Key takeaway:** Reduce hallucination risk through grounding, validation, uncertainty handling, and evaluation—not prompting alone.

## 38. What is model quantization?

**Answer**

Quantization represents model weights, and sometimes activations, using lower-precision numeric formats. For example, a model may use 8-bit or 4-bit representations instead of 16-bit or 32-bit floating-point values.

Potential benefits:
- Lower model memory requirements.
- Reduced memory bandwidth.
- Faster inference on compatible hardware.
- More feasible local or edge deployment.

Potential trade-offs:
- Accuracy or quality degradation.
- Hardware-specific performance differences.
- Additional complexity in calibration and runtime support.

For a rough illustration, storing 7 billion parameters at 16 bits each requires about 14 billion bytes for raw weights, or approximately 13.0 GiB, before accounting for runtime overhead. At 4 bits per parameter, the raw weight storage is approximately one quarter of that amount, ignoring metadata and packing overhead.

Actual memory requirements are higher when the runtime needs caches, temporary buffers, activations, or other resources. For autoregressive LLMs, the key-value (KV) cache can be significant.

**Key takeaway:** Quantization reduces numerical precision to improve memory and potentially speed, while requiring quality and hardware testing.

## 39. What is MLOps, and why is it important?

**Answer**

MLOps combines machine learning, software engineering, data management, and operations practices to develop, deploy, monitor, and maintain ML systems reliably.

Common components:
- Version control for code, data, and model artifacts.
- Reproducible training pipelines.
- Automated testing and CI/CD.
- Model and dataset registries.
- Deployment automation and rollback.
- Monitoring of latency, errors, data drift, and model quality.
- Governance, access controls, and audit trails.

**What is data drift?** A change in the input data distribution over time.

**What is concept drift?** A change in the relationship between inputs and the target or outcome.

For example, a deployed image model may encounter a new scanner protocol that was uncommon during training. Monitoring should detect changes and trigger investigation rather than silently assuming performance remains stable.

**Key takeaway:** MLOps makes ML systems reproducible, observable, maintainable, and safer to update.

## 40. What are the key responsible AI considerations?

**Answer**

Responsible AI includes the processes needed to build systems that are fair, reliable, secure, transparent where appropriate, and accountable.

Important considerations:
- **Fairness:** Evaluate performance across relevant populations and settings.
- **Privacy:** Minimize sensitive data exposure, control access, and follow applicable data-protection requirements.
- **Security:** Protect model endpoints, data, credentials, and model artifacts against misuse.
- **Transparency:** Explain intended use, limitations, and uncertainty.
- **Robustness:** Test edge cases, distribution shifts, and adversarial inputs.
- **Human oversight:** Define who reviews outputs and who is accountable for decisions.
- **Traceability:** Record relevant model versions, inputs, outputs, and provenance while respecting privacy.
- **Governance:** Establish validation, approval, incident response, and update procedures.

In medical AI, a model should be evaluated for its intended clinical use, and the system should make its limitations clear. A radiologist-support tool should not present generated findings as independently verified diagnoses.

Privacy also requires more than offline inference: local access control, encryption where appropriate, secure storage, logging discipline, and deletion policies may still be necessary.

**Key takeaway:** Responsible AI is a lifecycle requirement spanning data collection, development, evaluation, deployment, and monitoring.

---

# Final Revision: Important Formulas

| Concept | Formula |
|---|---|
| Linear model | \(\hat y=w^\top x+b\) |
| Gradient descent | \(\theta_{t+1}=\theta_t-\eta\nabla_\theta L\) |
| Mean squared error | \(\frac{1}{N}\sum_i(y_i-\hat y_i)^2\) |
| Precision | \(TP/(TP+FP)\) |
| Recall | \(TP/(TP+FN)\) |
| F1-score | \(2PR/(P+R)\) |
| Sigmoid | \(1/(1+e^{-z})\) |
| Scaled dot-product attention | \(\mathrm{softmax}(QK^\top/\sqrt{d_k})V\) |
| Cosine similarity | \((A\cdot B)/(\|A\|\|B\|)\) |
| LoRA update | \(W=W_0+(\alpha/r)BA\) |

# Suggested Study Plan

1. **Days 1–2:** Questions 1–10: ML fundamentals and mathematics.
2. **Days 3–4:** Questions 11–15 and 30–33: deep learning and optimization.
3. **Days 5–6:** Questions 16–19 and 34–38: LLMs and generative AI.
4. **Day 7:** Questions 20–29 and 39–40: evaluation, classical algorithms, deployment, and responsible AI.
5. **Final practice:** Explain each answer aloud in 2–3 minutes, draw the relevant architecture, and give a practical example.

## Interview advice

For each question, practice answering in this order:
1. Define the concept in one or two sentences.
2. Explain how it works.
3. Give an example.
4. State its strengths, limitations, and when you would use it.

For applied AI engineering roles, be ready to discuss data leakage, evaluation metrics, latency, memory constraints, deployment architecture, and failure handling—not only model theory.

# 40 Important AI/ML Interview Questions with Detailed Answers

### Q1. What is Bias-Variance Tradeoff?
**Bias** is error from overly simplistic assumptions (underfitting). **Variance** is error from sensitivity to small fluctuations (overfitting). 
Total Error = Bias² + Variance + Irreducible Noise. Goal is to find sweet spot via regularization, more data, or model complexity tuning.

### Q2. Explain Overfitting and how to prevent it.
Model memorizes training data but fails on unseen data. Prevention: L1/L2 regularization, Dropout, Early Stopping, Cross-Validation, Data Augmentation, Reducing model complexity, Getting more data.

### Q3. L1 vs L2 Regularization
**L1 (Lasso):** Adds |w| penalty, produces sparse weights, feature selection, not differentiable at 0.
Loss = MSE + λΣ|w|
**L2 (Ridge):** Adds w² penalty, shrinks weights smoothly, handles multicollinearity.
Loss = MSE + λΣw²

### Q4. What is Cross-Validation?
Technique to assess generalization. k-Fold splits data into k parts, train on k-1, test on 1, rotate. Stratified k-Fold preserves class ratio. Prevents data leakage.

### Q5. Precision, Recall, F1-Score
Precision = TP/(TP+FP) -> When cost of FP is high (spam).
Recall = TP/(TP+FN) -> When cost of FN is high (cancer).
F1 = 2*PR/(P+R) harmonic mean. Use for imbalanced data.

### Q6. ROC vs Precision-Recall Curve
ROC plots TPR vs FPR. Good for balanced data. PR Curve plots Precision vs Recall. Better for highly imbalanced data where negatives dominate.

### Q7. What is Confusion Matrix?
Table of TP, TN, FP, FN to evaluate classification.

### Q8. Why is Accuracy a bad metric for imbalanced data?
If 99% negative, model predicting always negative gets 99% accuracy but 0% recall. Use F1, AUC-PR, Balanced Accuracy.

### Q9. What is Gradient Descent and its variants?
Optimization to minimize loss by moving opposite to gradient.
- **Batch GD:** Uses whole dataset, slow, stable.
- **SGD:** One sample at a time, noisy, fast.
- **Mini-batch GD:** Batch of 32/64/128, best of both.
- **Momentum, Adam, RMSProp** add adaptive learning rates.

### Q10. Explain Adam Optimizer
Adam = Adaptive Moment Estimation. Combines Momentum (m = EMA of grad) and RMSProp (v = EMA of grad²).
m_t = β1*m + (1-β1)*g, v_t = β2*v + (1-β2)*g²
Update: θ = θ - α * m_t_hat / (sqrt(v_t_hat)+eps). Works well with sparse gradients.

### Q11. What is Learning Rate and Learning Rate Schedulers?
Step size for weight update. Too high -> diverge, too low -> slow. Schedulers: Step Decay, Exponential Decay, Cosine Annealing, ReduceLROnPlateau, Warmup.

### Q12. What is the Vanishing/Exploding Gradient Problem?
In deep RNNs, gradients multiplied through many layers -> <1 (vanish) or >1 (explode). Vanishing: weights stop learning. Fix: ReLU, BatchNorm, Residual Connections, LSTM/GRU, Gradient Clipping, Proper Weight Init (He/Xavier).

### Q13. Explain ReLU and its variants
ReLU(x)=max(0,x). Solves vanishing gradient, fast. Problems: Dying ReLU (0 gradient for negative). Variants: Leaky ReLU (0.01x for x<0), ELU, GELU (used in Transformers), Swish.

### Q14. What is Batch Normalization? Why does it work?
Normalizes layer input: (x-mean)/sqrt(var+eps) * gamma + beta. Reduces internal covariate shift, allows higher LR, acts as regularizer. At inference uses running stats.

### Q15. Dropout vs BatchNorm
Dropout randomly zeroes neurons during training to prevent co-adaptation. BatchNorm normalizes activations. Use Dropout after FC layers, BatchNorm after Conv.

### Q16. What is CNN? Explain Convolution.
CNN learns spatial hierarchies. Convolution: filter slides over image computing dot product. Components: Conv Layer (feature maps), Pooling (Max/Avg to reduce size), FC. Concepts: Parameter sharing, Translation invariance.

### Q17. What is Transfer Learning?
Reuse model trained on large dataset (ImageNet) for new task. Freeze early layers (general features), fine-tune last layers. Useful when data is less. Example: ResNet50 -> Medical images.

### Q18. ResNet and Skip Connections
Deep networks degrade. ResNet adds identity: y = F(x) + x. Allows gradient to flow directly, enables 100+ layer training. Solves vanishing gradient.

### Q19. What is Object Detection? R-CNN vs YOLO
Detect bounding box + class. R-CNN: Two-stage, region proposals + classification, accurate but slow. YOLO: Single-stage, divides image into grid, predicts boxes directly, real-time (YOLOv8).

### Q20. What is RNN? Why LSTM/GRU?
RNN processes sequence: h_t = tanh(W_hh*h_{t-1} + W_xh*x_t). Suffers vanishing gradient. LSTM has gates (Input, Forget, Output) and cell state to remember long term. GRU is simplified LSTM with 2 gates.

### Q21. What is Transformer? Explain Self-Attention
Transformer replaces recurrence with attention. Self-Attention:
Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V
Multi-head runs this multiple times in parallel. Positional Encoding adds order info. Used in GPT, BERT, ViT.

### Q22. BERT vs GPT Difference
BERT: Encoder-only, bidirectional, Masked LM, for understanding (classification, NER).
GPT: Decoder-only, causal/autoregressive (left-to-right), for generation.

### Q23. What is Positional Encoding?
Since Transformer has no recurrence, add sin/cos functions of different frequencies to input embeddings to inject position information.

### Q24. What is Fine-tuning vs Prompting vs RAG?
- Fine-tuning: Update weights on domain data.
- Prompting: In-context examples without weight update.
- RAG: Retrieve relevant docs from vector DB and inject into prompt to ground answer and reduce hallucination.

### Q25. What are Embeddings? How to train?
Dense vector representation of discrete objects (words, images). Word2Vec (Skip-gram/CBOW), GloVe. Trained to make similar words close in vector space. Cosine similarity measures similarity.

### Q26. What is K-Means and how to choose K?
Unsupervised clustering. Algorithm: Initialize K centroids, assign points to nearest, update centroids mean, repeat. Choose K via Elbow Method, Silhouette Score. Sensitive to initialization (k-means++).

### Q27. Explain Decision Tree, Random Forest, XGBoost
Decision Tree: Splits by Gini/Entropy.
Random Forest: Bagging of many trees on random data+features, reduces variance.
XGBoost: Boosting, sequential trees correcting previous errors, uses gradient, regularization, handles missing values, best for tabular data.

### Q28. What is Bagging vs Boosting?
Bagging (Bootstrap Aggregating): Parallel trees on random subsets, reduces variance (RF).
Boosting: Sequential, each model focuses on errors of previous, reduces bias (AdaBoost, GBM, XGBoost).

### Q29. Explain SVM and Kernel Trick
SVM finds max margin hyperplane. Kernel Trick maps data to higher dimension without computing explicitly to make non-linearly separable data separable: Linear, RBF (exp(-γ||x-y||²)), Poly.

### Q30. What is PCA?
Unsupervised dimensionality reduction. Finds principal components (eigenvectors of covariance matrix) that maximize variance. Use for visualization, denoising, speed. Keep 95% variance.

### Q31. What is Data Leakage?
When train data contains info about test data (e.g., scaling before split, including future data). Leads to over-optimistic metrics. Fix: All preprocessing inside CV pipeline.

### Q32. What is LLM Hallucination and how to reduce it?
Model generates plausible but false info. Reduce via RAG, lower temperature, better prompting, fine-tuning on factual data, tool use, self-consistency checks.

### Q33. What is Temperature, Top-P, Top-K in LLMs?
- Temperature: Controls randomness. 0=deterministic, >1=creative.
- Top-K: Sample from K most likely tokens.
- Top-P (nucleus): Sample from smallest set whose cumulative prob >= P.

### Q34. Explain Mixture of Experts (MoE)
Instead of one large FFN, have many expert networks. Router selects top 1-2 experts per token. Only subset of parameters active per token -> scales to huge params with same compute (Mixtral 8x7B, Gemma 4 MoE).

### Q35. What is Quantization? Q4_K_M?
Reduces model weight precision from FP16 (16-bit) to INT4/INT8 to save memory/speed. Q4_K_M is 4-bit k-quants with mixed precision, ~4x smaller with ~2% accuracy drop. Essential for mobile/edge LLM (MedGemma GGUF).

### Q36. What is LoRA (Low-Rank Adaptation)?
Fine-tune LLMs efficiently. Instead of updating full W (dxd), learn ΔW = A*B where A(d x r), B(r x d), r << d (r=8/16). Trainable params <1%. Used for MedGemma fine-tuning.

### Q37. Explain Evaluation Metrics for LLM: BLEU, ROUGE, BERTScore
BLEU: Precision-based n-gram overlap for translation.
ROUGE: Recall-based for summarization.
BERTScore: Semantic similarity using BERT embeddings, better than n-gram.

### Q38. What is Reinforcement Learning from Human Feedback (RLHF)?
Steps: 1) SFT model 2) Train Reward Model from human rankings 3) Optimize policy with PPO against reward model. Used to align ChatGPT. Now DPO is simpler alternative.

### Q39. What is Overfitting in LLMs and how to detect?
LLM memorizes training data. Detect via high train loss low, eval loss increases, or model reproduces copyrighted text verbatim. Fix with weight decay, early stopping, data deduplication.

### Q40. System Design: Build a Medical Image Diagnosis App on Phone
- Model: MedGemma 1.5 4B Q4_K_M + mmproj Q4 for offline CT/MRI.
- Stack: Android Native C++ + llama.cpp mtmd for vision, DICOM via dcm4che, slice to PNG.
- Pipeline: Pick 3 middle slices -> run VLM -> structured FINDINGS/IMPRESSION -> doctor verifies.
- Optimization: Download model on first run, 3 images max, 4 threads, quantized.
- Safety: Disclaimer "Assistive only", not FDA device, radiologist must verify.
