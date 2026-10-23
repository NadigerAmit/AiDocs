# 40 Essential Artificial Intelligence & Machine Learning Questions and Answers

---

## Table of Contents
1. [Foundational Concepts (Q1 – Q10)](#1-foundational-concepts-q1--q10)
2. [Machine Learning Algorithms & Models (Q11 – Q20)](#2-machine-learning-algorithms--models-q11--q20)
3. [Deep Learning & Neural Networks (Q21 – Q30)](#3-deep-learning--neural-networks-q21--q30)
4. [Advanced Topics, NLP & Generative AI (Q31 – Q40)](#4-advanced-topics-nlp--generative-ai-q31--q40)

---

## 1. Foundational Concepts (Q1 – Q10)

### Q1: What is the difference between Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL)?
**Answer:**
- **Artificial Intelligence (AI):** The broad field of computer science focused on building smart systems capable of performing tasks that typically require human intelligence (e.g., reasoning, problem-solving, planning).
- **Machine Learning (ML):** A subset of AI that enables systems to automatically learn and improve from experience without being explicitly programmed, using statistical techniques to find patterns in data.
- **Deep Learning (DL):** A specialized subset of ML based on Artificial Neural Networks with many layers (deep architectures) that automatically extract features from raw data (e.g., images, text, audio).

```
+-------------------------------------------------------+
| Artificial Intelligence (AI)                          |
|   +-------------------------------------------------+ |
|   | Machine Learning (ML)                           | |
|   |   +-------------------------------------------+ | |
|   |   | Deep Learning (DL)                        | | |
|   |   +-------------------------------------------+ | |
|   +-------------------------------------------------+ |
+-------------------------------------------------------+
```

---

### Q2: What are Supervised, Unsupervised, and Reinforcement Learning?
**Answer:**
1. **Supervised Learning:** The model is trained on labeled data $(X, y)$, where input $X$ corresponds to a known output label $y$.
   - *Goal:* Learn a mapping function $f(X) \approx y$.
   - *Examples:* Classification (spam detection) and Regression (house price prediction).
2. **Unsupervised Learning:** The model receives unlabeled data $(X)$ and must uncover hidden patterns, clusters, or structures independently.
   - *Goal:* Discover underlying distributions or groupings in data.
   - *Examples:* Clustering ($K$-Means), Dimensionality Reduction (PCA).
3. **Reinforcement Learning (RL):** An agent learns to make decisions by taking actions in an environment to maximize cumulative reward.
   - *Goal:* Learn an optimal policy $\pi^*$ mapping states to actions.
   - *Examples:* Game playing (AlphaGo), autonomous navigation.

---

### Q3: What is the Bias-Variance Tradeoff?
**Answer:**
The **Bias-Variance Tradeoff** describes the conflict in trying to simultaneously minimize two sources of error in supervised learning models:

* **Bias:** Error introduced by approximating a complex real-world problem with a simplified model.
  * *High Bias:* Leads to **underfitting**—the model misses relevant relations between features and target output.
* **Variance:** Sensitivity of the model to small fluctuations in the training dataset.
  * *High Variance:* Leads to **overfitting**—the model models random noise in the training data rather than the intended output.

$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

Optimal model performance requires balancing both errors to minimize total generalization error on unseen data.

---

### Q4: What are the main differences between Overfitting and Underfitting, and how can they be prevented?
**Answer:**

| Feature | Underfitting | Overfitting |
| :--- | :--- | :--- |
| **Definition** | Model is too simple to capture patterns. | Model learns noise and details of training data too well. |
| **Training Error** | High | Low |
| **Testing Error** | High | High |
| **Remedies** | - Increase model complexity<br>- Add relevant features<br>- Decrease regularization | - Cross-validation<br>- Regularization ($L_1/L_2$)<br>- Early stopping<br>- Pruning / Dropout |

---

### Q5: What is Cross-Validation, and why is $K$-Fold Cross-Validation preferred over a simple train-test split?
**Answer:**
Cross-validation is a statistical technique used to evaluate the generalization performance of a machine learning model on unseen data.

In **$K$-Fold Cross-Validation**:
1. The dataset is randomly split into $K$ equal-sized subsets (folds).
2. The model is trained on $K-1$ folds and tested on the remaining single fold.
3. This step is repeated $K$ times, each time using a different fold as the validation set.
4. The final metric is the average performance across all $K$ iterations.

**Why it is preferred:**
- Mitigates variance associated with a single random train-test split.
- Utilizes the entire dataset for both training and validation.
- Reduces sample bias, providing a more reliable estimate of model performance.

---

### Q6: What are the differences between Generative and Discriminative Models?
**Answer:**
* **Discriminative Models:**
  * Model the conditional probability $P(Y | X)$.
  * Learn the decision boundary between classes.
  * Direct answer to "Is this image a dog or a cat?"
  * *Examples:* Logistic Regression, Support Vector Machines (SVM), Random Forests.
* **Generative Models:**
  * Model the joint probability distribution $P(X, Y)$ or input distribution $P(X)$.
  * Learn how the data was generated to compute probabilities or produce new samples.
  * Answer to "What does a typical cat image look like?"
  * *Examples:* Naive Bayes, Generative Adversarial Networks (GANs), Variational Autoencoders (VAEs).

---

### Q7: Explain Precision, Recall, $F_1$-Score, and ROC-AUC.
**Answer:**
Using Confusion Matrix elements—True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN):

1. **Precision:** Out of all positive predictions, how many were correct?
   $$\text{Precision} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$
2. **Recall (Sensitivity):** Out of all actual positive instances, how many were correctly identified?
   $$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
3. **$F_1$-Score:** The harmonic mean of Precision and Recall, useful when class distribution is imbalanced.
   $$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
4. **ROC-AUC:** Receiver Operating Characteristic curve plots True Positive Rate vs. False Positive Rate across decision thresholds. The Area Under the Curve (AUC) measures overall class separation capacity across all thresholds.

---

### Q8: What is the Curse of Dimensionality and how does it affect ML models?
**Answer:**
The **Curse of Dimensionality** refers to phenomena that arise when organizing and analyzing data in high-dimensional spaces that do not occur in low-dimensional settings.

**Effects on ML:**
- **Data Sparsity:** As dimensions increase, the volume of space increases exponentially, making the available data sparse.
- **Distance Metric Breakdown:** Distance metrics (e.g., Euclidean distance) become less meaningful because distances between points converge to similar values.
- **Overfitting Risk:** Models require exponentially more data to generalize effectively in higher dimensions.

**Mitigation:** Dimensionality reduction techniques like Principal Component Analysis (PCA) or feature selection methods.

---

### Q9: Explain Data Normalization and Standardization. When should you use each?
**Answer:**

* **Min-Max Normalization:** Rescales values into a fixed range, typically $[0, 1]$.
  $$X_{\text{norm}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$$
  * *Use Case:* When features need bounded distributions or when algorithms do not assume any underlying data distribution (e.g., K-Nearest Neighbors, Neural Networks). Sensitive to outliers.

* **Standardization ($Z$-Score Normalization):** Rescales data so it has a mean ($\mu$) of $0$ and standard deviation ($\sigma$) of $1$.
  $$X_{\text{std}} = \frac{X - \mu}{\sigma}$$
  * *Use Case:* When algorithms assume Gaussian-distributed data (e.g., Linear Regression, Logistic Regression, Principal Component Analysis). Handles outliers better than min-max scaling.

---

### Q10: What is Data Leakage in Machine Learning and how can it be prevented?
**Answer:**
**Data Leakage** occurs when information from outside the training dataset is mistakenly used to create the model, leading to overly optimistic evaluation scores during training/testing that fail in production.

**Common Causes:**
- Normalizing or scaling the entire dataset before splitting into train/test sets.
- Including target variable proxy features in predictor variables.
- Time-series data split randomly instead of temporally.

**Prevention Strategies:**
- Always split data into training and testing sets **before** performing preprocessing (imputation, scaling, feature engineering).
- Use pipeline frameworks (e.g., `scikit-learn Pipeline`) to encapsulate transformations.
- Ensure strict temporal separation for time-series data.

---

## 2. Machine Learning Algorithms & Models (Q11 – Q20)

### Q11: How does Linear Regression work, and what are its key assumptions?
**Answer:**
Linear Regression models a linear relationship between a continuous dependent variable $y$ and one or more independent variables $X$:

$$y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon$$

It finds optimal parameters $\beta$ by minimizing the Mean Squared Error (MSE) via Ordinary Least Squares (OLS) or Gradient Descent.

**Key Assumptions:**
1. **Linearity:** Relationship between features and target is linear.
2. **Independence:** Errors/residuals are independent.
3. **Homoscedasticity:** Constant variance of error terms.
4. **Normality:** Residuals are normally distributed.
5. **No Multicollinearity:** Independent variables are not highly correlated with each other.

---

### Q12: How does Logistic Regression work, and why is the Sigmoid Function used?
**Answer:**
Logistic Regression is a binary classification algorithm that predicts the probability $P(Y=1|X)$ of an outcome.

It applies the linear combination of inputs to the **Sigmoid function** $\sigma(z)$:

$$z = w^T X + b$$
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Why the Sigmoid Function is used:**
- Maps any real value $z \in (-\infty, \infty)$ into a valid probability range $(0, 1)$.
- Smooth, differentiable curve, making it suitable for gradient-based optimization algorithms using Binary Cross-Entropy Loss.

---

### Q13: How do Decision Trees work, and how are splits decided (Gini Impurity vs. Entropy)?
**Answer:**
A Decision Tree recursively splits data based on feature thresholds that maximize class purity in child nodes.

**Split Criteria:**
1. **Gini Impurity:** Measures the probability of misclassifying a randomly chosen element.
   $$\text{Gini}(t) = 1 - \sum_{i=1}^{C} p_i^2$$
2. **Entropy (Information Gain):** Measures randomness or uncertainty in a node.
   $$\text{Entropy}(t) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$
   $$\text{Information Gain} = \text{Entropy}(\text{Parent}) - \text{Weighted Entropy}(\text{Children})$$

The feature split that yields the highest Gini reduction or Information Gain is selected at each step.

---

### Q14: Compare Bagging and Boosting ensemble methods.
**Answer:**

| Feature | Bagging (Bootstrap Aggregating) | Boosting |
| :--- | :--- | :--- |
| **Concept** | Builds base estimators independently in parallel. | Builds base estimators sequentially in series. |
| **Goal** | Reduces **Variance**. | Reduces **Bias** (and Variance). |
| **Sampling** | Random bootstrap samples with replacement. | Samples weighted based on misclassification errors of previous models. |
| **Combination** | Majority voting or simple averaging. | Weighted sum based on individual model accuracy. |
| **Examples** | Random Forest, Bagged Decision Trees. | AdaBoost, Gradient Boosting, XGBoost, LightGBM. |

---

### Q15: How does a Random Forest model work?
**Answer:**
Random Forest is an ensemble learning method based on bagging decision trees:

1. **Bootstrap Sampling:** Generates multiple subsets of training data via random sampling with replacement.
2. **Feature Subspacing:** When splitting each node in a decision tree, it selects randomly from a subset of features rather than all available features.
3. **Aggregation:** Combines individual tree outputs—using majority vote for classification or average prediction for regression.

This double randomness (sample level and feature level) decorrelates individual trees, dramatically lowering overall model variance without increasing bias.

---

### Q16: How do Support Vector Machines (SVM) work, and what is the Kernel Trick?
**Answer:**
SVM finds an optimal hyperplane that maximizes the decision margin separating distinct classes in feature space.

**Margin & Support Vectors:**
Points closest to the decision boundary are called *support vectors*. The algorithm maximizes the distance between these points and the decision boundary.

**The Kernel Trick:**
When data is not linearly separable in low-dimensional space, SVM uses kernel functions $K(x_i, x_j)$ (e.g., RBF, Polynomial) to compute dot products in higher-dimensional space *without* explicitly transforming data points into that higher space. This enables efficient non-linear classification.

---

### Q17: How does the $K$-Means Clustering algorithm work, and how do you choose $K$?
**Answer:**
$K$-Means is an iterative unsupervised clustering algorithm:

1. Choose $K$ random initial centroids.
2. **Assignment Step:** Assign each data point to the nearest centroid using Euclidean distance.
3. **Update Step:** Recompute centroids as the mean of all data points assigned to each cluster.
4. Repeat steps 2–3 until centroids stabilize (convergence).

**Choosing $K$:**
- **Elbow Method:** Plot Within-Cluster Sum of Squares (WCSS) vs. $K$ and select the "elbow" point where error reduction slows down.
- **Silhouette Coefficient:** Measures how close each point in one cluster is to points in neighboring clusters (range $[-1, 1]$).

---

### Q18: What is Principal Component Analysis (PCA) and how does it reduce dimensions?
**Answer:**
PCA is an unsupervised linear dimensionality reduction method that transforms correlated features into a smaller set of uncorrelated features called **Principal Components**.

**Steps:**
1. Standardize features to zero mean and unit variance.
2. Compute the Covariance Matrix of features.
3. Compute Eigenvalues and Eigenvectors of the covariance matrix.
4. Sort eigenvectors by descending eigenvalues; eigenvalues represent the proportion of explained variance.
5. Project the original data onto top $k$ eigenvectors to form a reduced feature space.

---

### Q19: Explain Naive Bayes and why it is called "Naive."
**Answer:**
Naive Bayes is a probabilistic classifier based on Bayes' Theorem:

$$P(Y | X_1, X_2, \dots, X_n) = \frac{P(Y) \prod_{i=1}^{n} P(X_i | Y)}{P(X_1, X_2, \dots, X_n)}$$

**Why it is "Naive":**
It assumes that all input features $X_i$ are **conditionally independent** of each other given the class label $Y$. Although this assumption rarely holds true in real-world data, the algorithm performs exceptionally well in practice, particularly for text classification (e.g., spam detection).

---

### Q20: Explain Gradient Boosting and how XGBoost optimizes it.
**Answer:**
**Gradient Boosting** builds an ensemble sequentially. Each new base model predicts the negative gradient (pseudo-residuals) of the loss function of previous models, gradually steering overall prediction error toward zero.

**XGBoost (eXtreme Gradient Boosting) Improvements:**
- **Regularization:** Includes $L_1$ and $L_2$ terms in the objective function to limit leaf complexity and prevent overfitting.
- **Second-Order Gradients:** Uses Taylor expansion to approximate loss function using first (gradient) and second-order derivatives (Hessian).
- **Parallel Computing:** Parallelizes column block sorting for fast split finding.
- **Handling Missing Values:** Automatically learns default split directions for missing data.

---

## 3. Deep Learning & Neural Networks (Q21 – Q30)

### Q21: What is an Artificial Neural Network (ANN) and how does forward propagation work?
**Answer:**
An ANN consists of connected layers of nodes (input, hidden, output) modeled after biological neurons.

**Forward Propagation Steps:**
1. Input vector $X$ is multiplied by layer weights $W$ and added to bias vector $b$:
   $$z = W \cdot X + b$$
2. Non-linear Activation Function $g(\cdot)$ is applied to $z$ to produce activation output $a$:
   $$a = g(z)$$
3. Output $a$ becomes the input to the subsequent layer until reaching the final output layer to compute the prediction $\hat{y}$.

---

### Q22: What is Backpropagation and how does Gradient Descent update network weights?
**Answer:**
**Backpropagation** calculates the gradient of the loss function $L$ with respect to every weight $W$ in the network using the mathematical **Chain Rule**.

1. Compute prediction error: $L(\hat{y}, y)$.
2. Calculate partial derivatives of $L$ backward from output layer to input layer: $\frac{\partial L}{\partial W}$.
3. **Weight Update Rule:** Adjust parameters in the opposite direction of the gradient to minimize total loss:
   $$W_{\text{new}} = W_{\text{old}} - \eta \cdot \frac{\partial L}{\partial W}$$
   *(where $\eta$ is the learning rate).*

---

### Q23: Why are Activation Functions necessary, and how do Sigmoid, Tanh, ReLU, and Leaky ReLU compare?
**Answer:**
Activation functions introduce **non-linearity** into neural networks. Without them, stacked linear transformations $W_2(W_1X + b_1) + b_2$ reduce mathematically to a single linear layer, preventing deep networks from learning complex structures.

| Activation | Formula | Output Range | Key Pros / Cons |
| :--- | :--- | :--- | :--- |
| **Sigmoid** | $\frac{1}{1 + e^{-z}}$ | $(0, 1)$ | Good for binary outputs; suffers from Vanishing Gradient problem. |
| **Tanh** | $\frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $(-1, 1)$ | Zero-centered output; still suffers from Vanishing Gradient. |
| **ReLU** | $\max(0, z)$ | $[0, \infty)$ | Computationally fast; mitigates vanishing gradient; susceptible to "Dying ReLU". |
| **Leaky ReLU** | $\max(\alpha z, z), \alpha \approx 0.01$ | $(-\infty, \infty)$ | Prevents "Dying ReLU" by allowing a small constant gradient for negative values. |

---

### Q24: What are the Vanishing and Exploding Gradient problems, and how can they be solved?
**Answer:**
- **Vanishing Gradient:** During backpropagation in deep networks, gradients shrink exponentially as they propagate backward through layers (especially with Sigmoid/Tanh). Weights in early layers stop updating.
- **Exploding Gradient:** Gradients grow exponentially during backpropagation, leading to unstable weight updates and matrix overflows (`NaN` values).

**Solutions:**
- Use **ReLU / Leaky ReLU** activations instead of Sigmoid/Tanh.
- Proper weight initialization techniques (e.g., **He Initialization**, **Xavier Initialization**).
- **Batch Normalization** to stabilize internal covariate shifts.
- **Gradient Clipping** (specifically for exploding gradients).
- Residual connections (e.g., **ResNets**).

---

### Q25: How do Convolutional Neural Networks (CNNs) work, and what are their primary components?
**Answer:**
CNNs are specialized deep learning architectures designed primarily for grid-structured data like images.

**Primary Components:**
1. **Convolutional Layer:** Applies trainable filters/kernels over input images using element-wise multiplication to create feature maps (detecting edges, textures, patterns).
2. **Activation Layer (ReLU):** Applies non-linear activation functions to feature maps.
3. **Pooling Layer (Max/Average Pooling):** Downsamples spatial dimensions (width $\times$ height) to reduce parameters and computation while retaining dominant spatial features.
4. **Fully Connected (FC) Layer:** Flattens high-level features into a dense vector to generate output predictions (classification scores).

---

### Q26: How do Recurrent Neural Networks (RNNs) work, and what are LSTMs and GRUs?
**Answer:**
**RNNs** process sequential data (time-series, text) by maintaining an internal hidden state $h_t$ that carries information from previous time steps $t-1$:

$$h_t = f(W_h h_{t-1} + W_x x_t + b)$$

Standard RNNs suffer from vanishing gradients over long sequences, causing long-term memory loss.

**Solutions:**
- **LSTM (Long Short-Term Memory):** Introduces a cell state $C_t$ controlled by three gating mechanisms:
  - *Forget Gate:* Decides what information to discard.
  - *Input Gate:* Decides what new information to store.
  - *Output Gate:* Decides what information to output.
- **GRU (Gated Recurrent Unit):** A simplified version of LSTM combining cell state and hidden state, using two gates (*Reset Gate* and *Update Gate*).

---

### Q27: What is Dropout, and how does it prevent Overfitting?
**Answer:**
**Dropout** is a regularization technique used in neural networks during training.

**How it Works:**
- At each training step, individual nodes (and their connection weights) are randomly dropped out (deactivated) with probability $p$.
- Forces network nodes to learn robust representations independently without relying on specific co-adapted neighboring units.
- During inference (testing), all nodes are active, and weights are scaled down by factor $(1-p)$ to ensure matched network activation scales.

---

### Q28: What is Batch Normalization, and why is it useful?
**Answer:**
**Batch Normalization (BatchNorm)** normalizes layer inputs across mini-batches during training:

$$\hat{x}^{(k)} = \frac{x^{(k)} - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y^{(k)} = \gamma \hat{x}^{(k)} + \beta$$

*(where $\gamma, \beta$ are learnable scale and shift parameters).*

**Benefits:**
- Reduces Internal Covariate Shift (changes in distribution of network activations).
- Enables higher learning rates and accelerates training convergence.
- Acts as a mild regularizer, reducing dependency on dropout.

---

### Q29: What are Optimizers in Deep Learning? Compare SGD, Momentum, RMSprop, and Adam.
**Answer:**
Optimizers adjust neural network weights to reduce loss function output.

* **Stochastic Gradient Descent (SGD):** Updates weights based on single or mini-batch samples. High variance in updates causes noisy oscillations.
* **SGD with Momentum:** Adds a fraction of the previous step's update vector to accelerate gradient vectors in relevant directions and damp oscillations.
* **RMSprop:** Scales learning rate by dividing by the moving average of squared gradients, adapting rates per parameter.
* **Adam (Adaptive Moment Estimation):** Combines Momentum (first moment, mean of gradients) and RMSprop (second moment, uncentered variance of gradients). It is currently the standard default optimizer for deep learning.

---

### Q30: What is Transfer Learning, and when should you fine-tune a model vs. use feature extraction?
**Answer:**
**Transfer Learning** reuses a model pre-trained on a massive dataset (e.g., ImageNet, BERT) as a starting point for a task-specific target dataset.

**Strategy Matrix:**

| Dataset Size | Similarity to Source Task | Recommended Strategy |
| :--- | :--- | :--- |
| **Small Data** | High Similarity | **Feature Extraction:** Freeze base pre-trained layers; train only new top classifier head. |
| **Small Data** | Low Similarity | **Train Early/Mid Layers:** Unfreeze select intermediate layers; re-train with low learning rate. |
| **Large Data** | High Similarity | **Fine-Tuning:** Fine-tune entire model end-to-end with low learning rate. |
| **Large Data** | Low Similarity | **Train from Scratch / Full Fine-tune:** Re-initialize weights or fine-tune all layers completely. |

---

## 4. Advanced Topics, NLP & Generative AI (Q31 – Q40)

### Q31: What is the Transformer Architecture, and why did it replace Recurrent Networks in NLP?
**Answer:**
Introduced in *Attention Is All You Need* (Vaswani et al., 2017), the **Transformer** relies entirely on self-attention mechanisms, discarding recurrent sequential processing.

**Why Transformers replaced RNNs:**
- **Parallelization:** Processing whole sentences simultaneously enables GPU training parallelization (unlike sequential processing in RNNs).
- **Long-range Dependencies:** Direct connections between words eliminate gradient attenuation across long token distances.

---

### Q32: How does the Self-Attention Mechanism work mathematically?
**Answer:**
Self-attention maps input representation sequences to Queries ($Q$), Keys ($K$), and Values ($V$) using weight matrices:

$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

**Scaled Dot-Product Attention Equation:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

1. **$Q K^T$:** Computes dot-product similarity between every Query and Key pair.
2. **$\sqrt{d_k}$ Scaling:** Divides by scaling factor ($d_k$ = key dimension) to prevent extremely large dot products from causing vanishing gradients in softmax.
3. **Softmax:** Normalizes weights into probability distribution scores.
4. **Weighted $V$:** Multiplies probabilities by Values to produce contextual output embeddings.

---

### Q33: What are Generative Adversarial Networks (GANs), and how do the Generator and Discriminator interact?
**Answer:**
A **GAN** consists of two neural networks competing in a minimax game:

1. **Generator ($G$):** Tries to create realistic synthetic samples from random noise vector inputs $z$.
2. **Discriminator ($D$):** Tries to distinguish between real data samples ($x$) and fake samples ($G(z)$).

**Minimax Objective Function:**

$$\min_G \max_D V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]$$

Training stops when the generator produces outputs so realistic that the discriminator cannot distinguish synthetic samples from real ones ($D(\text{fake}) = 0.5$).

---

### Q34: What are Autoencoders and Variational Autoencoders (VAEs)?
**Answer:**
* **Autoencoder (AE):**
  - Consists of an **Encoder** mapping input $X$ to low-dimensional latent space $z$, and a **Decoder** reconstructing $\hat{X}$ from $z$.
  - Used for deterministic dimensionality reduction and noise reduction.
* **Variational Autoencoder (VAE):**
  - Generative model mapping inputs to continuous probabilistic latent space parameterized by **Mean ($\mu$)** and **Variance ($\sigma^2$)**.
  - Uses the **Reparameterization Trick** ($z = \mu + \sigma \odot \epsilon, \, \epsilon \sim \mathcal{N}(0, I)$) to allow backpropagation through random sampling.
  - Generates novel output samples by sampling from latent Gaussian distribution space.

---

### Q35: What is Reinforcement Learning from Human Feedback (RLHF) and why is it used in LLMs?
**Answer:**
**RLHF** aligns Large Language Models (LLMs) with human intent, safety, and helpfulness criteria.

**Key Steps:**
1. **Supervised Fine-Tuning (SFT):** Fine-tune base language model on high-quality demonstrative datasets.
2. **Reward Model Training:** Collect human preference ratings for multiple output variations to train a scalar Reward Model.
3. **PPO Optimization:** Optimize SFT base policy model using **Proximal Policy Optimization (PPO)** algorithms, maximizing predicted rewards while using KL-divergence penalties to prevent policy drift.

---

### Q36: What is Retrieval-Augmented Generation (RAG), and what problem does it solve?
**Answer:**
**RAG** combines external document search mechanisms with generative language models.

**The Problem It Solves:**
- Prevents LLM **hallucinations** (confident generation of factual errors).
- Provides access to private, real-time, or domain-specific knowledge bases without requiring computationally expensive model retraining or fine-tuning.

**Workflow:**
1. **Indexing:** Convert documents into embedding vectors and store in a Vector Database.
2. **Retrieval:** Compute semantic similarity between user query and vector database to pull top relevant context chunks.
3. **Generation:** Pass query together with retrieved contextual chunks to an LLM to generate evidence-backed answers.

---

### Q37: What is the difference between Tokenization, Word Embeddings, and Positional Encodings in NLP?
**Answer:**
- **Tokenization:** Breaks raw character text into sub-word unit strings called tokens (e.g., Byte-Pair Encoding / BPE).
- **Word Embeddings:** Maps discrete tokens into continuous vector representations capturing semantic meaning (e.g., Word2Vec, GloVe, BERT embeddings).
- **Positional Encodings:** Adds vector offsets to input embeddings to convey positional sequence order, since attention mechanisms process tokens in parallel without inherent order awareness.

---

### Q38: What are Quantization and Pruning in Model Compression?
**Answer:**
* **Quantization:** Reduces numerical precision of model weights and activations (e.g., converting 32-bit floating-point `FP32` parameters into 8-bit integers `INT8`).
  * *Benefit:* Reduces memory footprint by up to $75\%$ and accelerates inference on hardware accelerators with minimal accuracy loss.
* **Pruning:** Removes non-critical parameters (weights with near-zero value parameters) or entire structural attention heads/channels from neural networks.
  * *Benefit:* Reduces storage footprint and execution operations without significant performance degradation.

---

### Q39: What is the difference between L1 and L2 Regularization?
**Answer:**
Regularization penalizes large weight parameters to reduce overfitting.

* **L1 Regularization (Lasso):** Adds sum of absolute weights to the loss function.
  $$\text{Loss}_{\text{L1}} = \text{Loss} + \lambda \sum |w_i|$$
  * *Property:* Drives redundant parameter weights to exact zero, performing automatic **feature selection**.

* **L2 Regularization (Ridge):** Adds sum of squared weights to the loss function.
  $$\text{Loss}_{\text{L2}} = \text{Loss} + \lambda \sum w_i^2$$
  * *Property:* Distributes penalty smoothly across all parameters, shrinking weights close to zero without zeroing them out entirely.

---

### Q40: What are Hallucinations in Large Language Models, and how can they be mitigated?
**Answer:**
**Hallucination** refers to instances where a generative language model outputs text that is syntactically coherent but factually incorrect or unsupported by source inputs.

**Mitigation Strategies:**
1. **Retrieval-Augmented Generation (RAG):** Ground predictions using context retrieved from trusted external sources.
2. **Prompt Engineering:** Include explicit instructions like *"Answer based only on the provided context. If unsure, state 'I don't know.'"*
3. **Lowering Decoding Temperature:** Reduce generation temperature/top-$p$ parameters to limit stochastic sampling randomness.
4. **Fact-Checking Guardrails:** Implement secondary evaluation chains or verification models to check response validity prior to serving outputs.

##  ========================================================================================

# AI & Machine Learning — 40 Questions with Detailed Answers

---

## Table of Contents

1. [Foundations](#part-1--foundations-18)
2. [Classical Machine Learning Algorithms](#part-2--classical-machine-learning-algorithms-916)
3. [Model Evaluation](#part-3--model-evaluation-1721)
4. [Deep Learning](#part-4--deep-learning-2230)
5. [Modern AI: LLMs, RL & Generative Models](#part-5--modern-ai-llms-rl--generative-models-3136)
6. [Production, MLOps & Ethics](#part-6--production-mlops--ethics-3740)

---

## Part 1 — Foundations (1–8)

### Q1. What is the relationship between AI, Machine Learning, Deep Learning, and Generative AI?

**Answer:**
These are nested subsets:

- **Artificial Intelligence (AI)** — the broadest field: any technique that enables machines to perform tasks that normally require human intelligence (reasoning, perception, language, planning).
- **Machine Learning (ML)** — a subset of AI where systems *learn patterns from data* instead of being explicitly programmed with rules. Example: spam detection learned from labelled emails.
- **Deep Learning (DL)** — a subset of ML using **multi-layer artificial neural networks**. It excels at unstructured data (images, audio, text) because it learns hierarchical feature representations automatically.
- **Generative AI** — a subset of DL (mostly transformers, diffusion models, GANs) that *creates* new content (text, images, code, audio) rather than only classifying or predicting.

**Analogy:** AI is the whole ocean, ML is the fish, DL is a particular species of fish, and GenAI is a specialised organ within that species.

**Key point:** Not all AI is ML (e.g., classic rule-based expert systems), and not all ML is DL (e.g., linear regression, random forests).

---

### Q2. Explain supervised, unsupervised, semi-supervised, self-supervised, and reinforcement learning.

**Answer:**

| Paradigm | Data | Goal | Examples |
|---|---|---|---|
| **Supervised** | Labelled `(X, y)` | Learn mapping `X → y` | Spam detection, price prediction, image classification |
| **Unsupervised** | Unlabelled `X` | Find structure/patterns | Clustering (k-means), PCA, anomaly detection |
| **Semi-supervised** | Small labelled + large unlabelled | Use both to improve accuracy | Medical imaging with few expert labels |
| **Self-supervised** | Unlabelled, labels generated from data itself | Learn general representations | Masked language modelling (BERT), next-token prediction (GPT) |
| **Reinforcement Learning** | No fixed dataset; agent–environment interaction | Maximise cumulative reward | Robotics, game playing (AlphaGo), RLHF for LLMs |

**Why it matters:** Self-supervised learning is the engine behind modern LLMs — it turns raw text into a training signal by hiding part of the input and predicting it. RL is used *after* pretraining to align models with human preferences (RLHF).

---

### Q3. What is the difference between classification and regression?

**Answer:**
Both are **supervised** tasks; the difference is the *type of output*.

- **Classification** predicts a **discrete category** (class label). Output is a probability per class, thresholded to a label. Examples: "fraud / not fraud", "cat / dog / bird". Metrics: accuracy, precision, recall, F1, ROC-AUC. Loss: cross-entropy.
- **Regression** predicts a **continuous numeric value**. Examples: house price, temperature, demand forecast. Metrics: MSE, RMSE, MAE, R², MAPE. Loss: squared error / Huber.

**Nuances:**
- **Ordinal regression** sits in between (ordered categories like star ratings).
- **Multi-label** classification allows multiple simultaneous labels per example (e.g., tags on a photo).
- A regression model can be turned into a classifier by thresholding, and vice versa (e.g., logistic regression is a *classifier* despite its name).

---

### Q4. What are features, labels, and feature engineering? Why does feature engineering matter?

**Answer:**
- **Feature (X)** — an input variable describing an example (age, pixel value, word count).
- **Label (y)** — the target the model should predict.
- **Feature engineering** — the process of creating, transforming, and selecting features to make patterns easier for a model to learn.

**Common techniques:**
- **Encoding** categoricals: one-hot, ordinal, target/mean encoding, embeddings.
- **Scaling**: standardisation (z-score), min-max normalisation — critical for distance-based models (k-NN, SVM) and for neural network convergence.
- **Transformation**: log/sqrt for skewed distributions, binning, polynomial features.
- **Temporal features**: lags, rolling averages, day-of-week, time-since-event.
- **Interaction features**: ratios, products.
- **Text**: TF-IDF, n-grams, embeddings.
- **Selection**: filter methods (correlation, mutual information), wrapper methods (RFE), embedded methods (L1/Lasso).

**Why it matters:** For classical ML on tabular data, good features often matter more than the choice of algorithm. In deep learning the *model* learns features, but domain-driven feature design still helps, especially with limited data.

---

### Q5. Why do we split data into training, validation, and test sets?

**Answer:**
- **Training set (~60–80%)** — used to fit model parameters.
- **Validation set (~10–20%)** — used for *model selection*: tuning hyperparameters, choosing architecture, early stopping. It gives an unbiased estimate during development.
- **Test set (~10–20%)** — touched **once**, at the very end, to estimate real-world generalisation. Using it repeatedly leaks information and inflates your reported performance.

**Key rules:**
- Never tune on the test set.
- Split **before** any preprocessing that learns from data (scaling, imputation, encoding) — fit the transformer on train only, then apply to val/test.
- For **time series**, split chronologically (past → future), never randomly.
- For **grouped data** (patients, users), use `GroupKFold` so the same entity doesn't appear in both train and test.
- For **small datasets**, use **cross-validation** instead of a single validation split.

---

### Q6. Explain the bias–variance tradeoff.

**Answer:**
Total expected error decomposes as:

```
Error = Bias² + Variance + Irreducible Noise
```

- **Bias** — error from overly simplistic assumptions. A high-bias model *underfits*: it misses the true pattern (e.g., fitting a line to a curve). Train and test error are both high and similar.
- **Variance** — error from excessive sensitivity to the training sample. A high-variance model *overfits*: it memorises noise. Train error is low, test error is high.

**The tradeoff:** Increasing model complexity lowers bias but raises variance, and vice versa. The optimal point minimises total error.

**Practical implications:**
- High bias → add features, use a more expressive model, reduce regularisation, train longer.
- High variance → more data, regularisation (L1/L2, dropout), simpler model, feature selection, ensembling/bagging.

**Modern note:** Very large neural networks can exhibit **double descent**, where test error decreases again past the interpolation threshold — so "more capacity always overfits" is not universally true at massive scale.

---

### Q7. What are overfitting and underfitting, and how do you fix them?

**Answer:**
- **Overfitting** — the model fits training noise; train performance ≫ validation performance.
- **Underfitting** — the model is too simple or under-trained; both train and validation performance are poor.

**Fixes for overfitting:**
1. **More data** (or data augmentation).
2. **Regularisation**: L1/L2 weight decay, dropout, label smoothing.
3. **Early stopping** on validation loss.
4. **Simplify** the model (fewer layers/features, pruning trees).
5. **Cross-validation** for more reliable estimates.
6. **Ensembling** (bagging, averaging).
7. **Reduce noise** in features/labels (fix mislabelled data).

**Fixes for underfitting:**
1. **More capacity** (deeper network, more trees, higher-degree features).
2. **Better features** / feature engineering.
3. **Train longer**, tune learning rate, better optimiser.
4. **Reduce over-regularisation.**
5. **Check for bugs** — e.g., a broken data pipeline or all-zero gradients.

**Diagnostic rule of thumb:** Compare train vs validation metrics. Big gap → overfitting. Both bad → underfitting.

---

### Q8. What is the curse of dimensionality and how do you mitigate it?

**Answer:**
As the number of features `d` grows, the volume of the feature space grows exponentially, and data becomes **sparse**. Consequences:

- **Distance concentration** — the ratio of nearest to farthest neighbour distance approaches 1, so "nearest neighbour" becomes meaningless. This breaks k-NN, k-means, and RBF kernels.
- **Exponential data requirement** — you'd need exponentially more samples to densely cover the space.
- **Overfitting risk** increases and computation grows.

**Mitigations:**
- **Dimensionality reduction**: PCA, t-SNE/UMAP (for visualisation), autoencoders, random projection.
- **Feature selection** (Lasso, mutual information, RFE).
- **Regularisation** (L1/L2).
- **Tree-based models** (random forests, gradient boosting), which handle irrelevant features reasonably well.
- **Embeddings** for high-cardinality categoricals.
- **Domain knowledge** to drop redundant or derived features.

---

## Part 2 — Classical Machine Learning Algorithms (9–16)

### Q9. How does linear regression work, and what are its assumptions?

**Answer:**
Linear regression models the target as a linear combination of features:

```
ŷ = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ = wᵀx + b
```

Parameters are found by **minimising the sum of squared errors** (ordinary least squares), which has a closed-form solution `w = (XᵀX)⁻¹Xᵀy` or is solved iteratively with gradient descent. With L2 regularisation it becomes **Ridge**; with L1 it becomes **Lasso** (which also performs feature selection).

**Key assumptions (for valid inference / OLS optimality):**
1. **Linearity** in parameters.
2. **Independence** of errors (violated in time series).
3. **Homoscedasticity** — constant error variance.
4. **Normality** of residuals (needed for p-values/confidence intervals, not for prediction).
5. **No perfect multicollinearity** (check VIF).
6. **No omitted-variable bias** / errors uncorrelated with predictors.

**When assumptions fail:** use transformations, robust regression, or switch to tree-based models. Note that for pure *prediction*, regression still works reasonably even when assumptions are violated — assumptions mainly matter for inference and uncertainty estimates.

---

### Q10. How does logistic regression work?

**Answer:**
Despite the name, logistic regression is a **classification** algorithm. It models the log-odds of a binary outcome as a linear function:

```
log(p/(1-p)) = wᵀx + b
p = σ(wᵀx + b) = 1 / (1 + e^(-(wᵀx+b)))
```

The **sigmoid** squashes any real value into `(0, 1)`, interpretable as a probability. Parameters are learned by **minimising binary cross-entropy (log loss)**, not squared error (which would be non-convex here). There's no closed form — use gradient descent or Newton/IRLS.

**Extensions:**
- **Multinomial (softmax) regression** for >2 classes.
- **One-vs-rest** for multi-class with binary logistic.
- `class_weight` / threshold tuning to handle imbalance.

**Strengths:** fast, interpretable (coefficients = log-odds ratios), well-calibrated probabilities, works well when classes are roughly linearly separable.

**Weaknesses:** assumes a linear decision boundary — needs feature engineering (polynomials, interactions) for non-linear problems.

---

### Q11. How do decision trees decide where to split?

**Answer:**
A decision tree recursively partitions the feature space by choosing, at each node, the split that maximises **purity** of the resulting child nodes. Common criteria:

- **Gini impurity**: `Gini = 1 - Σ pᵢ²` — probability of misclassifying a random sample.
- **Entropy / Information Gain**: `H = -Σ pᵢ log₂ pᵢ`; gain = parent entropy − weighted child entropy.
- **Variance reduction** (MSE) for regression trees.

The algorithm (CART) evaluates candidate thresholds for each feature, picks the best split, and recurses until a stopping condition (max depth, min samples per leaf, impurity threshold) is met.

**Strengths:** interpretable, handles non-linearity and interactions, needs no scaling, robust to outliers, mixed data types.

**Weaknesses:** high variance (small data changes → very different tree), prone to overfitting without pruning, biased toward features with many levels, piecewise-constant outputs.

**Fixes:** pruning, max depth, min samples per leaf, and — most importantly — **ensembling** (random forests, gradient boosting).

---

### Q12. What is bagging, and how does Random Forest improve on a single tree?

**Answer:**
**Bagging (Bootstrap Aggregating):** train many models on **bootstrap samples** (random draws with replacement) of the training data, then aggregate — majority vote for classification, average for regression. This **reduces variance** without increasing bias much.

**Random Forest** = bagging of decision trees **plus feature randomness**: at each split, only a random subset of `m` features (typically `√d` for classification, `d/3` for regression) is considered. This **decorrelates** the trees, which is what makes averaging effective — if all trees were identical, averaging would achieve nothing.

**Why it works:** Averaging `B` models with pairwise correlation `ρ` and variance `σ²` gives variance `ρσ² + (1-ρ)σ²/B`. Reducing `ρ` (via feature subsampling) is what makes random forests much stronger than plain bagged trees.

**Extras:** out-of-bag (OOB) error as a free validation estimate, feature importance, handles high-dimensional data, minimal tuning.

**Limitation:** cannot extrapolate beyond the training range; less interpretable than a single tree.

---

### Q13. How does gradient boosting work (e.g., XGBoost, LightGBM)?

**Answer:**
Boosting builds models **sequentially**, each new weak learner correcting the errors of the ensemble so far. In gradient boosting, this is framed as **gradient descent in function space**:

1. Start with a constant prediction (e.g., mean).
2. Compute the **negative gradient of the loss** (pseudo-residuals) for each example.
3. Fit a shallow tree to predict those residuals.
4. Add it to the ensemble with a **learning rate (shrinkage)** `η`: `Fₘ = Fₘ₋₁ + η·hₘ`.
5. Repeat for `M` iterations.

**Bagging vs boosting:**

| | Bagging (RF) | Boosting (GBM) |
|---|---|---|
| Training | Parallel, independent | Sequential |
| Main effect | Reduces **variance** | Reduces **bias** |
| Trees | Deep | Shallow (stumps) |
| Overfitting risk | Low | Higher — needs careful tuning |

**Modern implementations:** XGBoost (regularised objective, second-order gradients, sparsity-aware splits), LightGBM (histogram binning, leaf-wise growth, GOSS), CatBoost (ordered target statistics for categoricals). They dominate tabular ML benchmarks.

**Key hyperparameters:** `n_estimators`, `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, plus early stopping on a validation set.

---

### Q14. Explain Support Vector Machines and the kernel trick.

**Answer:**
An SVM finds the **maximum-margin hyperplane** separating classes. The margin is the distance to the closest points (**support vectors**). Maximising it improves generalisation.

**Soft margin:** real data isn't linearly separable, so slack variables allow some violations, controlled by `C`. Small `C` → wider margin, more tolerance (more regularisation); large `C` → stricter fit.

**The kernel trick:** SVM optimisation only requires **dot products** `⟨xᵢ, xⱼ⟩`. Replace them with a kernel `K(xᵢ, xⱼ) = ⟨φ(xᵢ), φ(xⱼ)⟩` to operate in a high-dimensional feature space **without ever computing φ explicitly**.

- **Linear**: `xᵀz`
- **Polynomial**: `(γxᵀz + r)^d`
- **RBF/Gaussian**: `exp(-γ‖x - z‖²)` — maps to infinite dimensions; the most common default
- **Sigmoid**: rarely used

**Strengths:** effective in high dimensions, memory-efficient (uses only support vectors), strong theoretical grounding.

**Weaknesses:** doesn't scale to very large datasets (O(n²)–O(n³)), sensitive to `C`/`γ` and feature scaling, no native probability output (Platt scaling needed), struggles with overlapping classes.

---

### Q15. How does k-Nearest Neighbours work, and what are its pitfalls?

**Answer:**
k-NN is a **lazy, instance-based** learner: it stores the training data and, at prediction time, finds the `k` closest training points and takes a majority vote (classification) or average (regression). There is no explicit training phase.

**Key choices:**
- **Distance metric**: Euclidean, Manhattan, cosine (for text/embeddings), Mahalanobis.
- **k**: small `k` → low bias, high variance (noisy, sensitive to outliers); large `k` → smooth but may blur class boundaries. Tune via cross-validation; use odd `k` for binary classification to avoid ties.
- **Weighting**: distance-weighted voting often helps.

**Pitfalls:**
- **Must scale features** — otherwise large-range features dominate the distance.
- **Curse of dimensionality** — distances become uninformative in high dimensions; reduce dimensions first or use cosine similarity on embeddings.
- **Slow inference** — O(n·d) per query; needs approximate nearest neighbour indexes (KD-trees, HNSW, FAISS) at scale.
- **Class imbalance** — majority class dominates the vote; use distance weighting or resampling.

**Strength:** zero assumptions about the data distribution, naturally multi-class, easy to explain ("similar cases looked like this").

---

### Q16. How does k-means clustering work, and how do you choose k?

**Answer:**
k-means partitions `n` points into `k` clusters by minimising **within-cluster sum of squares (inertia)**:

```
J = Σₖ Σ_{x∈Cₖ} ‖x - μₖ‖²
```

**Algorithm (Lloyd's):**
1. Initialise `k` centroids (k-means++ is standard — spreads initial seeds apart).
2. **Assign** each point to the nearest centroid.
3. **Update** each centroid to the mean of its assigned points.
4. Repeat until assignments stop changing or a max iteration count is hit.

**Choosing k:**
- **Elbow method** — plot inertia vs k, look for the "elbow".
- **Silhouette score** — measures cohesion vs separation; pick the k with the highest average silhouette.
- **Gap statistic**, **Davies–Bouldin index**.
- **Domain knowledge** — often the real constraint.

**Limitations:** assumes spherical, similarly-sized clusters; `k` must be specified; sensitive to outliers and scaling; converges to local optima (use `n_init` restarts). Alternatives: DBSCAN/HDBSCAN (arbitrary shapes, finds noise), Gaussian Mixture Models (soft assignments), agglomerative clustering.

---

## Part 3 — Model Evaluation (17–21)

### Q17. Explain the confusion matrix, precision, recall, F1, and specificity.

**Answer:**
For binary classification with "positive" as the class of interest:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | TP | FN |
| **Actual Negative** | FP | TN |

- **Accuracy** = `(TP+TN) / (TP+TN+FP+FN)` — misleading under class imbalance.
- **Precision** = `TP / (TP + FP)` — of everything flagged positive, how many were correct? High precision → few false alarms.
- **Recall / Sensitivity / TPR** = `TP / (TP + FN)` — of all true positives, how many did we catch? High recall → few misses.
- **Specificity / TNR** = `TN / (TN + FP)` — of all true negatives, how many did we correctly clear?
- **F1** = `2·(P·R)/(P+R)` — harmonic mean; balances precision and recall, useful with imbalance.
- **Fβ** — weights recall `β`× more than precision (β>1 favours recall).

**Choosing a metric is a business decision:**
- **Cancer screening** → maximise **recall** (missing a case is catastrophic); accept lower precision.
- **Spam filter** → maximise **precision** (don't delete real email); accept spam slipping through.
- **Fraud detection** → usually recall-first, with a review-capacity constraint on precision.

---

### Q18. When should you use ROC-AUC vs PR-AUC?

**Answer:**
Both summarise performance across all classification thresholds.

- **ROC curve** plots TPR vs FPR. **ROC-AUC** is the probability that a random positive is ranked above a random negative. It is **insensitive to class balance**, which is both its strength (comparable across datasets) and its weakness (it can look excellent even when positive-class performance is poor).
- **PR curve** plots precision vs recall. **PR-AUC / average precision** focuses entirely on the positive class and is **highly sensitive to the prevalence of positives**.

**Guidance:**
- **Balanced classes (≈50/50)** → ROC-AUC is fine and intuitive.
- **Heavy imbalance (fraud, rare disease, click-through)** → prefer **PR-AUC**, because a huge TN count inflates ROC-AUC and hides poor precision.

**Caution:** any threshold-free metric summarises *ranking* quality, not the quality of the specific threshold you'll deploy. Also report **precision/recall at the operating threshold** you'll actually use, plus a **calibration** check (reliability curve, Brier score) if probabilities are consumed downstream.

---

### Q19. What is cross-validation, and which variant should you use?

**Answer:**
Cross-validation (CV) estimates generalisation performance by rotating which fold is held out, so every example is used for both training and validation. It gives a mean and **standard deviation** of performance — far more reliable than a single split, especially with small datasets.

**Variants:**
- **k-fold** (k=5 or 10) — standard default. 5-fold is a good bias/variance/cost compromise; 10-fold is more precise but 2× costlier.
- **Stratified k-fold** — preserves class proportions per fold. **Always use this for classification**, especially imbalanced.
- **Repeated k-fold** — repeat with different random seeds for lower-variance estimates.
- **Leave-One-Out (LOOCV)** — k = n; nearly unbiased but high variance and extremely expensive.
- **GroupKFold** — keeps all records from the same entity (patient, user, session) in one fold. Essential when records are correlated.
- **TimeSeriesSplit** — expanding-window splits, always training on the past and testing on the future. **Never use random CV on time series.**

**Critical rule:** all preprocessing that learns from data (scaling, imputation, target encoding, feature selection, SMOTE) must be **inside** the CV pipeline — otherwise you leak information and overestimate performance.

---

### Q20. Explain MSE, RMSE, MAE, R², and MAPE for regression.

**Answer:**

| Metric | Formula | Properties |
|---|---|---|
| **MSE** | `(1/n)Σ(yᵢ-ŷᵢ)²` | Heavily penalises large errors; same units squared; sensitive to outliers |
| **RMSE** | `√MSE` | Same units as target → most interpretable for magnitude of error |
| **MAE** | `(1/n)Σ\|yᵢ-ŷᵢ\|` | Robust to outliers; equals the median-error interpretation; all errors weighted equally |
| **R²** | `1 - SS_res/SS_tot` | Fraction of variance explained; 1 = perfect, 0 = no better than predicting the mean, can go negative |
| **MAPE** | `(100/n)Σ\|(yᵢ-ŷᵢ)/yᵢ\|` | Scale-free percentage; intuitive for business, but explodes when `y ≈ 0` and is asymmetric (penalises over-prediction more) |

**Guidance:**
- Use **RMSE** when large errors are disproportionately costly.
- Use **MAE** when you want robustness to outliers or a "typical error".
- Report **R²** alongside a raw error metric — R² alone hides the scale.
- Avoid **MAPE** when targets can be zero or near zero; use **sMAPE** or **MASE** instead.
- **Never compare RMSE across different target scales** (e.g., rupees vs dollars) — it isn't scale-free.

---

### Q21. How do you handle class imbalance?

**Answer:**
Imbalance means one class dominates (e.g., 99:1 fraud data). Accuracy becomes useless — predicting all-negative gives 99%.

**Data-level approaches:**
- **Random undersampling** of the majority class — fast, but discards information.
- **Oversampling** — duplicate minority samples (risk of overfitting) or use **SMOTE** to synthesise interpolated minority points.
- **Combined** SMOTE + Tomek links / ENN.
- Apply resampling **only to training folds**, never to validation/test.

**Algorithm-level approaches:**
- **Class weights** (`class_weight='balanced'`) — cheap and often the best first step.
- **Threshold moving** — pick the operating point from the PR curve that matches your business constraint.
- **Cost-sensitive learning** — encode explicit FP/FN costs.
- **Focal loss** for deep learning.

**Evaluation-level:**
- Use **precision, recall, F1, PR-AUC, balanced accuracy, Matthews correlation coefficient** — not accuracy.
- Report per-class metrics and the confusion matrix.

**Best practice order:** start with class weights + PR-AUC evaluation → then try threshold tuning → then resampling if still insufficient. Ensure you preserve **calibration** if probabilities are consumed downstream (resampling distorts them — recalibrate with Platt/isotonic).

---

## Part 4 — Deep Learning (22–30)

### Q22. What is a perceptron, what is an MLP, and why do we need non-linear activation functions?

**Answer:**
A **perceptron** computes a weighted sum plus bias and applies a step function: `y = step(wᵀx + b)`. It can only solve **linearly separable** problems (it famously fails on XOR).

A **Multi-Layer Perceptron (MLP)** stacks layers of units: `h = σ(Wx + b)`, `ŷ = softmax(Vh + c)`. With enough hidden units, an MLP is a **universal function approximator**.

**Why non-linearity is essential:** stacking linear layers gives `W₂(W₁x) = (W₂W₁)x` — still a single linear map. Without a non-linear activation, depth is meaningless. Non-linearities let the network compose piecewise-linear (or smooth) regions and approximate complex functions.

**Common activations:**
- **ReLU** `max(0,x)` — cheap, mitigates vanishing gradients, sparse. Default for hidden layers.
- **Leaky ReLU / GELU / SiLU** — avoid dead neurons; GELU is standard in transformers.
- **Sigmoid** — binary output layer (saturates, avoid in hidden layers).
- **Tanh** — zero-centred, still saturates.
- **Softmax** — multi-class output, produces a probability distribution.

**Modern practice:** ReLU/GELU in hidden layers, sigmoid/softmax at the output, and choose the output activation to match the loss (sigmoid+BCE, softmax+CE, linear+MSE).

---

### Q23. Explain backpropagation.

**Answer:**
Backpropagation is the **reverse-mode automatic differentiation** algorithm used to compute gradients of the loss with respect to every parameter efficiently.

**Forward pass:** compute activations layer by layer, `a⁽ˡ⁾ = σ(W⁽ˡ⁾a⁽ˡ⁻¹⁾ + b⁽ˡ⁾)`, and the final loss `L`.

**Backward pass** — apply the chain rule from the output back to the input:
1. Compute the output error `δ⁽ᴸ⁾ = ∂L/∂z⁽ᴸ⁾`.
2. Propagate: `δ⁽ˡ⁾ = (W⁽ˡ⁺¹⁾ᵀδ⁽ˡ⁺¹⁾) ⊙ σ'(z⁽ˡ⁾)`.
3. Accumulate parameter gradients: `∂L/∂W⁽ˡ⁾ = δ⁽ˡ⁾a⁽ˡ⁻¹⁾ᵀ`, `∂L/∂b⁽ˡ⁾ = δ⁽ˡ⁾`.
4. Update: `W ← W - η·∂L/∂W`.

**Why it's efficient:** the naive approach computes each parameter's gradient separately, costing O(P) forward passes. Backprop reuses intermediate values, so the whole gradient costs roughly **one extra forward pass** — O(1) relative to parameters.

**Key implementation detail:** gradients are **accumulated** across a batch, so you must zero them each iteration (`optimizer.zero_grad()`).

**Enablers:** modern frameworks (PyTorch, TensorFlow) build a **computation graph** dynamically and automate this via autograd.

---

### Q24. What are vanishing and exploding gradients, and how do you fix them?

**Answer:**
In deep networks, gradients are products of many Jacobians across layers. If the typical factor is `< 1`, gradients shrink **exponentially** with depth (**vanishing**) — early layers learn nothing. If `> 1`, they blow up (**exploding**) — loss becomes NaN.

**Causes:** saturating activations (sigmoid/tanh, derivative ≤ 0.25), poor weight initialisation, very deep or recurrent architectures, unstable learning rates.

**Fixes:**
- **Better activations** — ReLU/GELU instead of sigmoid/tanh in hidden layers.
- **Careful initialisation** — Xavier/Glorot (tanh), He/Kaiming (ReLU).
- **Batch/Layer normalisation** — stabilises activation and gradient scales.
- **Residual (skip) connections** — gradient flows through the identity path; the key enabler of 100+ layer networks (ResNet).
- **Gradient clipping** — cap the global norm (standard in RNNs and LLM training).
- **Gated architectures** — LSTM/GRU forget gates mitigate vanishing gradients in sequences.
- **Learning rate warm-up** and lower learning rates.
- **Architectural choices** — transformers use LayerNorm + residuals + scaled attention specifically to keep gradients well-conditioned.

---

### Q25. Compare SGD, momentum, RMSProp, and Adam.

**Answer:**

| Optimiser | Idea | Notes |
|---|---|---|
| **SGD** | `θ ← θ - η·∇L` | Unbiased but noisy; slow in ravines; needs careful LR tuning |
| **Momentum** | Accumulate a velocity: `v ← βv + ∇L; θ ← θ - ηv` | Dampens oscillation, accelerates along consistent directions (β≈0.9) |
| **Nesterov** | Compute the gradient *after* the momentum step | Slightly better convergence, "look-ahead" |
| **AdaGrad** | Per-parameter LR scaled by historical squared gradients | LR decays too aggressively for long training |
| **RMSProp** | AdaGrad with **exponential moving average** of squared gradients | Fixes decay; good for RNNs |
| **Adam** | Momentum + RMSProp + bias correction | Fast, robust, the default; `β₁=0.9, β₂=0.999, ε=1e-8` |
| **AdamW** | Adam with **decoupled** weight decay | Better regularisation; standard for transformers |

**Practical guidance:**
- **Adam/AdamW** — default for transformers, CNNs, and quick prototyping (less LR tuning needed).
- **SGD + momentum** — often generalises slightly better on vision tasks when tuned well; requires more LR/epoch tuning.
- **Schedule** the learning rate (cosine decay, step decay, warm-up) — often matters more than the optimiser choice.
- Adam's adaptive LR can hurt generalisation in some regimes; AdamW + weight decay is the modern standard.

---

### Q26. How do Convolutional Neural Networks work?

**Answer:**
CNNs exploit three properties of images: **local connectivity**, **translation invariance**, and **hierarchical structure**.

**Core operations:**
- **Convolution** — a small learnable filter (e.g., 3×3) slides over the input, computing dot products. It detects local patterns. **Parameter sharing** means the same filter is applied everywhere — drastically fewer parameters than a fully connected layer, and built-in translation equivariance.
- **Padding** — `same` preserves spatial size; `valid` shrinks it.
- **Stride** — how far the filter moves; stride > 1 downsamples.
- **Pooling** — max or average pooling reduces spatial dimensions and adds local invariance.
- **ReLU** after convolutions.
- **Stacking** — early layers learn edges/textures; deeper layers learn parts and objects (**hierarchical features**).
- **Fully connected head** — maps the final feature map to class logits.

**Key hyperparameters:** number of filters, kernel size, stride, padding, pooling.

**Milestones:** LeNet → AlexNet (ReLU + GPU) → VGG (uniform 3×3 stacks) → **ResNet** (skip connections, enabled 100+ layers) → **EfficientNet** (compound scaling). Modern vision increasingly uses **Vision Transformers (ViT)** and hybrids, but CNNs remain strong, efficient, and better with limited data.

---

### Q27. What problem do RNNs solve, and how do LSTM and GRU address their limitations?

**Answer:**
**RNNs** process sequences by maintaining a hidden state: `hₜ = tanh(Wₕhₜ₋₁ + Wₓxₜ + b)`. They share parameters across time, so they handle variable-length input and can model order — unlike MLPs, which treat inputs independently.

**Limitation:** training uses **backpropagation through time**, which multiplies many Jacobians → **vanishing/exploding gradients**, so plain RNNs struggle to learn long-range dependencies (practically ~10–20 steps).

**LSTM** adds a **cell state** `cₜ` and three **gates**:
- **Forget gate** — what to discard from memory.
- **Input gate** — what new information to store.
- **Output gate** — what to expose as the hidden state.

The additive cell-state update gives gradients a relatively unimpeded path through time, allowing learning over hundreds of steps.

**GRU** is a simplified version with two gates (update + reset), fewer parameters, comparable accuracy, and faster training.

**Modern context:** for most sequence tasks, **Transformers** have replaced RNNs because they parallelise across time and capture long-range dependencies via attention. RNNs/LSTMs still appear in streaming/edge scenarios with tight latency or memory budgets.

---

### Q28. Explain attention and the Transformer architecture.

**Answer:**
**Attention** lets each token look at every other token and weight them by relevance:

```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) · V
```

- **Q (query)** — what this token is looking for.
- **K (key)** — what each token offers.
- **V (value)** — the content actually aggregated.
- The `√dₖ` scaling prevents large dot products from saturating the softmax.
- **Multi-head attention** runs several attention heads in parallel with different learned projections, capturing different relationship types (syntax, coreference, position), then concatenates and projects.

**Transformer block:**
1. Multi-head self-attention
2. Add & LayerNorm (residual)
3. Feed-forward network (2 linear layers + GELU, ~4× wider)
4. Add & LayerNorm

Stacked N times. **Positional encodings** (sinusoidal or RoPE) inject order, since attention itself is permutation-invariant.

**Advantages over RNNs:** fully parallel across sequence positions (huge GPU efficiency), constant path length between any two tokens (better long-range learning), and it scales predictably with data and compute.

**Cost:** self-attention is **O(n²)** in sequence length — the main driver of long-context research (sparse attention, FlashAttention, linear attention, sliding windows).

---

### Q29. What regularisation techniques are used in deep learning?

**Answer:**

| Technique | How it works |
|---|---|
| **L2 / weight decay** | Penalises large weights; smooths the function |
| **L1** | Encourages sparsity (rarely used in DL alone) |
| **Dropout** | Randomly zeroes activations during training; prevents co-adaptation; at inference, scale by keep-probability (or use inverted dropout) |
| **Early stopping** | Stop when validation loss stops improving; keeps the best checkpoint |
| **Data augmentation** | Random crops, flips, rotations, colour jitter, mixup/cutmix, text augmentation; the most effective regulariser when data is limited |
| **Batch / Layer normalisation** | Stabilises training; has a mild regularising side effect |
| **Label smoothing** | Softens targets (e.g., 0.9/0.1) to prevent overconfidence |
| **Stochastic depth / DropPath** | Randomly drops residual blocks (used in deep vision transformers) |
| **Noise injection** | Gaussian noise on inputs or embeddings |
| **Weight tying / parameter sharing** | Fewer parameters, e.g., shared embedding and output layers |
| **Pretraining** | Transfer learning is arguably the strongest regulariser of all |

**Practical note:** regularisation strength should be tuned on validation data. Over-regularising causes underfitting (high train *and* validation loss). In the LLM era, dropout is often set to 0 for large models trained on massive data, since data scale itself regularises.

---

### Q30. What is transfer learning, and how does fine-tuning differ from feature extraction?

**Answer:**
**Transfer learning** reuses knowledge from a model trained on a large source task (ImageNet, web-scale text) for a new target task. It works because early/middle representations (edges, textures, syntax, general semantics) are broadly useful, so you need far less target data and compute.

**Two main strategies:**

1. **Feature extraction** — freeze the pretrained backbone, train only a new head (classifier/regressor). Cheapest; best when target data is small and similar to the source domain.
2. **Fine-tuning** — unfreeze some or all layers and continue training with a **small learning rate** (e.g., 1e-5) so pretrained knowledge isn't destroyed. Often done **progressively** (unfreeze from the top down) or with **discriminative learning rates** (lower LR for earlier layers).

**Best practices:**
- Use a lower LR than pretraining; consider LR warm-up.
- Use a small, well-designed head initially, then unfreeze.
- Watch for **catastrophic forgetting** with tiny datasets.
- **Parameter-efficient fine-tuning (PEFT)** — LoRA, adapters, prefix/prompt tuning — updates <1% of parameters and is now standard for LLMs.
- Match preprocessing (normalisation, tokenizer) exactly to the pretrained model.

---

## Part 5 — Modern AI: LLMs, RL & Generative Models (31–36)

### Q31. What is reinforcement learning? Explain MDPs, Q-learning, and policy gradients.

**Answer:**
**RL** trains an **agent** to maximise **cumulative reward** through interaction with an **environment** — no labelled "correct answer" is provided.

**Markov Decision Process (MDP)** formalises this as `(S, A, P, R, γ)`:
- `S` states, `A` actions, `P(s'|s,a)` transition dynamics, `R(s,a)` reward, `γ ∈ [0,1]` discount factor.
- The **policy** `π(a|s)` is what we learn. The **value function** `V^π(s)` is expected discounted return from `s`.

**Value-based — Q-learning:** learn `Q(s,a)`, the expected return of taking action `a` in `s`. Update rule:
`Q(s,a) ← Q(s,a) + α[r + γ·max Q(s',a') - Q(s,a)]`
It is **off-policy** (can learn from old data). **DQN** uses a neural network for Q plus experience replay and a target network to stabilise training.

**Policy-based — policy gradients:** parameterise `π_θ(a|s)` directly and ascend `∇J = E[∇log π_θ(a|s)·A(s,a)]`. Handles continuous actions and stochastic policies. **REINFORCE**, **A2C/A3C**, **PPO** (the workhorse for RLHF) are key variants.

**Actor–Critic** combines both: an actor updates the policy, a critic estimates the value to reduce variance.

**In modern AI:** **RLHF** applies PPO to align LLMs — a reward model trained on human preference pairs serves as the reward signal.

---

### Q32. Compare GANs, VAEs, and diffusion models.

**Answer:**

| | **GAN** | **VAE** | **Diffusion** |
|---|---|---|---|
| **Idea** | Generator vs discriminator in a minimax game | Encoder to latent distribution + decoder; maximise ELBO | Learn to reverse a gradual noising process |
| **Training** | Adversarial; unstable, mode collapse | Stable; clear likelihood bound | Stable but slow (many steps); needs a noise schedule |
| **Sample quality** | Sharp, but can miss modes | Blurry (MSE-like objective) | State-of-the-art quality and diversity |
| **Latent space** | No explicit encoder (unless BiGAN) | Structured, smooth, good for interpolation | Can be inverted; latent diffusion (Stable Diffusion) compresses to a latent space for efficiency |
| **Speed** | Fast (one forward pass) | Fast | Slow (many denoising steps); mitigated by DDIM/distillation |
| **Use cases** | Face synthesis, super-resolution, style transfer | Anomaly detection, representation learning | Image/audio/video generation — dominant today |

**Diffusion in brief:** forward process adds Gaussian noise over T steps; a U-Net (or transformer) learns to predict and remove that noise conditioned on a prompt (via cross-attention). Sampling iteratively denoises random noise into an image.

**Current landscape:** diffusion dominates image generation; GANs remain useful for speed-critical tasks; VAEs serve as components of larger systems (e.g., latent diffusion's encoder).

---

### Q33. How do Large Language Models work?

**Answer:**
LLMs are **decoder-only transformers** trained on enormous text corpora.

**Pipeline:**
1. **Tokenisation** — text → subword tokens (BPE/WordPiece/SentencePiece). Typical vocab: 32k–200k tokens.
2. **Embedding** — tokens → dense vectors; positional information added (RoPE, ALiBi, learned).
3. **Transformer stack** — dozens of layers of causal (masked) self-attention + feed-forward, with residual connections and LayerNorm/RMSNorm.
4. **Pretraining** — **next-token prediction**: minimise cross-entropy of the true next token given all previous tokens. Doing this over trillions of tokens forces the model to internalise grammar, facts, reasoning patterns, and style. This is **self-supervised** — no human labels needed.
5. **Post-training** — **SFT** (supervised fine-tuning on instruction–response pairs), then **alignment** via RLHF or **DPO** to follow instructions, be helpful, and refuse harmful requests.
6. **Inference** — autoregressive decoding with a sampling strategy (greedy, temperature, top-k, top-p/nucleus, beam search).

**Emergent abilities** appear at scale: in-context learning, chain-of-thought reasoning, tool use, code generation.

**Key facts:** parameters range from ~1B to >1T; training is compute-bound (GPU/TPU clusters, weeks to months); context window is limited by O(n²) attention cost.

---

### Q34. What is prompt engineering? Explain zero-shot, few-shot, and chain-of-thought.

**Answer:**
**Prompt engineering** is designing inputs that steer an LLM toward the desired output without changing its weights.

**Core techniques:**

- **Zero-shot** — just the instruction. Works for capable models on common tasks.
- **Few-shot (in-context learning)** — include 2–10 examples of input→output. The model infers the pattern without any gradient updates. Choose examples covering edge cases; order and label balance matter.
- **Chain-of-Thought (CoT)** — instruct the model to "think step by step" before answering. Dramatically improves multi-step arithmetic, logic, and reasoning. Variants: **zero-shot CoT** ("Let's think step by step"), **self-consistency** (sample multiple reasoning paths and majority-vote the answer), **tree-of-thought**.
- **Role/system prompting** — set persona, constraints, and output format.
- **Structured output** — request JSON, XML tags, or a schema; enforce with constrained decoding or function calling.
- **Decomposition** — break a complex task into subtasks.
- **ReAct** — interleave reasoning with tool calls (search, calculator, code interpreter).
- **Prompt chaining** — output of one prompt feeds the next.

**Best practices:** be specific, give format examples, put instructions before long context, use delimiters, and always **evaluate prompts systematically** against a held-out test set — prompt changes are model changes.

---

### Q35. What is Retrieval-Augmented Generation (RAG)?

**Answer:**
**RAG** augments an LLM with an external knowledge base retrieved at query time, so the model can answer using facts it was never trained on and cite sources.

**Pipeline:**
1. **Ingest & chunk** — split documents into passages (respecting structure; overlap chunks for continuity).
2. **Embed** — encode chunks with a text embedding model (e.g., a bi-encoder).
3. **Index** — store vectors in a vector database (FAISS, Pinecone, Weaviate, pgvector) with metadata.
4. **Retrieve** — embed the user query, find top-k nearest chunks (ANN search).
5. **Rerank** (optional but high-value) — a cross-encoder scores query–chunk pairs more accurately.
6. **Generate** — insert retrieved chunks into the prompt as context; instruct the model to answer *only* from the context and cite sources.

**Why RAG over fine-tuning:**
- **Freshness** — update the index without retraining.
- **Attribution** — citations reduce hallucination and build trust.
- **Cost** — far cheaper than retraining.
- **Access control** — filter documents per user at retrieval time.
- **Private data** — never leaves your infrastructure.

**Advanced variants:** hybrid search (BM25 + dense vectors), query rewriting/expansion, multi-hop retrieval, agentic retrieval loops, GraphRAG.

**Failure modes:** bad chunking, embedding/domain mismatch, retrieval misses, context-stuffing, and the model ignoring retrieved context. Evaluate retrieval and generation **separately**.

---

### Q36. Why do LLMs hallucinate, and how do you mitigate it?

**Answer:**
**Hallucination** = fluent, plausible output that is factually wrong or unsupported.

**Causes:**
- **Training objective** rewards plausible next tokens, not truth. The model is a probability model, not a database.
- **No grounding** — facts must be memorised in weights, which is lossy and compresses imperfectly.
- **Knowledge cutoff** — the model doesn't know recent events.
- **Long-tail facts** — rare entities are poorly represented.
- **Overconfidence** — the model rarely says "I don't know" unless trained to.
- **Prompt/context confusion** — ambiguous questions or contradictory context.
- **Decoding** — high temperature increases fabrication.

**Mitigations:**
- **RAG** — ground answers in retrieved, cited sources. The single highest-leverage fix.
- **Prompt constraints** — "answer only from the provided context; if the answer isn't present, say you don't know."
- **Tool use** — calculators, code execution, search APIs for anything computable or verifiable.
- **Structured outputs & schema validation** — reject malformed/unsupported outputs.
- **Self-consistency / self-verification** — sample multiple answers, check agreement; ask the model to critique its own output.
- **Lower temperature** for factual tasks.
- **Post-hoc verification** — NLI-based entailment checks against retrieved evidence.
- **Human review** in high-stakes domains.
- **Evaluation** — build a labelled hallucination test set and track the rate over time.

**Note:** you can reduce hallucination substantially but not eliminate it. Design systems assuming the model will sometimes be wrong.

---

## Part 6 — Production, MLOps & Ethics (37–40)

### Q37. What is MLOps, and what does a full ML lifecycle look like?

**Answer:**
**MLOps** applies DevOps principles to ML: automating and standardising the path from data to a reliable production model.

**Lifecycle stages:**
1. **Problem framing** — define the business objective, success metric, and constraints.
2. **Data collection & labelling** — sourcing, annotation, quality checks, privacy compliance.
3. **EDA & feature engineering** — understanding distributions, building a feature pipeline.
4. **Experimentation** — training candidates, tracking runs (MLflow, W&B), comparing against a baseline.
5. **Evaluation & validation** — offline metrics, slice analysis, fairness checks, business-metric simulation.
6. **Packaging** — containerise with pinned dependencies (Docker); version model + data + code.
7. **Deployment** — batch, real-time API, streaming, or edge.
8. **Monitoring** — latency, throughput, error rates, prediction distributions, **data drift** and **concept drift**, business KPIs.
9. **Retraining & iteration** — triggered on a schedule or by drift alerts; shadow/A-B rollout before full promotion.

**Core MLOps principles:**
- **Version everything** — code, data, model artefacts, hyperparameters, environments.
- **Reproducibility** — same inputs → same model.
- **Automation** — CI/CD for training and deployment (CT).
- **Continuous monitoring** — models silently degrade; treat them as living systems.
- **Feature store** — consistent features between training and serving.

**Common tools:** MLflow, Kubeflow, Airflow, DVC, Feast, Docker/Kubernetes, Evidently, Prometheus/Grafana.

---

### Q38. How do you deploy and monitor ML models in production?

**Answer:**
**Deployment patterns:**

| Pattern | Description | Use when |
|---|---|---|
| **Batch** | Score large datasets on a schedule | Churn scoring, daily forecasts |
| **Real-time API** | Synchronous request/response (REST/gRPC) | Fraud checks, recommendations |
| **Streaming** | Consume events from Kafka, score continuously | IoT, live anomaly detection |
| **Edge/on-device** | Model embedded in the client | Privacy, offline, low latency |

**Serving considerations:** latency budget, throughput, autoscaling, GPU vs CPU cost, model size (quantisation, distillation, ONNX), fallback behaviour on model failure, and **shadow/canary** deployment to limit blast radius.

**Monitoring — the four layers:**
1. **Operational** — uptime, latency p50/p99, error rate, resource usage.
2. **Data quality** — schema violations, nulls, out-of-range values, feature distribution shifts (PSI, KS test).
3. **Model behaviour** — prediction distribution shift, confidence/calibration drift, **data drift** (input distribution changed) vs **concept drift** (input→output relationship changed). Note that **ground-truth labels often arrive late**, so monitor proxies and use delayed-label backfill for accuracy tracking.
4. **Business KPIs** — conversion, revenue, false-positive cost.

**Retraining triggers:** scheduled cadence, drift alarms, performance degradation, or new labelled data. Always validate in shadow mode before promotion, and keep the previous model version ready for instant rollback.

---

### Q39. What is data leakage, and why is it so dangerous?

**Answer:**
**Data leakage** occurs when information from the target or from the test set leaks into training, producing models that look excellent offline but fail in production. It is one of the most common and costly ML mistakes.

**Common types:**
- **Preprocessing leakage** — fitting a scaler, imputer, or encoder on the *full* dataset before splitting. Fix: fit on train only (use `Pipeline` inside CV).
- **Target leakage** — a feature that is a proxy for the label or only available *after* the outcome. E.g., including `treatment_date` when predicting `disease_onset`, or `account_closed` when predicting churn.
- **Temporal leakage** — using future data to predict the past. Fix: chronological splits, point-in-time joins.
- **Group leakage** — the same patient/user appearing in both train and test. Fix: GroupKFold.
- **Duplicate leakage** — near-duplicate records split across train and test. Fix: deduplicate before splitting.
- **Resampling leakage** — SMOTE applied before the split. Fix: resample inside CV folds only.
- **Feature-selection leakage** — selecting features using the full dataset including test. Fix: select inside the pipeline.

**How to detect it:** suspiciously high performance; one feature with dominant importance; sudden drops in production; **permutation tests** (shuffle the target — a legitimate model should score at chance).

**Why it matters:** it produces false confidence, wasted deployment effort, and — in medicine or finance — real harm. Always ask: *"Would this exact feature value be available at prediction time?"*

---

### Q40. What are the key ethical concerns in AI, and how do you address them?

**Answer:**
**1. Bias & fairness**
Models inherit and amplify biases in data (historical hiring, lending, policing). Fairness has competing definitions that cannot all be satisfied simultaneously — demographic parity, equalised odds, individual fairness. **Mitigations:** representative data, fairness audits across subgroups, bias-mitigation techniques (pre/in/post-processing), and **disaggregated evaluation** — never report only aggregate accuracy.

**2. Explainability**
High-stakes decisions need justification. Use inherently interpretable models where possible; otherwise SHAP, LIME, counterfactuals, attention analysis, and partial dependence. Note the tension: explanations can be approximations and may not reflect true causal drivers.

**3. Privacy**
Training data can be memorised and leaked (especially LLMs). Use anonymisation/pseudonymisation, differential privacy, federated learning, and PII scrubbing. Comply with GDPR/DPDP/CCPA — including rights to explanation and deletion.

**4. Accountability & governance**
Who is responsible when a model errs? Establish model cards, data sheets, audit trails, human-in-the-loop for high-stakes decisions, and clear escalation paths.

**5. Misuse & safety**
Deepfakes, disinformation, phishing, autonomous weapons. Requires content provenance (watermarking, C2PA), usage policies, red-teaming, and refusal training.

**6. Environmental & economic cost**
Large-scale training consumes substantial energy; automation displaces labour. Report compute/emissions and consider smaller, distilled models where sufficient.

**7. Transparency**
Disclose when users interact with AI; document limitations and intended use.

**Practical takeaway:** ethics is not a final checklist — it must be embedded throughout the lifecycle: in problem framing, data collection, evaluation (subgroup metrics), deployment (monitoring for disparate impact), and post-deployment review.

---

## Quick Reference — Concept Map

```
AI
└── Machine Learning
    ├── Supervised ── Classification / Regression
    │   ├── Linear & Logistic Regression
    │   ├── Trees → Random Forest (bagging) → XGBoost/LightGBM (boosting)
    │   ├── SVM (kernels)
    │   └── k-NN
    ├── Unsupervised ── Clustering (k-means, DBSCAN), PCA, Anomaly Detection
    ├── Self-supervised ── Pretraining (BERT, GPT)
    ├── Reinforcement Learning ── Q-learning, PPO, RLHF
    └── Deep Learning
        ├── MLP → CNN (vision) → RNN/LSTM (sequences)
        ├── Transformers → LLMs (GPT, Claude, Llama), ViT
        └── Generative → GANs, VAEs, Diffusion
```

---

*End of document — 40 questions, 40 detailed answers.*

# ===================================================================================

# AI & Machine Learning Study Guide
## 40 Interview and Learning Questions with Detailed Answers

---

# 1. What is Artificial Intelligence (AI)?

### Answer:
Artificial Intelligence (AI) is the field of computer science that focuses on creating machines capable of performing tasks that normally require human intelligence.

Examples:
- Image recognition
- Speech recognition
- Language translation
- Medical diagnosis
- Autonomous vehicles

### Key Idea
AI = Making machines think and act intelligently.

---

# 2. What is Machine Learning (ML)?

### Answer:
Machine Learning is a subset of AI where systems learn patterns from data instead of being explicitly programmed.

Traditional Programming:

Input + Rules → Output

Machine Learning:

Input + Output → Learn Rules

### Example
Spam Email Detection:
- Input: Emails
- Output: Spam or Not Spam
- Model learns the rules automatically.

---

# 3. What is Deep Learning?

### Answer:
Deep Learning is a subset of Machine Learning based on multi-layer neural networks.

It automatically learns features from raw data.

### Example
Image Classification:

Traditional ML:
- Manual feature extraction

Deep Learning:
- Learns edges
- Learns shapes
- Learns objects

Automatically.

---

# 4. Difference between AI, ML, and Deep Learning?

### Answer

AI
- Broad field
- Makes machines intelligent

ML
- Subset of AI
- Learns from data

Deep Learning
- Subset of ML
- Uses deep neural networks

### Hierarchy

AI
└── ML
    └── Deep Learning

---

# 5. What is Supervised Learning?

### Answer

Training data contains labels.

Example:

| Input | Output |
|---------|---------|
| X-Ray | Cancer |
| X-Ray | No Cancer |

Model learns relationship between input and output.

### Applications
- Classification
- Regression

---

# 6. What is Unsupervised Learning?

### Answer

No labels exist.

Goal:
Find hidden patterns in data.

### Applications

- Customer Segmentation
- Anomaly Detection
- Clustering

Example:
Grouping patients with similar diseases.

---

# 7. What is Reinforcement Learning?

### Answer

Agent learns by interacting with environment through rewards and penalties.

### Components

- Agent
- Environment
- Action
- Reward
- State

Example:
Chess AI learns by winning games.

---

# 8. What is Classification?

### Answer

Predicting categories.

### Examples

Email:
- Spam
- Not Spam

Medical:
- Tumor
- No Tumor

Common Algorithms:
- Logistic Regression
- SVM
- Random Forest
- Neural Networks

---

# 9. What is Regression?

### Answer

Predicting continuous values.

### Examples

- House Price
- Temperature
- Organ Size

Algorithms:
- Linear Regression
- Random Forest Regression
- XGBoost

---

# 10. What is a Dataset?

### Answer

Collection of examples used to train a model.

Example:

| Age | Weight | Disease |
|-------|---------|----------|
| 45 | 60 | Yes |
| 32 | 70 | No |

---

# 11. What are Features?

### Answer

Input variables used for prediction.

Example:

Predict Heart Disease

Features:
- Age
- Gender
- Blood Pressure
- Cholesterol

---

# 12. What is a Label?

### Answer

Expected target output.

Example

Features:
- CT Scan

Label:
- Cancer Present

---

# 13. What is Training Data?

### Answer

Data used for learning model parameters.

Usually:
- 70%
- 80%
- 90%

of total dataset.

---

# 14. What is Test Data?

### Answer

Used for final evaluation.

Model must never see it during training.

Purpose:
Measure real-world performance.

---

# 15. What is Validation Data?

### Answer

Used for model tuning.

Training → Learn

Validation → Tune

Test → Evaluate

---

# 16. What is Overfitting?

### Answer

Model memorizes training data.

Characteristics:
- Excellent training accuracy
- Poor test accuracy

Example:
Student memorizes answers instead of concepts.

---

# 17. What is Underfitting?

### Answer

Model too simple.

Fails on:
- Training data
- Testing data

Example:
Trying to fit straight line to a complex curve.

---

# 18. What is Generalization?

### Answer

Model performance on unseen data.

Good AI system:
- Learns pattern
- Does not memorize

---

# 19. What is Bias in ML?

### Answer

Error due to wrong assumptions.

Example:
Using very simple model on complex problem.

Produces underfitting.

---

# 20. What is Variance?

### Answer

Sensitivity to training data.

High variance:
- Model changes significantly for small dataset changes.

Produces overfitting.

---

# 21. What is the Bias-Variance Tradeoff?

### Answer

Need balance between:

Low Bias
and
Low Variance

Too much bias:
Underfitting

Too much variance:
Overfitting

---

# 22. What is Feature Engineering?

### Answer

Creating useful inputs from raw data.

Example:

Birth Date
→ Age

Height + Weight
→ BMI

---

# 23. What is Data Preprocessing?

### Answer

Preparing data before training.

Includes:
- Cleaning
- Scaling
- Encoding
- Handling missing values

---

# 24. What is Normalization?

### Answer

Scaling values between 0 and 1.

Example:

Original:
20, 50, 100

Normalized:
0.2, 0.5, 1.0

---

# 25. What is Standardization?

### Answer

Transforms data using:

z = (x − mean) / std

Produces:
- Mean = 0
- Standard Deviation = 1

---

# 26. What is Cross Validation?

### Answer

Technique for reliable evaluation.

Most common:
K-Fold Cross Validation

Dataset split multiple times.

Results averaged.

---

# 27. What is Confusion Matrix?

### Answer

Classification evaluation table.

| | Pred Yes | Pred No |
|-----|--------|--------|
| Actual Yes | TP | FN |
| Actual No | FP | TN |

---

# 28. What is Accuracy?

### Answer

Accuracy = Correct Predictions / Total Predictions

Example

95 Correct
100 Total

Accuracy = 95%

---

# 29. What is Precision?

### Answer

Precision = TP / (TP + FP)

Measures:
How many positive predictions were correct.

Important in:
Cancer Detection

---

# 30. What is Recall?

### Answer

Recall = TP / (TP + FN)

Measures:
How many actual positives were found.

Important in medical diagnosis.

---

# 31. What is F1 Score?

### Answer

Combines:
- Precision
- Recall

Formula:

F1 = 2 × Precision × Recall / (Precision + Recall)

---

# 32. What is ROC Curve?

### Answer

Shows model performance at different thresholds.

Higher curve:
Better classifier.

---

# 33. What is AUC?

### Answer

Area Under ROC Curve.

Values:

0.5 = Random

0.8 = Good

0.9+ = Excellent

---

# 34. What is a Neural Network?

### Answer

Brain-inspired model composed of neurons.

Structure:

Input Layer
↓
Hidden Layers
↓
Output Layer

Learns complex patterns.

---

# 35. What is an Activation Function?

### Answer

Adds non-linearity.

Popular Functions:

- ReLU
- Sigmoid
- Tanh
- GELU

Without activation:
Neural network becomes linear.

---

# 36. What is Backpropagation?

### Answer

Algorithm used to train neural networks.

Steps:
1. Forward Pass
2. Compute Error
3. Propagate Error Backward
4. Update Weights

---

# 37. What is Gradient Descent?

### Answer

Optimization algorithm.

Goal:
Minimize loss function.

Basic Rule:

New Weight =
Old Weight - Learning Rate × Gradient

---

# 38. What is a Transformer?

### Answer

Modern neural network architecture.

Introduced in:

"Attention is All You Need"

Features:
- Parallel processing
- Self-Attention
- Scalable training

Used in:
- GPT
- Gemini
- Claude
- Llama
- MedGemma

---

# 39. What is Self-Attention?

### Answer

Allows model to focus on important parts of input.

Example:

"The doctor examined the patient because he was sick."

Self-attention helps determine:
"He" refers to patient.

---

# 40. What is a Large Language Model (LLM)?

### Answer

A Transformer trained on massive text data.

Capabilities:
- Question Answering
- Summarization
- Translation
- Coding
- Reasoning

Examples:
- GPT-5
- Gemini
- Claude
- Llama
- MedGemma

### Limitations
- Hallucinations
- Bias
- Knowledge Cutoff
- High Compute Cost

---

# Quick Revision Cheat Sheet

AI
→ Machines act intelligently

ML
→ Learns from data

Deep Learning
→ Neural Networks

Supervised Learning
→ Labeled Data

Unsupervised Learning
→ No Labels

Reinforcement Learning
→ Rewards & Penalties

Overfitting
→ Memorization

Underfitting
→ Too Simple

Precision
→ Positive Prediction Quality

Recall
→ Positive Detection Rate

F1 Score
→ Precision + Recall Balance

CNN
→ Images

RNN
→ Sequential Data

Transformer
→ Attention-Based Architecture

LLM
→ Large Transformer Language Model

---

# Recommended Learning Path

1. Python
2. NumPy
3. Pandas
4. Matplotlib
5. Statistics
6. Linear Algebra
7. Machine Learning
8. Scikit-Learn
9. Deep Learning
10. PyTorch
11. CNN
12. RNN
13. Transformers
14. LLMs
15. RAG
16. Agents
17. Multimodal AI
18. Medical AI

---

# Final Advice

Do not rush directly into LLMs.

Build a strong foundation:

Math
→ Statistics
→ Machine Learning
→ Neural Networks
→ Deep Learning
→ Transformers
→ LLMs
→ AI Agents

With this approach, you will understand not only how to use AI models such as MedGemma, Llama, GPT, and Gemma, but also how they work internally and how to build your own AI systems from scratch.
# -------------------------------------------------------------------------------------
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
