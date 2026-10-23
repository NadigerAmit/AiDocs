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