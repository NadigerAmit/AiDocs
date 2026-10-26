# Technical Architecture Guide: Embedded Intent Classification for TV/STB via MCP

This document establishes the technical engineering blueprint for deploying a highly optimized, resource-constrained intent classification system on Smart TVs and Set-Top Boxes (STBs). The system architecture is written natively in **Rust** to process user voice or text inputs into a predefined set of semantic intents. These classification outputs map directly as JSON-RPC payloads to call tools exposed by local or remote **Model Context Protocol (MCP)** server instances.

---

## 1. Target Intent Actions
The classifier routes incoming query strings across seven discrete functional targets:
*   `tune_to`: Actively modify hardware tuner parameters to view a station (e.g., *"switch to ESPN"*, *"open Das Erste"*).
*   `channel_lookup`: Query station metadata directories for station availability (e.g., *"is CNN available?"*, *"where is BBC"*).
*   `search_programs`: Interrogate the Electronic Program Guide (EPG) for specific genres, actors, or themes (e.g., *"look for movies"*, *"find shows about cricket"*).
*   `list_channels`: Display all accessible services or network streams (e.g., *"show available services"*, *"browse channels please"*).
*   `current_programs`: Return metadata regarding active real-time airings (e.g., *"what is on now"*, *"show epg for me"*, *"pf right now"*).
*   `favorites`: Expose user-curated list parameters (e.g., *"tell me list favorites right now"*, *"kindly show favourites"*).
*   `unsupported`: Gracefully capture inputs outside the TV system's operational control capability (e.g., *"please book a taxi"*, *"write a story"*).

---

## 2. High-Level Architecture Comparison Matrix

| Technical Metric | Option 1: TF-IDF + Linear Classifier | Option 2: ONNX Runtime + Embeddings | The Hybrid Pipeline (Recommended) |
| :--- | :--- | :--- | :--- |
| **Needs Training?** | 🔄 **Yes.** Requires pre-compiling your custom JSON dataset to extract feature weights. | ❌ **No.** Uses pre-trained semantic weight configurations out-of-the-box. | 🔄 **Yes (For Option 1 layer only).** Option 2 remains an un-trained semantic baseline. |
| **Active Memory (RAM)** | ⚡ **Ultra Lean (< 2 MB).** Zero risk of causing memory strain on low-tier SoCs. | 💾 **~25 MB - 30 MB.** Small for deep learning, but higher baseline memory usage. | 🧠 **Dynamic (< 2 MB Idle).** Lazily initializes Option 2 structures *only* during active fallbacks. |
| **CPU Processing Latency** | 🚀 **Sub-millisecond (< 500 µs).** Fast matrix calculations on simple arrays. | 🐢 **2–8 Milliseconds.** Runs multi-layer neural network text token matching loops. | 🏎️ **Fast-Tracked (< 500 µs average).** Only encounters neural network latency during fallback exceptions. |
| **Synonym Handling** | ❌ **Weak.** Relies on explicit word patterns mapped during text extraction compilation. | 🎯 **Exceptional.** Understands vector distances between abstract concepts natively. | 🛡️ **Robust.** Captures standard operations instantly while processing loose language safely. |
| **Binary/Asset Size On-Disk** | 📦 **Minimal (< 1 MB).** Vocabulary weights compile right into your application binary file. | 📦 **~35 MB total.** Requires storing an external `.onnx` weight model and `tokenizer.json` on disk. | 📦 **~35 MB total.** Bundles both engines, keeping the heavy ONNX session inactive until triggered. |

> **Measurement note:** Memory, latency, model size, and synonym-quality entries above are illustrative estimates from an assumed implementation, not portable guarantees or benchmark results. Measure cold start, warm p50/p95 latency, peak resident memory, package size, and energy on each supported device with representative inputs. A confidence threshold must be selected and validated on held-out, deployment-like data; a threshold alone cannot guarantee out-of-domain rejection.

---

## 3. Handling Out-of-Dataset Queries & Future Scalability

### 3.1 Behavior with Unseen Text Queries
*   **In Option 1:** Text inputs are mapped directly to a sparse array of structural word frequencies. If a user inputs text entirely unrelated to your system (e.g., *"How do I bake a chocolate cake?"*), the words yield no matches in the active vocabulary. This returns equal, uniform probability distributions across all intents. Setting a strict confidence threshold (e.g., `< 0.60`) guarantees the system rejects these queries rather than predicting a random channel function.
*   **In Option 2:** The abstract phrase is transformed into a continuous numerical embedding vector. It is compared mathematically against target reference points via **Cosine Similarity**. If the similarity score across all active operations falls below an explicit threshold boundary (e.g., `< 0.35`), it flags the query as `unsupported`.

### 3.2 Extending Operations for Future Upgrades
When adding new operations down the line (e.g., `increase_volume`, `decrease_volume`, `watch_epg`):
1.  **Updating Option 1:** Add the new sample commands directly into your developer JSON dataset file (e.g., `{"instruction": "make it louder", "output": "increase_volume"}`). Re-run the standalone compilation script on your development PC to generate a fresh model coefficient slice, then ship it via a routine Over-The-Air (OTA) firmware patch.
2.  **Updating Option 2:** **Do not modify or replace the heavy `.onnx` file.** The baseline model already natively understands the semantic concept of words like "volume" or "audio." You only need to calculate the target embedding vector for your new intent string once on your PC, then append that 384-dimensional array chunk into the tiny local configuration file on the TV.

---

## 4. Option 1: TF-IDF + Linear Classifier Blueprint

### How to Achieve
This variant builds a local word index mapping feature from token frequencies found within historical JSON instructions, passing values down into a pure Rust Support Vector Machine (Linear SVC) or Logistic Regression classifier matrix.

### Configuration (`Cargo.toml`)
```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
smartcore = { version = "0.3", features = ["linalg-ndarray"] }
ndarray = "0.15"
```

### Pure Rust Implementation
```rust
use serde::Deserialize;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct TrainingItem {
    instruction: String,
    output: String,
}

fn main() {
    // 1. Load the dataset from your provided JSON file schema
    let raw_data = r#"
    [
        {"instruction": "please book a taxi quickly", "output": "unsupported"},
        {"instruction": "could you show epg for me", "output": "current_programs"},
        {"instruction": "open Das Erste", "output": "tune_to"},
        {"instruction": "locate CNN", "output": "channel_lookup"},
        {"instruction": "tell me list favorites right now", "output": "favorites"},
        {"instruction": "look for movies", "output": "search_programs"},
        {"instruction": "show available services quickly", "output": "list_channels"}
    ]
    "#; 

    let dataset: Vec<TrainingItem> = serde_json::from_str(raw_data).unwrap();

    // 2. Build Vocabulary Dictionary (Bag of Words Features)
    let mut vocab: HashMap<String, usize> = HashMap::new();
    let mut intent_map: HashMap<String, usize> = HashMap::new();
    let mut reverse_intent_map: HashMap<usize, String> = HashMap::new();
    
    let mut word_idx = 0;
    let mut intent_idx = 0;

    for item in &dataset {
        for word in item.instruction.to_lowercase().split_whitespace() {
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric()).to_string();
            if !vocab.contains_key(&clean_word) && !clean_word.is_empty() {
                vocab.insert(clean_word, word_idx);
                word_idx += 1;
            }
        }
        
        if !intent_map.contains_key(&item.output) {
            intent_map.insert(item.output.clone(), intent_idx);
            reverse_intent_map.insert(intent_idx, item.output.clone());
            intent_idx += 1;
        }
    }

    // 3. Assemble Feature Matrices
    let num_samples = dataset.len();
    let num_features = vocab.len();
    let mut x_data = vec![0.0f32; num_samples * num_features];
    let mut y_data = vec![0.0f32; num_samples];

    for (row, item) in dataset.iter().enumerate() {
        for word in item.instruction.to_lowercase().split_whitespace() {
            let clean_word = word.trim_matches(|c: char| !c.is_alphanumeric());
            if let Some(&col) = vocab.get(clean_word) {
                x_data[row * num_features + col] += 1.0;
            }
        }
        y_data[row] = *intent_map.get(&item.output).unwrap() as f32;
    }

    let x = DenseMatrix::from_array(num_samples, num_features, &x_data);
    let y = y_data;

    // 4. Fit the native classifier matrix
    let model = LogisticRegression::fit(&x, &y, Default::default()).unwrap();
    println("Option 1 compilation complete. Dict footprint sizes: {} features.", num_features);

    // 5. Embedded Query Runtime Test
    let query = "open Das Erste please";
    let mut test_features = vec![0.0f32; num_features];
    for word in query.to_lowercase().split_whitespace() {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
        if let Some(&col) = vocab.get(clean) {
            test_features[col] += 1.0;
        }
    }
    
    let test_matrix = DenseMatrix::from_array(1, num_features, &test_features);
    let prediction = model.predict(&test_matrix).unwrap();
    let final_intent = reverse_intent_map.get(&(prediction as usize)).unwrap();
    
    println!("Query matched: '{}' -> Target Intent Token: '{}'", query, final_intent);
}
```

---

## 5. Option 2: The Semantic Variant (ONNX Runtime + Embeddings)

### How to Achieve
This setup maps sentences into a vector space using a pre-trained language model, resolving synonyms dynamically. We use Microsoft's Open Neural Network Exchange (ONNX) format for cross-platform execution directly on the STB device.

### 5.1 Model Selection & Download from Hugging Face
We recommend using **`all-MiniLM-L6-v2`** due to its micro-sized dimensions (384 output vector space) and optimal balance of speed and footprint.

*   **Official Model Repository:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
*   **Alternative Option (Higher Context Baseline):** [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)

### 5.2 How to Convert to ONNX & Quantize (INT8 Compression)
To compress the model weights from 130MB down to roughly ~30MB for low-tier hardware, perform this conversion step on a development workstation using Python's `optimum` toolkit:

```bash
# Install toolchains
pip install optimum[onnxruntime]

# Download, convert, and export as optimized quantized binary structures
optimum-cli export onnx \
  --model sentence-transformers/all-MiniLM-L6-v2   --optimize O2 \
  --quantize INT8 \
  ./tv_onnx_assets/
```
*This produces two highly critical files needed for the device local filesystem: `model_quantized.onnx` and `tokenizer.json`.*

### 5.3 Configuration (`Cargo.toml`)
```toml
[dependencies]
ort = { version = "2.0", features = ["ndarray"] }
tokenizers = "0.15" 
ndarray = "0.15"
```

### 5.4 Implementation Example using `ort`
```rust
use ndarray::{Array1, Array2, Axis};
use ort::{inputs, Session};
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize local assets from device flash storage safely
    let tokenizer = Tokenizer::from_file("tv_onnx_assets/tokenizer.json")?;
    let session = Session::builder()?.commit_from_file("tv_onnx_assets/model_quantized.onnx")?;

    let query = "could you switch the broadcast to the CNN stream?";

    // 2. Tokenize textual sequences
    let encodings = tokenizer.encode(query, true)?;
    let input_ids: Vec<i64> = encodings.get_ids().iter().map(|&x| x as i64).collect();
    let attention_mask: Vec<i64> = encodings.get_attention_mask().iter().map(|&x| x as i64).collect();
    let seq_len = input_ids.len();

    let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids)?;
    let attention_mask_array = Array2::from_shape_vec((1, seq_len), attention_mask)?;

    // 3. Compute Vector Embeddings via ONNX Engine
    let outputs = session.run(inputs![
        "input_ids" => input_ids_array,
        "attention_mask" => attention_mask_array,
    ]?)?;

    let token_embeddings = outputs["token_embeddings"].try_extract_tensor::<f32>()?;
    
    // Mean pooling aggregates multi-token outputs into a distinct 384 float footprint array
    let output_matrix = token_embeddings.to_owned().into_dyn();
    let mean_pooled_vector = output_matrix.mean_axis(Axis(1)).unwrap().into_shape(384)?;

    println!("Vector generation complete. Output Dimensions: {:?}", mean_pooled_vector.shape());
    Ok(())
}
```

---

## 6. The Production Hybrid Pipeline (Recommended Engine Design)

### 6.1 Process Workflow Mechanics
The Hybrid Pipeline functions as a tiered gatekeeper system. It passes inbound strings to Option 1 first to run lightweight keyword routing. If Option 1 returns a low confidence score, the system dynamically spins up Option 2's ONNX graph as a fallback.

```
[ Inbound User Command String ]
               │
               ▼
   ┌───────────────────────┐
   │  Execute Option 1     │ 🧠 RAM Footprint: < 2 MB
   │ (smartcore tf-idf)    │ 🏎️ Latency Processing: Sub-Millisecond
   └───────────┬───────────┘
               │
      Is Confidence Score >= 0.65?
               │
       ┌───────┴───────┐
      YES              NO
       │               │
       ▼               ▼
┌──────────────┐ ┌───────────────────────────────┐
│ Route Direct │ │  Lazy-Load & Wake Option 2    │ 🧠 RAM Footprint: ~25 MB (Allocated only on-demand)
│  MCP Tool    │ │     (ort ONNX Embedding)      │ 🐢 Latency Processing: 2 - 8 Milliseconds
└──────────────┘ └─────────────┬─────────────────┘
                               │
                    Is Cosine Match >= 0.35?
                               │
                       ┌───────┴───────┐
                      YES              NO
                       │               │
                       ▼               ▼
                ┌──────────────┐ ┌───────────────┐
                │ Route Direct │ │ Return Model  │
                │  MCP Tool    │ │ "unsupported" │
                └──────────────┘ └───────────────┘
```

### 6.2 Full Pipeline Implementation Code Blueprint
```rust
use ndarray::{Array1, Array2, Axis};
use ort::{inputs, Session};
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;
use tokenizers::Tokenizer;
use std::collections::HashMap;

#[derive(Debug, PartialEq)]
enum PipelineOutput {
    DirectIntent(String),
    FallbackTriggered,
}

struct STBIntentRouter {
    // Stage 1 Storage Blocks
    vocab: HashMap<String, usize>,
    logistic_model: LogisticRegression<f32, DenseMatrix<f32>>,
    reverse_intent_labels: HashMap<usize, String>,
    confidence_limit: f32,

    // Stage 2 Storage Blocks (Set inside standard Options to allow lazy loading)
    onnx_session: Option<Session>,
    tokenizer: Option<Tokenizer>,
    reference_profiles: HashMap<String, Vec<f32>>, // Holds target embeddings for intents
}

impl STBIntentRouter {
    pub fn route_query(&mut self, query: &str) -> String {
        // Step 1: Run fast Option 1 routing logic
        match self.evaluate_stage1(query) {
            PipelineOutput::DirectIntent(intent) => intent,
            PipelineOutput::FallbackTriggered => {
                // Step 2: Fall back to semantic processing if Option 1 fails
                self.evaluate_stage2(query)
            }
        }
    }

    fn evaluate_stage1(&self, query: &str) -> PipelineOutput {
        let num_features = self.vocab.len();
        let mut features = vec![0.0f32; num_features];
        let mut active_tokens = 0;

        for word in query.to_lowercase().split_whitespace() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
            if let Some(&col) = self.vocab.get(clean) {
                features[col] += 1.0;
                active_tokens += 1;
            }
        }

        if active_tokens == 0 {
            return PipelineOutput::FallbackTriggered;
        }

        let matrix = DenseMatrix::from_array(1, num_features, &features);
        let probabilities = self.logistic_model.predict_probabilities(&matrix).unwrap();
        
        let (best_idx, &highest_prob) = probabilities.row(0)
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap();

        if highest_prob >= self.confidence_limit {
            PipelineOutput::DirectIntent(self.reverse_intent_labels.get(&best_idx).unwrap().clone())
        } else {
            PipelineOutput::FallbackTriggered
        }
    }

    fn evaluate_stage2(&mut self, query: &str) -> String {
        // Initialize heavy session objects ONLY when a fallback occurs to preserve runtime allocation
        if self.onnx_session.is_none() {
            println!("Initializing ONNX infrastructure dynamically...");
            self.tokenizer = Some(Tokenizer::from_file("tv_onnx_assets/tokenizer.json").unwrap());
            self.onnx_session = Some(Session::builder().unwrap().commit_from_file("tv_onnx_assets/model_quantized.onnx").unwrap());
        }

        let tokenizer = self.tokenizer.as_ref().unwrap();
        let session = self.onnx_session.as_ref().unwrap();

        let encodings = tokenizer.encode(query, true).unwrap();
        let input_ids: Vec<i64> = encodings.get_ids().iter().map(|&x| x as i64).collect();
        let attention_mask: Vec<i64> = encodings.get_attention_mask().iter().map(|&x| x as i64).collect();
        let seq_len = input_ids.len();

        let input_ids_array = Array2::from_shape_vec((1, seq_len), input_ids).unwrap();
        let attention_mask_array = Array2::from_shape_vec((1, seq_len), attention_mask).unwrap();

        let outputs = session.run(inputs![
            "input_ids" => input_ids_array,
            "attention_mask" => attention_mask_array,
        ].unwrap()).unwrap();

        let token_embeddings = outputs["token_embeddings"].try_extract_tensor::<f32>().unwrap();
        let query_vec = token_embeddings.to_owned().into_dyn().mean_axis(Axis(1)).unwrap().into_shape(384).unwrap();

        let mut best_intent = String::from("unsupported");
        let mut max_similarity = 0.0f32;

        for (intent, ref_vec) in &self.reference_profiles {
            let similarity = calculate_cosine(&query_vec, ref_vec);
            if similarity > max_similarity {
                max_similarity = similarity;
                best_intent = intent.clone();
            }
        }

        // Catch entirely unrelated queries using a similarity lower boundary limit
        if max_similarity < 0.35 {
            String::from("unsupported")
        } else {
            best_intent
        }
    }
}

fn calculate_cosine(v1: &Array1<f32>, v2: &Vec<f32>) -> f32 {
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for i in 0..384 {
        dot += v1[i] * v2[i];
        norm_a += v1[i] * v1[i];
        norm_b += v2[i] * v2[i];
    }
    dot / (norm_a.sqrt() * norm_b.sqrt())
}

fn main() {
    println!("Production Hybrid Pipeline compiled cleanly.");
}
```

## Production Qualification And Safe Routing

Treat the classifier as a proposal generator, not as an authorization boundary. A predicted intent must pass deterministic application checks before it can invoke an MCP tool:

1. Validate the intent against a closed enum and the tool against an explicit allowlist.
2. Validate every argument against a typed schema; do not pass raw user text into privileged command fields without parsing and validation.
3. Enforce user/session authorization independently of model output.
4. Ask a clarifying question for ambiguous intent/entity pairs and require confirmation for actions with meaningful side effects.
5. Handle timeout, unavailable model, invalid output, low margin, out-of-domain input, and runtime errors with a safe non-action result.
6. Bound input length, inference time, memory, and model initialization; avoid logging sensitive utterances unnecessarily.

### Evaluation Protocol

Build a representative, versioned dataset with multiple paraphrases per intent, entity variation, spelling/noise variation, ambiguous requests, unsupported requests, and adversarially similar examples. Split by speaker/session or source where correlated utterances could otherwise leak across partitions. Keep a final test set untouched until model and threshold choices are complete.

Report per-intent precision and recall, macro-F1, the confusion matrix, rejection coverage, false-action rate, and latency/resource distributions. Evaluate the reject/abstain policy as part of the system. Review whether the examples represent accents, languages, speech-recognition errors, and device naming conventions expected in deployment. Re-evaluate after changing tokenizer, vocabulary, embedding model, classifier, threshold, or runtime.

For cosine similarity, reject zero-norm vectors and check all values are finite before normalization. Ensure query and intent vectors come from the same embedding model and preprocessing. Store model identity, tokenizer revision, label map, and vector normalization policy with the shipped prototypes.

The implementation sketch below is a teaching blueprint, not drop-in production code: its `unwrap()` calls, hard-coded dimensions, error handling, placeholder MCP types, and direct action routing require replacement with validated fallible APIs and application authorization before use. The embeddings and similarity scores also need task-specific validation; do not treat a successful compile as evidence of classifier quality.

### Correction To The Unseen-Input Example

The numeric thresholds and claim about uniform probability in Section 3.1 are not mathematical guarantees. An all-zero TF-IDF vector can produce class priors/bias-based outputs, while an embedding similarity score is not a calibrated class probability. Use explicit empty-feature detection, margin/threshold policies calibrated on held-out data, and a tested abstention path. Confirm the behavior on the actual classifier implementation.

For the learning order, continue through the [AI Engineering Study Roadmap](STUDY_ROADMAP.md).

---

## 7. Connecting to Model Context Protocol (MCP) Client Handlers

Once the hybrid pipeline outputs a confirmed intent token, pass the intent down into your async JSON-RPC network layer to trigger active tool calls across the system's **MCP server network**:

```rust
use serde_json::json;

async fn execute_mcp_routing_bridge(user_query: &str, router: &mut STBIntentRouter, mcp_client: &McpClientInstance) {
    let predicted_intent = router.route_query(user_query);
    
    match predicted_intent.as_str() {
        "tune_to" => {
            let arguments = json!({ "raw_text_command": user_query });
            mcp_client.call_tool("hardware_controls", "change_tv_channel", arguments).await.unwrap();
        },
        "search_programs" => {
            let arguments = json!({ "filter_query": user_query });
            mcp_client.call_tool("media_guide_server", "query_show_schedules", arguments).await.unwrap();
        },
        "current_programs" => {
            let arguments = json!({});
            mcp_client.call_tool("media_guide_server", "fetch_live_epg_grid", arguments).await.unwrap();
        },
        "unsupported" => {
            trigger_stb_error_overlay("This operation is not mapped to device functions.");
        },
        _ => {}
    }
}
```
