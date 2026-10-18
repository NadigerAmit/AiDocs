# Embedded Intent Classification for TV/STB via MCP: A Technical Guide

This guide details two lightweight architectures to map user voice or text inputs into a predefined set of intents (`tune_to`, `channel_lookup`, `search_programs`, `list_channels`, `current_programs`, `favorites`, `unsupported`) inside a resource-constrained environment (Smart TV / Set-Top Box) using the **Rust programming language**. These mapped intents can then be natively routed as tool calls to a local or remote **Model Context Protocol (MCP)** server.

---

## Pre-Defined Intent Actions
Your application maps text instructions to these specific tool tokens:
*   `tune_to`: Switch to a specific channel (e.g., *"open Das Erste"*).
*   `channel_lookup`: Check availability or information of a channel (e.g., *"is CNN available"*).
*   `search_programs`: Search content by genre, actor, or type (e.g., *"look for movies"*).
*   `list_channels`: List or browse accessible services (e.g., *"show available services"*).
*   `current_programs`: Electronic Program Guide / Now-Next queries (e.g., *"could you show epg"*).
*   `favorites`: Retrieve saved or bookmarked lists (e.g., *"tell me list favorites"*).
*   `unsupported`: Gracefully catch out-of-scope actions (e.g., *"please book a taxi"*).

---

## Architectural Comparison Matrix

| Feature / Metric | Option 1: TF-IDF + Linear Classifier (`smartcore`) | Option 2: ONNX Runtime + Embeddings (`ort`) |
| :--- | :--- | :--- |
| **Needs Training?** | 🔄 **Yes.** Requires compilation or offline training on your specific dataset. | ❌ **No.** Uses the general language understanding capabilities of the model out-of-the-box. |
| **Setup Complexity** | 🟢 **Low.** Pure Rust dependencies; zero external runtime environment needed. | 🟡 **Moderate.** Requires downloading, converting, and bundling asset binaries. |
| **Memory (RAM) Footprint** | ⚡ **Ultra Lean (< 2 MB).** Negligible overhead for memory-starved SoCs. | 💾 **Compact (~25 MB - 30 MB).** Tiny for AI, but significantly more than Option 1. |
| **CPU Inference Latency** | 🚀 **Microseconds (< 1ms).** Simple dot-product math over a vocabulary matrix. | 🐢 **2–8 Milliseconds.** Highly optimized matrix transformations executed on the CPU core. |
| **Handling Synonyms** | ❌ **Weak.** Relies entirely on matching variations or keywords present during compilation. | 🎯 **Exceptional.** Conceptually understands that *"fetch"*, *"find"*, and *"locate"* share the same meaning. |
| **Binary Asset Size** | 📦 **Minimal (< 1 MB).** Vocabulary maps and floats compile right into your application binary. | 📦 **~35 MB total.** Requires storing an external `.onnx` weight model and `tokenizer.json` configuration file on the device. |

---

## Option 1: The Lean Path (TF-IDF + Linear Classifier)

### How to Achieve
This setup builds an N-gram vocabulary token table out of your dataset and trains a Support Vector Machine (Linear SVC) or Logistic Regression model. The model coefficients compile down to direct array lookups at runtime.

### Configuration (`Cargo.toml`)
```toml
[dependencies]
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
smartcore = { version = "0.3", features = ["linalg-ndarray"] }
ndarray = "0.15"
```

### Pure Rust Blueprint Crate Example
```rust
use serde::Deserialize;
use smartcore::linalg::naive::dense_matrix::DenseMatrix;
use smartcore::svc::linear_svc::LinearSVC;
use std::collections::HashMap;

#[derive(Debug, Deserialize)]
struct TrainingItem {
    instruction: String,
    output: String,
}

fn main() {
    // 1. Ingestion point for your provided JSON training schema
    let raw_data = r#"
    [
        {"instruction": "please book a taxi quickly", "output": "unsupported"},
        {"instruction": "could you show epg for me", "output": "current_programs"},
        {"instruction": "open Das Erste", "output": "tune_to"},
        {"instruction": "locate CNN", "output": "channel_lookup"}
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
    let y = vec![y_data];

    // 4. Fit the native classifier matrix
    let model = LinearSVC::fit(&x, &y.into_iter().next().unwrap(), Default::default()).unwrap();
    println!("Compilation successful! Loaded tokens: {}", num_features);

    // 5. Embedded Query Runtime Application
    let live_query = "change to Das Erste";
    let mut query_features = vec![0.0f32; num_features];
    for word in live_query.to_lowercase().split_whitespace() {
        let clean = word.trim_matches(|c: char| !c.is_alphanumeric());
        if let Some(&col) = vocab.get(clean) {
            query_features[col] += 1.0;
        }
    }
    
    let query_matrix = DenseMatrix::from_array(1, num_features, &query_features);
    let prediction = model.predict(&query_matrix).unwrap();
    let predicted_intent_id = prediction[0] as usize;
    
    let final_intent = reverse_intent_map.get(&predicted_intent_id).unwrap();
    println!("Resulting MCP Routing Action: Target Intent Tool -> '{}'", final_intent);
}
```

---

## Option 2: The Semantic Variant (ONNX Runtime + Embeddings)

### How to Achieve
This setup maps sentences into a vector space using a pre-trained language model, resolving synonyms dynamically. We use Microsoft's Open Neural Network Exchange (ONNX) format for cross-platform execution directly on the STB device.

### 1. Model Selection & Download from Hugging Face
We recommend using **`all-MiniLM-L6-v2`** due to its micro-sized dimensions (384 output vector space) and optimal balance of speed and footprint.

*   **Official Model Repository:** [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
*   **Alternative Option (Higher Context Baseline):** [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)

### 2. How to Convert to ONNX & Quantize (INT8 Compression)
To compress the model weights from 130MB down to roughly ~30MB for low-tier hardware, perform this conversion step on a development workstation using Python's `optimum` toolkit:

```bash
# Install toolchains
pip install optimum[onnxruntime]

# Download, convert, and export as optimized quantized binary structures
optimum-cli export onnx --model sentence-transformers/all-MiniLM-L6-v2 --optimize O3 --quantize int8 quantized_model/
```
*This produces two highly critical files needed for the device local filesystem: `model.onnx` (or `model_quantized.onnx`) and `tokenizer.json`.*

### 3. Configuration (`Cargo.toml`)
```toml
[dependencies]
ort = { version = "2.0", features = ["ndarray"] }
tokenizers = "0.15" 
ndarray = "0.15"
```

### 4. Implementation Example using `ort`
```rust
use ndarray::{Array1, Array2, Axis};
use ort::{inputs, Session};
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Initialize local assets on device storage
    let tokenizer = Tokenizer::from_file("tokenizer.json")?;
    let session = Session::builder()?.commit_from_file("model_quantized.onnx")?;

    let user_query = "can you please show me what channel CNN is on?";

    // 2. Tokenize the incoming string structure
    let encodings = tokenizer.encode(user_query, true)?;
    let input_ids: Vec<i64> = encodings.get_ids().iter().map(|&x| x as i64).collect();
    let attention_mask: Vec<i64> = encodings.get_attention_mask().iter().map(|&x| x as i64).collect();
    let length = input_ids.len();

    let input_ids_array = Array2::from_shape_vec((1, length), input_ids)?;
    let attention_mask_array = Array2::from_shape_vec((1, length), attention_mask)?;

    // 3. Compute Vector Embeddings via ONNX Engine
    let outputs = session.run(inputs![
        "input_ids" => input_ids_array,
        "attention_mask" => attention_mask_array,
    ]?)?;

    let token_embeddings = outputs["token_embeddings"].try_extract_tensor::<f32>()?;
    
    // Mean pooling aggregates multi-token outputs into a distinct 384 float footprint array
    let query_vector = mean_pooling(token_embeddings.to_owned().into_dyn());

    // 4. Intent classification matching routine
    let final_intent = find_closest_intent_match(query_vector);
    println!("Dynamic Routed Intent: {}", final_intent);

    Ok(())
}

fn mean_pooling(tensor: ndarray::ArrayD<f32>) -> Array1<f32> {
    tensor.mean_axis(Axis(1)).unwrap().into_shape(384).unwrap()
}

fn find_closest_intent_match(_vec: Array1<f32>) -> String {
    // Implement standard dot-product/cosine similarity against pre-stored vectors here
    "channel_lookup".to_string()
}
```

---

## Wiring Intent Output to the MCP Client Architecture

Once the engine outputs your clean intent target string, map the classification token to the payload properties expected by your Model Context Protocol Client driver instance:

```rust
async fn process_stb_input(user_command: &str) {
    // Resolve the string using either Option 1 or Option 2 routing algorithms
    let classified_intent = resolve_intent(user_command); 

    match classified_intent.as_str() {
        "tune_to" => {
            let mcp_args = serde_json::json!({ "target_channel": user_command });
            mcp_client.call_tool("hardware_manager", "change_channel", mcp_args).await.unwrap();
        },
        "search_programs" => {
            let mcp_args = serde_json::json!({ "filter_criteria": user_command });
            mcp_client.call_tool("media_database", "query_shows", mcp_args).await.unwrap();
        },
        "unsupported" => {
            log::warn!("Command out of scope for TV hardware controller context.");
        },
        _ => {}
    }
}
```
