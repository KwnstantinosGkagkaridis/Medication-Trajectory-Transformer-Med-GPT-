# Med-GPT: Autoregressive Transformer for Medication Trajectories

This repository contains a PyTorch implementation of a GPT-style (decoder-only) Transformer designed to model longitudinal patient medication histories. 

The model treats medical histories as a "language," where drug codes and time intervals form a sequence. By training the model to predict the next event in a patient's history, we can extract high-dimensional **Patient Embeddings** that summarize complex clinical trajectories for downstream phenotyping.

---

## 💡 The Concept

Standard clinical models often treat medical codes as a static "bag-of-words." This project captures **temporal dynamics** by representing a patient's history as an interleaved sequence:

`[START] -> [Δt_0] -> [ATC_1] -> [Δt_1] -> [ATC_2] ... [END]`

* **ATC Tokens:** Represent specific medication classes (Anatomical Therapeutic Chemical Classification System).
* **Time Tokens ($\Delta t$):** Represent discretized time bins between prescriptions (e.g., "0–7 days", "1 month").
* **Self-Supervised Learning:** By predicting the next token, the model learns the underlying patterns of polypharmacy and disease progression without requiring manual labels.



---

## 🏗️ Model Architecture

The core of the system is a **SimpleGPT** model featuring:

* **Causal Self-Attention:** Uses a triangular mask to ensure that at any position $t$, the model can only attend to past events ($<t$), preventing information leakage from the "future."
* **Transformer Decoder Blocks:** Multiple layers featuring:
    * **Multi-Head Attention:** To find correlations between medications separated by long time gaps.
    * **Pre-Normalization:** `LayerNorm` is applied before attention and MLP blocks for more stable training.
    * **Position-wise Feed-Forward Networks (MLP):** Two linear layers with ReLU activation.
* **Contextual Embeddings:** A dedicated `extract_hidden()` method pulls the final hidden states (the "thought vectors") for tasks like patient clustering or risk stratification.

---

## 🚀 Key Features

* **Sliding Window Processing:** Automatically chunks long patient histories into segments of `max_seq_len` with a configurable overlap to maintain longitudinal context.
* **ID & Chunk Tracking:** Maps chunk-level embeddings back to unique patient IDs for aggregated analysis.
* **Phenotype Validation:** Includes scripts to validate embeddings by color-coding patients based on clinical labels (e.g., Metabolic Diseases via ICD-10 "DE" codes).
* **Early Stopping:** Automatically halts training when validation loss plateaus to prevent overfitting.
* **Flexible Configuration:** All hyperparameters are managed via a single `YAML` file.

---

## 🛠️ Usage Guide

### 1. Training the Model
To start the autoregressive training process, run:
```bash
python gpt_train.py config.yaml
```

### 2. Visualization & Extraction
To extract the learned embeddings and generate a t-SNE plot (comparing Metabolic vs. Non-Metabolic phenotypes), run:
```bash
python visualize_embeddings.py
```

---

## 📊 Analysis & Results
### Trajectory Visualization (t-SNE)
The model projects complex, years-long medication histories into a 2D space. The resulting clusters demonstrate how the "language" of prescriptions naturally separates different patient phenotypes.

### File Outputs
All results are saved to the gpt_train_results/ directory:

* **best_model.pt: The trained model weights, optimizer state, and configuration.

* **loss_epoch_N.png: Visualizations of training and validation loss curves.

* **tsne_metabolic_distinction.png: A 2D projection showing the clinical separation between patient groups.
