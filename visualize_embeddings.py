import os
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.manifold import TSNE
from gpt_train import SimpleGPT 
from collections import defaultdict

# --- 1. Helper Functions ---

def mean_pool(hidden, mask):
    """Averages non-padded token embeddings for each chunk."""
    mask_float = (~mask).float().unsqueeze(-1)
    hidden = hidden * mask_float
    summed = hidden.sum(dim=1)
    counts = mask_float.sum(dim=1).clamp(min=1)
    return summed / counts

def collate_fn_trajectories(batch, pad_token_id):
    """Pads chunks within a batch for the extraction loader."""
    lengths = [len(s) for s in batch]
    max_len = max(lengths)
    padded, mask = [], []
    for s in batch:
        pad_len = max_len - len(s)
        padded.append(torch.cat([s, torch.full((pad_len,), pad_token_id)]))
        mask.append(torch.cat([torch.zeros_like(s, dtype=torch.bool), 
                               torch.ones(pad_len, dtype=torch.bool)]))
    return torch.stack(padded), torch.stack(mask)

class TrajectoryDataset(Dataset):
    """Dataset for full token sequences."""
    def __init__(self, sequences, pad_id):
        self.sequences = sequences
        self.pad_id = pad_id
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        item = self.sequences[idx]
        return item if torch.is_tensor(item) else torch.tensor(item, dtype=torch.long)

def process_chunks_with_id_tracking(raw_sequences, raw_ids, max_len, step):
    """Splits sequences into chunks while tracking which ID belongs to which chunk."""
    split_sequences = []
    chunk_to_id_map = [] 
    for seq, uid in zip(raw_sequences, raw_ids):
        if len(seq) <= max_len:
            if len(seq) > 10:
                split_sequences.append(torch.tensor(seq, dtype=torch.long))
                chunk_to_id_map.append(uid)
        else:
            for i in range(0, len(seq), step):
                chunk = seq[i : i + max_len]
                if len(chunk) > 10:
                    split_sequences.append(torch.tensor(chunk, dtype=torch.long))
                    chunk_to_id_map.append(uid)
                if i + max_len >= len(seq):
                    break
    return split_sequences, chunk_to_id_map

# --- 2. Main Execution ---

if __name__ == "__main__":
    # 1. Loading the best model and configurations
    checkpoint_path = "gpt_train_results/best_model.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Best model not found! Run training first.")
    
    checkpoint = torch.load(checkpoint_path)
    config = checkpoint['config']
    vocab = checkpoint['vocab']
    pad_token_id = vocab.get("<PAD>", len(vocab))

    # 2. Loading validation sequences and IDs
    with open("token_sequences_int_val.pkl", "rb") as f:
        val_data_raw = pickle.load(f)
    with open("ids_val.pkl", "rb") as f:
        val_ids_raw = pickle.load(f)

    # 3. Processing diagnoses to identify target metabolic diseases
    # We load the full TSV which might have multiple rows per individual
    diagnosis_df = pd.read_csv("diagnosis.tsv", sep="\t")

    # STEP: Identify all individuals who have AT LEAST ONE diagnosis starting with "ICD10:DE"
    # We convert the column to string and check the prefix
    metabolic_mask = diagnosis_df['diagnosis'].astype(str).str.startswith("ICD10:DE")
    metabolic_ids = set(diagnosis_df.loc[metabolic_mask, 'cpr_enc'].unique())

    # 4. Preparing trajectory chunks for the Transformer
    val_data = val_data_raw[:config['val_limit']]
    val_ids_limit = val_ids_raw[:config['val_limit']]
    
    max_len = config['max_seq_len']
    step = max_len - 100 
    val_chunks, val_chunk_ids = process_chunks_with_id_tracking(val_data, val_ids_limit, max_len, step)

    # 5. Model Inference (Embedding Extraction)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleGPT(
        vocab_size=len(vocab)+1, 
        d_model=config['d_model'], 
        n_heads=config['n_heads'], 
        d_ff=config['d_ff'], 
        n_layers=config['n_layers'],
        max_seq_len=config['max_seq_len']
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    dataset = TrajectoryDataset(val_chunks, pad_token_id)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False, 
                        collate_fn=lambda b: collate_fn_trajectories(b, pad_token_id))

    all_chunk_embeddings = []
    with torch.no_grad():
        for x, mask in loader:
            x, mask = x.to(device), mask.to(device)
            hidden = model.extract_hidden(x, key_padding_mask=mask)
            # Pool token-level embeddings to chunk-level
            chunk_emb = mean_pool(hidden, mask)
            all_chunk_embeddings.append(chunk_emb.cpu().numpy())

    all_chunk_embeddings = np.concatenate(all_chunk_embeddings, axis=0)

    # 6. Aggregating chunk embeddings to the patient level
    patient_to_chunks = defaultdict(list)
    for i, uid in enumerate(val_chunk_ids):
        patient_to_chunks[uid].append(all_chunk_embeddings[i])
    
    unique_ids, patient_embeddings = [], []
    for uid, chunks in patient_to_chunks.items():
        unique_ids.append(uid)
        # Average all chunks belonging to the same individual
        patient_embeddings.append(np.mean(chunks, axis=0))
    
    patient_embeddings = np.array(patient_embeddings)

    # 7. Binary Categorization for Plotting
    # STEP: Label each patient based on whether their ID was in the 'metabolic_ids' set
    labels = []
    for uid in unique_ids:
        if uid in metabolic_ids:
            labels.append("participants with metabolic disease")
        else:
            labels.append("participants without metabolic diseases")
    
    # Categorize labels for consistent coloring
    cat_labels = pd.Categorical(labels, categories=[
        "participants with metabolic disease", 
        "participants without metabolic diseases"
    ])
    color_indices = cat_labels.codes

    # 8. Visualization with t-SNE
    print(f">>> Projecting {len(unique_ids)} patients to 2D space...")
    # Adjust perplexity to avoid errors if the dataset is very small
    tsne_perp = min(30, max(1, len(unique_ids) - 1))
    tsne = TSNE(n_components=2, perplexity=tsne_perp, random_state=42)
    embeddings_2d = tsne.fit_transform(patient_embeddings)

    # 9. Creating and Saving the Plot
    plt.figure(figsize=(14, 10))
    
    # Define distinct colors for the binary classes
    colors = ['red', 'blue'] # Red for Metabolic, Blue for Non-Metabolic
    
    for i, label_name in enumerate(cat_labels.categories):
        mask = (cat_labels == label_name)
        plt.scatter(
            embeddings_2d[mask, 0], 
            embeddings_2d[mask, 1], 
            label=label_name,
            alpha=0.8, 
            c=colors[i],
            edgecolors='k',
            s=100
        )

    # Annotate points with IDs (optional, helps with verification)
    for i, label in enumerate(unique_ids):
        plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), fontsize=8, alpha=0.5)

    plt.title("t-SNE of Patient Trajectories (Metabolic vs Non-Metabolic Diseases)")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.legend(title="Disease Status")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Save the result
    save_path = "gpt_train_results/tsne_metabolic_distinction.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # Close to free memory and avoid showing the interactive window
    
    print(f">>> Visualization saved to {save_path}")