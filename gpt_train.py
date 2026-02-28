import os
import pickle
import argparse
import random
import yaml  
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

# Model Components 

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1): # d_model: dimensions of input embeddings, n_heads: no of attention heads, d_ff: hidden dimension
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model) # pre-normalization: Normalizes the input embeddings before attention (x_normalized = (x - mean) / std)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout) # attention: learns relationships between the tokens
                                                                             # output: weighted sum of values, per head, concatenated
        self.norm2 = nn.LayerNorm(d_model) # normalization again
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(), 
            nn.Linear(d_ff, d_model) # Linear -> ReLU -> Linear
        )
        self.dropout = nn.Dropout(dropout) # randomly zeroes some activations during training

    def forward(self, x, attn_mask=None, key_padding_mask=None): # input tensor
        x_norm = self.norm1(x) # pre-norm
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, 
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask) # queries, keys and values set to x_norm (self-attention)
                                                                   # src_key_padding_mask ensures padded positions are ingored
        x = x + self.dropout(attn_out) # residual connection
        x_norm = self.norm2(x) # 3rd normalization
        mlp_out = self.mlp(x_norm) # mlp transformation
        x = x + self.dropout(mlp_out) # residual connection again
        return x # returns tensor (seq_len, batch, d_model)


def generate_causal_mask(seq_len, device=None): # use a boolean tensor to make as True all future positions that need to be ingored 
    mask = torch.triu(torch.ones((seq_len, seq_len), device=device), diagonal=1).bool()
    return mask 

class SimpleGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, d_ff, n_layers, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model) # converts each integer token into a trainable d_model-dimensional vector
        self.pos_emb = nn.Embedding(max_seq_len, d_model) 
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers) # the input will go through <n_layers> encoder blocks sequentially
        ])
        self.norm = nn.LayerNorm(d_model) # final normalization layer 
        self.output = nn.Linear(d_model, vocab_size) # maps each contextualized token representation to a distribution across all possible next tokens

    def forward(self, x, key_padding_mask=None):
        batch_size, seq_len = x.size() # input shape
        device = x.device 
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)
        x = self.token_emb(x) + self.pos_emb(positions) # token embedding + positional information
        x = x.transpose(0, 1) # makes the shape from (batch, seq_len, d_model) to (seq_len, batch, d_model) which is the required for PyTorch MultiHeadAttention
        attn_mask = generate_causal_mask(seq_len, device=device)
        for layer in self.layers: # pass through all encoder blocks
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        x = self.norm(x) # final layer norm
        x = x.transpose(0, 1) # back to (batch, seq_len, d_model)
        logits = self.output(x) # predict next token logits
        return logits # these logits will go into a CrossEntropyLoss where softmax is applied internally
    
    def extract_hidden(self, x, key_padding_mask=None):
        """Returns the contextualized representations (batch, seq_len, d_model)."""
        batch_size, seq_len = x.size()
        device = x.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_emb(x) + self.pos_emb(positions)
        x = x.transpose(0, 1) # (seq_len, batch, d_model)
    
        attn_mask = generate_causal_mask(seq_len, device=device)

        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        x = self.norm(x)
        x = x.transpose(0, 1) # (batch, seq_len, d_model)
        return x
    
class MedicationDataset(Dataset): # Input / target splitting
    def __init__(self, sequences, pad_id):
        self.sequences = sequences
        self.pad_id = pad_id

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        x = seq[:-1]
        y = seq[1:]
        mask = (x == self.pad_id)
        return x, y, mask

# --- Main Execution ---

if __name__ == "__main__": 
    # 1. Setup Argparse to take the YAML file path
    parser = argparse.ArgumentParser(description="GPT Training with YAML config.")
    parser.add_argument("config", type=str, help="Path to the config.yaml file")
    args = parser.parse_args()

    # 2. Load the YAML configuration
    print(f">>> Stage 0: Loading configuration from {args.config}")
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    print(">>> Stage 1: Loading and Filtering Data")
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    # with open("token_sequences_int.pkl", "rb") as f:
    #     full_data = pickle.load(f)

    with open("token_sequences_int_train.pkl", "rb") as f:
        train_data_raw = pickle.load(f)

    with open("token_sequences_int_val.pkl", "rb") as f:
        val_data_raw = pickle.load(f)


    # Note: IDs are loaded but not used in training; they are kept for your future visualization needs.
    with open("ids_train.pkl", "rb") as f:
        train_ids = pickle.load(f)

    with open("ids_val.pkl", "rb") as f:
        val_ids = pickle.load(f)

    # # Use data_limit from config
    # token_sequences_int = full_data[:config['data_limit']]

    train_data = train_data_raw[:config['train_limit']]
    val_data = val_data_raw[:config['val_limit']]
    
    pad_token_id = vocab.get("<PAD>", len(vocab))
    vocab_size = len(vocab) if "<PAD>" in vocab else len(vocab) + 1

    # Stage 1: Loading and Filtering Data
    overlap = 100
    max_len = config['max_seq_len']
    step = max_len - overlap  # The distance to jump for each new chunk



    # Function to process the sliding window for both sets
    def process_chunks(raw_sequences):
        split_sequences = []
        for seq in raw_sequences:
            if len(seq) <= max_len:
                if len(seq) > 10:
                    split_sequences.append(torch.tensor(seq, dtype=torch.long))
            
            else:
                for i in range(0, len(seq), step):
                    chunk = seq[i : i + max_len]
                    if len(chunk) > 10:
                        split_sequences.append(torch.tensor(chunk, dtype=torch.long))
                    if i + max_len >=len(seq):
                        break
        return split_sequences
    
    train_chunks = process_chunks(train_data)
    val_chunks = process_chunks(val_data)



    train_padded = pad_sequence(train_chunks, batch_first=True, padding_value=pad_token_id)
    val_padded = pad_sequence(val_chunks, batch_first=True, padding_value=pad_token_id)

    train_dataset = MedicationDataset(train_padded, pad_token_id)
    val_dataset = MedicationDataset(val_padded, pad_token_id)

    print(">>> Stage 2: Data Loaders Setup")


    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

    print(">>> Stage 3: Model Setup")
    model = SimpleGPT(
        vocab_size=vocab_size, 
        d_model=config['d_model'], 
        n_heads=config['n_heads'], 
        d_ff=config['d_ff'], 
        n_layers=config['n_layers'], 
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)

    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    best_model_state = None 
    epochs_no_improve = 0
    patience = config.get('early_stopping_patience')
    best_epoch = 0

    print(">>> Stage 4: Starting Training Loop")
    os.makedirs("gpt_train_results", exist_ok=True)

    for epoch in range(config['epochs']):
        # Training
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        for x, y, mask in train_bar:
            optimizer.zero_grad()
            logits = model(x, key_padding_mask=mask)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]")
            for x, y, mask in val_bar:
                logits = model(x, key_padding_mask=mask)
                loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)


        # Best Model Tracking and Early Stopping Logic
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            epochs_no_improve = 0
            print(f"--- New best model found at epoch {best_epoch} (Val Loss: {best_val_loss:.4f}) ---")
        else:
            epochs_no_improve += 1
            print(f"--- No improvement for {epochs_no_improve} epoch(s). Best was epoch {best_epoch} ---")

        if epochs_no_improve >= patience:
            print(f">>> Early stopping triggered! Stopping training at epoch {epoch+1}.")
            break


        # Stage 5: Save Plot
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss")
        plt.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss")
        plt.legend()
        plt.savefig(f"gpt_train_results/loss_epoch_{epoch+1}.png")
        plt.close()


    # Save the best model for later use
    if best_model_state is not None:
        torch.save({
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': best_val_loss,
            'config': config,
            'vocab': vocab
        }, "gpt_train_results/best_model.pt")
        print(f"\n>>> Best model from epoch {best_epoch} saved to gpt_train_results/best_model.pt")

    print("\n>>> Done! Check gpt_train_results/ for your plots.")


