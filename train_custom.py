"""
GLTCH-2.7M Custom Data Training
================================
Train GLTCH on your own text data!

Usage:
    python train_custom.py --data your_file.txt
    python train_custom.py --data your_folder/

Examples:
    python train_custom.py --data novels.txt
    python train_custom.py --data ./my_dataset/
    python train_custom.py --data https://example.com/text.txt

Created by: cyberdreadx
"""

import argparse
import os
import glob
import requests
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================
# CONFIGURATION
# ============================================

config = {
    'batch_size': 64,
    'block_size': 128,
    'n_embd': 192,
    'n_head': 6,
    'n_layer': 6,
    'dropout': 0.1,
    'learning_rate': 3e-4,
    'max_iters': 5000,
    'eval_interval': 500,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

# ============================================
# DATA LOADING
# ============================================

def load_data(source: str) -> str:
    """Load text from file, folder, or URL"""
    
    # URL
    if source.startswith('http://') or source.startswith('https://'):
        print(f"📥 Downloading from URL...")
        return requests.get(source).text
    
    # Folder
    if os.path.isdir(source):
        print(f"📂 Loading from folder: {source}")
        texts = []
        for ext in ['*.txt', '*.md', '*.py', '*.json']:
            for file in glob.glob(os.path.join(source, '**', ext), recursive=True):
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    texts.append(f.read())
        return '\n\n'.join(texts)
    
    # Single file
    if os.path.isfile(source):
        print(f"📄 Loading file: {source}")
        with open(source, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    
    raise ValueError(f"Could not load data from: {source}")


# ============================================
# MODEL (same as GLTCH-2.7M)
# ============================================

class SelfAttention(nn.Module):
    def __init__(self, head_size, vocab_size):
        super().__init__()
        self.key = nn.Linear(config['n_embd'], head_size, bias=False)
        self.query = nn.Linear(config['n_embd'], head_size, bias=False)
        self.value = nn.Linear(config['n_embd'], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config['block_size'], config['block_size'])))
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        return self.dropout(F.softmax(wei, dim=-1)) @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, vocab_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(head_size, vocab_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config['n_embd'], 4 * config['n_embd']),
            nn.GELU(),
            nn.Linear(4 * config['n_embd'], config['n_embd']),
            nn.Dropout(config['dropout']),
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        head_size = config['n_embd'] // config['n_head']
        self.attention = MultiHeadAttention(config['n_head'], head_size, vocab_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])
    
    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        return x + self.ffwd(self.ln2(x))


class GLTCH(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, config['n_embd'])
        self.position_embedding = nn.Embedding(config['block_size'], config['n_embd'])
        self.blocks = nn.Sequential(*[TransformerBlock(vocab_size) for _ in range(config['n_layer'])])
        self.ln_f = nn.LayerNorm(config['n_embd'])
        self.lm_head = nn.Linear(config['n_embd'], vocab_size)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        x = self.token_embedding(idx) + self.position_embedding(torch.arange(T, device=idx.device))
        logits = self.lm_head(self.ln_f(self.blocks(x)))
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1)) if targets is not None else None
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -config['block_size']:])
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


# ============================================
# TRAINING
# ============================================

def train(data_source: str, output_name: str = "gltch_custom"):
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   GLTCH-2.7M Custom Training — Created by: cyberdreadx                        ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"🚀 Device: {config['device']}")
    
    # Load data
    text = load_data(data_source)
    print(f"📊 Dataset size: {len(text):,} characters")
    
    # Create tokenizer
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    print(f"📝 Vocabulary: {vocab_size} unique characters")
    
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi.get(c, 0) for c in s]
    decode = lambda l: ''.join([itos.get(i, '?') for i in l])
    
    # Prepare data
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]
    
    def get_batch(split):
        d = train_data if split == 'train' else val_data
        ix = torch.randint(len(d) - config['block_size'], (config['batch_size'],))
        x = torch.stack([d[i:i+config['block_size']] for i in ix])
        y = torch.stack([d[i+1:i+config['block_size']+1] for i in ix])
        return x.to(config['device']), y.to(config['device'])
    
    # Create model
    model = GLTCH(vocab_size).to(config['device'])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model: {n_params:,} parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    print(f"\n🏋️ Training for {config['max_iters']} steps...")
    print("-" * 50)
    
    for step in range(config['max_iters']):
        if step % config['eval_interval'] == 0:
            model.eval()
            losses = {split: torch.tensor([model(*get_batch(split))[1].item() for _ in range(50)]).mean() 
                      for split in ['train', 'val']}
            print(f"Step {step:5d} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f}")
            model.train()
        
        _, loss = model(*get_batch('train'))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    print("-" * 50)
    
    # Save model and tokenizer
    save_path = f"{output_name}.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': vocab_size,
        'chars': chars,
        'config': config
    }, save_path)
    print(f"💾 Model saved to: {save_path}")
    
    # Generate sample
    print("\n✨ Generated sample:")
    print("=" * 50)
    prompt = text[:20]  # Use first 20 chars as prompt
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=config['device'])
    generated = model.generate(context, max_new_tokens=300)
    print(decode(generated[0].tolist()))
    print("=" * 50)
    
    return model, encode, decode


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GLTCH on custom data")
    parser.add_argument("--data", required=True, help="Path to text file, folder, or URL")
    parser.add_argument("--output", default="gltch_custom", help="Output model name")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    
    args = parser.parse_args()
    config['max_iters'] = args.steps
    
    train(args.data, args.output)
