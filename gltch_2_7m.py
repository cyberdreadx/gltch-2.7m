"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║    ██████╗ ██╗  ████████╗ ██████╗██╗  ██╗     ██████╗    ███████╗███╗   ███╗  ║
║   ██╔════╝ ██║  ╚══██╔══╝██╔════╝██║  ██║     ╚════██╗   ╚════██║████╗ ████║  ║
║   ██║  ███╗██║     ██║   ██║     ███████║      █████╔╝       ██╔╝██╔████╔██║  ║
║   ██║   ██║██║     ██║   ██║     ██╔══██║     ██╔═══╝       ██╔╝ ██║╚██╔╝██║  ║
║   ╚██████╔╝███████╗██║   ╚██████╗██║  ██║     ███████╗██╗   ██║  ██║ ╚═╝ ██║  ║
║    ╚═════╝ ╚══════╝╚═╝    ╚═════╝╚═╝  ╚═╝     ╚══════╝╚═╝   ╚═╝  ╚═╝     ╚═╝  ║
║                                                                               ║
║   Generative Language Transformer with Contextual Hierarchy — 2.7M params    ║
║                                                                               ║
║   Created by: cyberdreadx                                                     ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

GLTCH-2.7M: A Language Model Built from Scratch
================================================

A complete, working transformer-based language model demonstrating:
  • Self-attention mechanisms
  • Multi-head attention with contextual hierarchy
  • Positional embeddings
  • Autoregressive text generation

Train on Google Colab (free GPU):
  1. Go to https://colab.research.google.com
  2. Create a new notebook  
  3. Runtime > Change runtime type > T4 GPU
  4. Paste this code and run!

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import requests

# ============================================================================
# CONFIGURATION - Tiny model that trains in ~5 minutes on Colab
# ============================================================================
config = {
    'batch_size': 64,        # How many sequences to process at once
    'block_size': 128,       # Maximum context length (how far back model looks)
    'n_embd': 192,           # Embedding dimension (model width)
    'n_head': 6,             # Number of attention heads
    'n_layer': 6,            # Number of transformer blocks (model depth)
    'dropout': 0.1,          # Dropout rate for regularization
    'learning_rate': 3e-4,   # Learning rate
    'max_iters': 3000,       # Training iterations
    'eval_interval': 300,    # How often to evaluate
    'eval_iters': 200,       # Iterations for loss estimation
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

print(f"🚀 Using device: {config['device']}")
if config['device'] == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# ============================================================================
# DATA - Download and prepare training text
# ============================================================================
print("\n📚 Loading training data...")

# Download Shakespeare (a classic small dataset for LLM demos)
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
text = response.text

print(f"   Dataset size: {len(text):,} characters")
print(f"   First 100 chars: {repr(text[:100])}")

# Create character-level tokenizer (simple but effective for demos)
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"   Vocabulary size: {vocab_size} unique characters")

# Character to integer mapping
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # string -> list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # list of integers -> string

# Encode the entire dataset
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
print(f"   Train/Val split: {len(train_data):,} / {len(val_data):,} tokens")

# Function to get a batch of training data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
    x = torch.stack([data[i:i+config['block_size']] for i in ix])
    y = torch.stack([data[i+1:i+config['block_size']+1] for i in ix])
    return x.to(config['device']), y.to(config['device'])

# ============================================================================
# MODEL ARCHITECTURE - The actual transformer!
# ============================================================================

class SelfAttention(nn.Module):
    """
    One head of self-attention.
    This is the core mechanism that lets the model look at other positions.
    """
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config['n_embd'], head_size, bias=False)
        self.query = nn.Linear(config['n_embd'], head_size, bias=False)
        self.value = nn.Linear(config['n_embd'], head_size, bias=False)
        # Causal mask: prevents looking at future tokens
        self.register_buffer('tril', torch.tril(torch.ones(config['block_size'], config['block_size'])))
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x):
        B, T, C = x.shape  # Batch, Time (sequence length), Channels (embedding dim)
        
        k = self.key(x)    # What do I contain?
        q = self.query(x)  # What am I looking for?
        v = self.value(x)  # What do I communicate?
        
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Causal mask
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        # Weighted aggregation of values
        out = wei @ v  # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads in parallel, then combined.
    Each head can learn to focus on different types of patterns.
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    Simple MLP: lets the model "think" about what it learned from attention.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config['n_embd'], 4 * config['n_embd']),
            nn.GELU(),  # Modern activation function
            nn.Linear(4 * config['n_embd'], config['n_embd']),
            nn.Dropout(config['dropout']),
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    One transformer block: attention + feedforward with residual connections.
    Stack multiple of these to create a deep network.
    """
    def __init__(self):
        super().__init__()
        head_size = config['n_embd'] // config['n_head']
        self.attention = MultiHeadAttention(config['n_head'], head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])
    
    def forward(self, x):
        x = x + self.attention(self.ln1(x))  # Residual connection
        x = x + self.ffwd(self.ln2(x))       # Residual connection
        return x


class GLTCH(nn.Module):
    """
    GLTCH-2.7M: Generative Language Transformer with Contextual Hierarchy
    2.7 million parameters | Created by: cyberdreadx
    """
    def __init__(self):
        super().__init__()
        # Token embedding: convert token IDs to vectors
        self.token_embedding = nn.Embedding(vocab_size, config['n_embd'])
        # Position embedding: encode where each token is in the sequence
        self.position_embedding = nn.Embedding(config['block_size'], config['n_embd'])
        # Stack of transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(config['n_layer'])])
        # Final layer norm
        self.ln_f = nn.LayerNorm(config['n_embd'])
        # Output projection to vocabulary
        self.lm_head = nn.Linear(config['n_embd'], vocab_size)
        
        # Count parameters
        self.n_params = sum(p.numel() for p in self.parameters())
        print(f"\n🧠 Model created with {self.n_params:,} parameters")
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Get embeddings
        tok_emb = self.token_embedding(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding(torch.arange(T, device=config['device']))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        
        # Apply transformer blocks
        x = self.blocks(x)
        x = self.ln_f(x)
        
        # Get logits (scores for each vocab token)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Calculate loss if we have targets
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        """
        Generate new tokens autoregressively.
        temperature: higher = more random, lower = more deterministic
        """
        for _ in range(max_new_tokens):
            # Crop context to block_size if needed
            idx_cond = idx[:, -config['block_size']:]
            # Get predictions
            logits, _ = self(idx_cond)
            # Focus on last time step
            logits = logits[:, -1, :] / temperature
            # Convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================================
# TRAINING - Teach the model to predict the next character
# ============================================================================

# Create GLTCH model and move to GPU
model = GLTCH().to(config['device'])
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

@torch.no_grad()
def estimate_loss():
    """Estimate loss on train and val sets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(config['eval_iters'])
        for k in range(config['eval_iters']):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print("\n🏋️ Starting training...")
print(f"   {config['max_iters']} iterations, evaluating every {config['eval_interval']}")
print("-" * 50)

for iter in range(config['max_iters']):
    # Evaluate periodically
    if iter % config['eval_interval'] == 0:
        losses = estimate_loss()
        print(f"   Step {iter:5d} | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f}")
    
    # Get batch and compute loss
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    
    # Backpropagation
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Final evaluation
losses = estimate_loss()
print("-" * 50)
print(f"   Final   | Train loss: {losses['train']:.4f} | Val loss: {losses['val']:.4f}")

# ============================================================================
# GENERATION - Let's see what our model learned!
# ============================================================================
print("\n✨ GLTCH is generating text...")
print("=" * 50)

# Start with a prompt
prompt = "ROMEO:"
context = torch.tensor([encode(prompt)], dtype=torch.long, device=config['device'])

# Generate!
generated = model.generate(context, max_new_tokens=500, temperature=0.8)
output = decode(generated[0].tolist())

print(output)
print("=" * 50)

# ============================================================================
# INTERACTIVE MODE - Try your own prompts!
# ============================================================================
print("\n🎮 GLTCH Interactive Mode! Enter a prompt (or 'quit' to exit):")

while True:
    try:
        prompt = input("\nYour prompt: ")
        if prompt.lower() == 'quit':
            break
        
        # Filter prompt to only include valid characters
        prompt = ''.join(c for c in prompt if c in stoi)
        if not prompt:
            print("(Using default prompt)")
            prompt = "The "
        
        context = torch.tensor([encode(prompt)], dtype=torch.long, device=config['device'])
        generated = model.generate(context, max_new_tokens=200, temperature=0.8)
        print("\n" + decode(generated[0].tolist()))
    except KeyboardInterrupt:
        break

print("\n👋 Thanks for training GLTCH — your own LLM!")
