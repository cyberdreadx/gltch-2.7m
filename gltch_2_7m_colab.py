# %% [markdown]
"""
# ╔═══════════════════════════════════════════════════════════════════════════════╗
# ║    ██████╗ ██╗  ████████╗ ██████╗██╗  ██╗                                     ║
# ║   ██╔════╝ ██║  ╚══██╔══╝██╔════╝██║  ██║                                     ║
# ║   ██║  ███╗██║     ██║   ██║     ███████║                                     ║
# ║   ██║   ██║██║     ██║   ██║     ██╔══██║                                     ║
# ║   ╚██████╔╝███████╗██║   ╚██████╗██║  ██║                                     ║
# ║    ╚═════╝ ╚══════╝╚═╝    ╚═════╝╚═╝  ╚═╝                                     ║
# ║                                                                               ║
# ║   Generative Language Transformer with Contextual Hierarchy                  ║
# ║                                                                               ║
# ║   Created by: cyberdreadx                                                     ║
# ╚═══════════════════════════════════════════════════════════════════════════════╝

## GLTCH: A Language Model Built from Scratch

**Instructions:**
1. Go to Runtime → Change runtime type → **T4 GPU**
2. Run each cell in order (Shift+Enter)
3. Training takes ~5 minutes
4. Generate your own text at the end!
"""

# %% [markdown]
"""
## 📦 Cell 1: Imports & Setup
"""

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

print("✅ Imports loaded!")
print(f"🔧 PyTorch version: {torch.__version__}")
print(f"🚀 CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")

# %% [markdown]
"""
## ⚙️ Cell 2: Configuration

Adjust these to change model size and training duration.
"""

# %%
config = {
    'batch_size': 64,        # Sequences per batch
    'block_size': 128,       # Context length
    'n_embd': 192,           # Embedding dimension
    'n_head': 6,             # Attention heads
    'n_layer': 6,            # Transformer blocks
    'dropout': 0.1,          # Regularization
    'learning_rate': 3e-4,   
    'max_iters': 3000,       # Training steps
    'eval_interval': 300,    
    'eval_iters': 200,       
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
}

print(f"📊 Config loaded! Using: {config['device']}")

# %% [markdown]
"""
## 📚 Cell 3: Load & Prepare Data

Downloads Shakespeare (~1MB) and creates a character-level tokenizer.
"""

# %%
# Download Shakespeare
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
text = requests.get(url).text

# Character-level tokenizer
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Train/val split
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == 'train' else val_data
    ix = torch.randint(len(d) - config['block_size'], (config['batch_size'],))
    x = torch.stack([d[i:i+config['block_size']] for i in ix])
    y = torch.stack([d[i+1:i+config['block_size']+1] for i in ix])
    return x.to(config['device']), y.to(config['device'])

print(f"📚 Data loaded!")
print(f"   Characters: {len(text):,}")
print(f"   Vocab size: {vocab_size}")
print(f"   Train/Val: {len(train_data):,} / {len(val_data):,}")

# %% [markdown]
"""
## 🧠 Cell 4: Model Architecture

The core transformer components:
- **SelfAttention**: How tokens look at each other
- **MultiHeadAttention**: Multiple attention patterns in parallel
- **FeedForward**: Processing after attention
- **TransformerBlock**: Attention + FeedForward with residuals
- **GLTCH**: The complete model!
"""

# %%
class SelfAttention(nn.Module):
    """One head of self-attention"""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(config['n_embd'], head_size, bias=False)
        self.query = nn.Linear(config['n_embd'], head_size, bias=False)
        self.value = nn.Linear(config['n_embd'], head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(config['block_size'], config['block_size'])))
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v


class MultiHeadAttention(nn.Module):
    """Multiple attention heads in parallel"""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([SelfAttention(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(config['n_embd'], config['n_embd'])
        self.dropout = nn.Dropout(config['dropout'])
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """MLP processing layer"""
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
    """Transformer block: attention + feedforward"""
    def __init__(self):
        super().__init__()
        head_size = config['n_embd'] // config['n_head']
        self.attention = MultiHeadAttention(config['n_head'], head_size)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(config['n_embd'])
        self.ln2 = nn.LayerNorm(config['n_embd'])
    
    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GLTCH(nn.Module):
    """
    GLTCH: Generative Language Transformer with Contextual Hierarchy
    Created by: cyberdreadx
    """
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, config['n_embd'])
        self.position_embedding = nn.Embedding(config['block_size'], config['n_embd'])
        self.blocks = nn.Sequential(*[TransformerBlock() for _ in range(config['n_layer'])])
        self.ln_f = nn.LayerNorm(config['n_embd'])
        self.lm_head = nn.Linear(config['n_embd'], vocab_size)
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=config['device']))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -config['block_size']:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Create the model
model = GLTCH().to(config['device'])
print(f"🧠 GLTCH created!")
print(f"   Parameters: {model.n_params:,}")

# %% [markdown]
"""
## 🏋️ Cell 5: Training

This trains the model for ~5 minutes on a T4 GPU.
Watch the loss decrease!
"""

# %%
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

@torch.no_grad()
def estimate_loss():
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

print("🏋️ Training GLTCH...")
print("-" * 50)

for iter in range(config['max_iters']):
    if iter % config['eval_interval'] == 0:
        losses = estimate_loss()
        print(f"Step {iter:5d} | Train: {losses['train']:.4f} | Val: {losses['val']:.4f}")
    
    xb, yb = get_batch('train')
    _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

losses = estimate_loss()
print("-" * 50)
print(f"✅ Done! | Train: {losses['train']:.4f} | Val: {losses['val']:.4f}")

# %% [markdown]
"""
## ✨ Cell 6: Generate Text!

Run this cell to generate Shakespeare-like text.
Change the `prompt` to try different starting points!
"""

# %%
prompt = "ROMEO:"  # 👈 Change this!
temperature = 0.8  # Higher = more random, Lower = more focused

context = torch.tensor([encode(prompt)], dtype=torch.long, device=config['device'])
generated = model.generate(context, max_new_tokens=500, temperature=temperature)
output = decode(generated[0].tolist())

print("✨ GLTCH Generated:")
print("=" * 50)
print(output)
print("=" * 50)

# %% [markdown]
"""
## 🎮 Cell 7: Interactive Mode (Optional)

Run this for a input-based interface. Type 'quit' to exit.
"""

# %%
print("🎮 GLTCH Interactive Mode!")
print("Type a prompt and press Enter. Type 'quit' to exit.\n")

while True:
    prompt = input("Your prompt: ")
    if prompt.lower() == 'quit':
        break
    
    prompt = ''.join(c for c in prompt if c in stoi) or "The "
    context = torch.tensor([encode(prompt)], dtype=torch.long, device=config['device'])
    generated = model.generate(context, max_new_tokens=200, temperature=0.8)
    print("\n" + decode(generated[0].tolist()) + "\n")

print("\n👋 Thanks for training GLTCH — your own LLM!")
