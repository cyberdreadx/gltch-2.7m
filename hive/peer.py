"""
GLTCH HIVE — Training Peer
===========================
Run this script to join the training hive and contribute GPU power.

Usage:
    python peer.py --server ws://localhost:8765 --name my-node

Created by: cyberdreadx
"""

import asyncio
import json
import argparse
import platform
import random
import time
from dataclasses import dataclass

# Try imports
try:
    import websockets
except ImportError:
    print("❌ websockets not installed. Run: pip install websockets")
    exit(1)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("⚠️  PyTorch not installed. Running in simulation mode.")


# ============================================
# GPU DETECTION
# ============================================

def get_gpu_info():
    """Detect GPU and return info string"""
    if HAS_TORCH and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    elif HAS_TORCH and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return "Apple Silicon (MPS)"
    else:
        return f"CPU ({platform.processor() or 'Unknown'})"


def get_device():
    """Get the best available device"""
    if HAS_TORCH:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
    return None


# ============================================
# MINI MODEL (for training demo)
# ============================================

class MiniTransformer(nn.Module):
    """Tiny transformer for demonstration"""
    def __init__(self, vocab_size=256, n_embd=64, n_head=4, n_layer=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, n_embd)
        self.pos = nn.Embedding(64, n_embd)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(n_embd, n_head, n_embd * 4, batch_first=True)
            for _ in range(n_layer)
        ])
        self.ln = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
    
    def forward(self, x):
        b, t = x.shape
        tok = self.embed(x)
        pos = self.pos(torch.arange(t, device=x.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        return self.head(x)


# ============================================
# TRAINING PEER
# ============================================

class TrainingPeer:
    def __init__(self, server_url: str, name: str):
        self.server_url = server_url
        self.name = name
        self.gpu = get_gpu_info()
        self.device = get_device()
        self.peer_id = None
        self.ws = None
        self.running = True
        self.step = 0
        
        # Initialize model if PyTorch available
        if HAS_TORCH and self.device:
            self.model = MiniTransformer().to(self.device)
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-3)
        else:
            self.model = None
            self.optimizer = None
    
    async def connect(self):
        """Connect to the hive server"""
        print(f"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║    ██████╗ ██╗  ████████╗ ██████╗██╗  ██╗    ██████╗ ███████╗███████╗██████╗  ║
║   ██╔════╝ ██║  ╚══██╔══╝██╔════╝██║  ██║    ██╔══██╗██╔════╝██╔════╝██╔══██╗ ║
║   ██║  ███╗██║     ██║   ██║     ███████║    ██████╔╝█████╗  █████╗  ██████╔╝ ║
║   ██║   ██║██║     ██║   ██║     ██╔══██║    ██╔═══╝ ██╔══╝  ██╔══╝  ██╔══██╗ ║
║   ╚██████╔╝███████╗██║   ╚██████╗██║  ██║    ██║     ███████╗███████╗██║  ██║ ║
║    ╚═════╝ ╚══════╝╚═╝    ╚═════╝╚═╝  ╚═╝    ╚═╝     ╚══════╝╚══════╝╚═╝  ╚═╝ ║
║                                                                               ║
║   Generative Language Transformer with Contextual Hierarchy — PEER NODE      ║
║   Created by: cyberdreadx                                                     ║
╚═══════════════════════════════════════════════════════════════════════════════╝
        """)
        
        print(f"🖥️  Node: {self.name}")
        print(f"🎮 GPU: {self.gpu}")
        print(f"🔗 Server: {self.server_url}")
        print("-" * 50)
        
        try:
            self.ws = await websockets.connect(self.server_url)
            
            # Register with server
            await self.ws.send(json.dumps({
                "type": "register",
                "name": self.name,
                "gpu": self.gpu
            }))
            
            # Wait for confirmation
            response = await self.ws.recv()
            data = json.loads(response)
            
            if data["type"] == "registered":
                self.peer_id = data["peer_id"]
                self.step = data.get("training_step", 0)
                print(f"✅ Registered as: {self.peer_id}")
                print(f"📊 Current training step: {self.step}")
                print("-" * 50)
                return True
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
        
        return False
    
    async def training_loop(self):
        """Main training loop"""
        print("🏋️ Starting training...")
        
        while self.running:
            try:
                # Perform local training step
                loss = await self.train_step()
                
                # Send gradient to server
                await self.ws.send(json.dumps({
                    "type": "gradient",
                    "gradient": self.get_gradient_summary(),
                    "loss": loss
                }))
                
                # Wait for aggregated gradient from server
                try:
                    response = await asyncio.wait_for(self.ws.recv(), timeout=10.0)
                    data = json.loads(response)
                    
                    if data["type"] == "aggregated_gradient":
                        self.step = data["step"]
                        server_loss = data["loss"]
                        
                        if self.step % 50 == 0:
                            print(f"   Step {self.step:5d} | Loss: {server_loss:.4f}")
                
                except asyncio.TimeoutError:
                    # No response, continue
                    pass
                
                await asyncio.sleep(0.1)  # Rate limit
                
            except websockets.exceptions.ConnectionClosed:
                print("🔌 Disconnected from server")
                self.running = False
                break
            except Exception as e:
                print(f"⚠️  Error: {e}")
                await asyncio.sleep(1)
    
    async def train_step(self) -> float:
        """Perform one training step"""
        if self.model and self.device:
            # Real training step
            self.model.train()
            
            # Generate random batch (in real implementation, use actual data)
            batch_size = 16
            seq_len = 32
            x = torch.randint(0, 256, (batch_size, seq_len), device=self.device)
            y = torch.randint(0, 256, (batch_size, seq_len), device=self.device)
            
            # Forward pass
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
        else:
            # Simulation mode
            await asyncio.sleep(0.05)
            return 4.0 - (self.step * 0.001) + random.random() * 0.1
    
    def get_gradient_summary(self) -> dict:
        """Get summary of gradients (for transmission)"""
        if self.model:
            # Get gradient norm as summary
            total_norm = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            return {"norm": total_norm}
        else:
            return {"norm": random.random()}
    
    async def run(self):
        """Main entry point"""
        if await self.connect():
            try:
                await self.training_loop()
            except KeyboardInterrupt:
                print("\n👋 Shutting down peer...")
            finally:
                if self.ws:
                    await self.ws.close()


# ============================================
# MAIN
# ============================================

def main():
    parser = argparse.ArgumentParser(description="GLTCH Hive Training Peer")
    parser.add_argument("--server", default="ws://localhost:8765", help="Server WebSocket URL")
    parser.add_argument("--name", default=f"node-{random.randint(1000, 9999)}", help="Peer name")
    
    args = parser.parse_args()
    
    peer = TrainingPeer(args.server, args.name)
    asyncio.run(peer.run())


if __name__ == "__main__":
    main()
