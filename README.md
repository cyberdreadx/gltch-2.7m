# GLTCH-2.7M

```
╔═══════════════════════════════════════════════════════════════════════════════╗
║    ██████╗ ██╗  ████████╗ ██████╗██╗  ██╗     ██████╗    ███████╗███╗   ███╗  ║
║   ██╔════╝ ██║  ╚══██╔══╝██╔════╝██║  ██║     ╚════██╗   ╚════██║████╗ ████║  ║
║   ██║  ███╗██║     ██║   ██║     ███████║      █████╔╝       ██╔╝██╔████╔██║  ║
║   ██║   ██║██║     ██║   ██║     ██╔══██║     ██╔═══╝       ██╔╝ ██║╚██╔╝██║  ║
║   ╚██████╔╝███████╗██║   ╚██████╗██║  ██║     ███████╗██╗   ██║  ██║ ╚═╝ ██║  ║
║    ╚═════╝ ╚══════╝╚═╝    ╚═════╝╚═╝  ╚═╝     ╚══════╝╚═╝   ╚═╝  ╚═╝     ╚═╝  ║
╚═══════════════════════════════════════════════════════════════════════════════╝
```

**Generative Language Transformer with Contextual Hierarchy**

A 2.7 million parameter language model built from scratch, with distributed training support via the GLTCH Hive network.

## Features

- 🧠 **Complete transformer architecture** — Self-attention, multi-head attention, feedforward networks
- 📊 **2.7M parameters** — Small enough to train on free Google Colab GPUs
- 🌐 **Distributed training** — GLTCH Hive allows peers to contribute GPU power
- 🎨 **Visual dashboard** — Animated node visualization of the training network

## Quick Start

### Train Locally (Single GPU)

```bash
# Clone the repo
git clone https://github.com/cyberdreadx/gltch-2.7m.git
cd gltch-2.7m

# Install dependencies
pip install torch requests

# Train the model
python gltch_2_7m.py
```

### Train on Google Colab (Free GPU)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `gltch_2_7m_colab.py`
3. Go to **Runtime → Change runtime type → T4 GPU**
4. Run each cell in order

Training takes ~5 minutes on a T4 GPU.

## GLTCH Hive — Distributed Training

Contribute GPU power to the hive or run your own training network.

### Start the Coordinator

```bash
cd hive
pip install websockets
python server.py
```

Dashboard available at: http://localhost:8080

### Join as a Peer

```bash
python hive/peer.py --server ws://localhost:8765 --name my-node
```

## Architecture

```
GLTCH-2.7M
├── Token Embedding (65 × 192)
├── Position Embedding (128 × 192)
├── 6× Transformer Blocks
│   ├── Multi-Head Attention (6 heads)
│   ├── Layer Norm
│   ├── Feed Forward (192 → 768 → 192)
│   └── Layer Norm
├── Final Layer Norm
└── Output Head (192 → 65)
```

| Component | Size |
|-----------|------|
| Parameters | 2,708,736 |
| Context Length | 128 tokens |
| Embedding Dim | 192 |
| Attention Heads | 6 |
| Layers | 6 |

## Project Structure

```
gltch-2.7m/
├── gltch_2_7m.py          # Main model (single file)
├── gltch_2_7m_colab.py    # Colab version with cells
├── README.md
├── LICENSE
└── hive/                  # Distributed training
    ├── index.html         # Dashboard
    ├── style.css          # Dark theme
    ├── hive.js            # Node visualization
    ├── server.py          # Coordinator
    └── peer.py            # Training peer
```

## Requirements

- Python 3.8+
- PyTorch 2.0+
- websockets (for Hive only)

## License

MIT License — see [LICENSE](LICENSE)

## Author

Created by **cyberdreadx**

---

*GLTCH — Generative Language Transformer with Contextual Hierarchy*
