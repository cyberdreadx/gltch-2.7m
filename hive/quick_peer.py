# GLTCH Hive — Peer Quick Connect
# Run this on your GPU machines (Office 4090, Laptop, etc.)
#
# Usage:
#   pip install torch websockets requests
#   python quick_peer.py YOUR_VPS_IP

import sys
import os

# Add the hive path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if len(sys.argv) < 2:
    print("""
╔═══════════════════════════════════════════════════════════════════════════════╗
║   GLTCH HIVE — Quick Peer Connect                                             ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Usage:
    python quick_peer.py YOUR_VPS_IP [node_name]

Examples:
    python quick_peer.py 123.45.67.89
    python quick_peer.py 123.45.67.89 office-4090
    python quick_peer.py my-vps.tailscale.net gaming-rig
    """)
    sys.exit(1)

server_ip = sys.argv[1]
node_name = sys.argv[2] if len(sys.argv) > 2 else None

# Add ws:// if not present
if not server_ip.startswith("ws://"):
    server_ip = f"ws://{server_ip}"

# Add port if not present
if ":8765" not in server_ip:
    server_ip = f"{server_ip}:8765"

print(f"🔗 Connecting to: {server_ip}")

# Import and run peer
from peer import TrainingPeer
import asyncio

if node_name:
    peer = TrainingPeer(server_ip, node_name)
else:
    import random
    peer = TrainingPeer(server_ip, f"node-{random.randint(1000,9999)}")

asyncio.run(peer.run())
