#!/bin/bash
# GLTCH Hive Coordinator - VPS Setup Script
# Run this on your Hostinger VPS
#
# Usage: curl -sSL https://raw.githubusercontent.com/cyberdreadx/gltch-llm/main/hive/setup_coordinator.sh | bash

set -e

echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║   GLTCH HIVE — Coordinator Setup                                              ║"
echo "║   Created by: cyberdreadx                                                     ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""

# Install Python if needed
if ! command -v python3 &> /dev/null; then
    echo "📦 Installing Python..."
    apt update && apt install -y python3 python3-pip python3-venv
fi

# Create directory
echo "📁 Setting up GLTCH Hive..."
mkdir -p ~/gltch-hive
cd ~/gltch-hive

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install websockets

# Download server files
echo "📥 Downloading coordinator..."
curl -sSL https://raw.githubusercontent.com/cyberdreadx/gltch-llm/main/hive/server.py -o server.py
curl -sSL https://raw.githubusercontent.com/cyberdreadx/gltch-llm/main/hive/index.html -o index.html
curl -sSL https://raw.githubusercontent.com/cyberdreadx/gltch-llm/main/hive/style.css -o style.css
curl -sSL https://raw.githubusercontent.com/cyberdreadx/gltch-llm/main/hive/hive.js -o hive.js

# Create systemd service
echo "🔧 Creating systemd service..."
sudo tee /etc/systemd/system/gltch-hive.service > /dev/null << EOF
[Unit]
Description=GLTCH Hive Coordinator
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$HOME/gltch-hive
ExecStart=$HOME/gltch-hive/venv/bin/python server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable gltch-hive
sudo systemctl start gltch-hive

# Get server IP
IP=$(curl -s ifconfig.me)

echo ""
echo "✅ GLTCH Hive Coordinator is running!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Dashboard:    http://$IP:8080"
echo "🔌 WebSocket:    ws://$IP:8765"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "To connect a peer (on your GPU machine):"
echo "  python peer.py --server ws://$IP:8765 --name my-gpu"
echo ""
echo "Commands:"
echo "  sudo systemctl status gltch-hive   # Check status"
echo "  sudo systemctl restart gltch-hive  # Restart"
echo "  sudo journalctl -u gltch-hive -f   # View logs"
