#!/usr/bin/env python3
"""
GLTCH Data Downloader
=====================
Download datasets from HuggingFace, Kaggle, or use built-in presets.
Created by: cyberdreadx

Usage:
    python gltch_data.py --preset tiny-stories
    python gltch_data.py --source huggingface --dataset "roneneldan/TinyStories"
    python gltch_data.py --url https://example.com/data.txt
"""

import argparse
import os
import sys
from pathlib import Path

# Check for required packages
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

# ═══════════════════════════════════════════════════════════════════════════════
# PRESET DATASETS
# ═══════════════════════════════════════════════════════════════════════════════

PRESETS = {
    "tiny-stories": {
        "source": "huggingface",
        "dataset": "roneneldan/TinyStories",
        "split": "train",
        "text_field": "text",
        "description": "Small stories designed for training small LMs (recommended!)",
        "size": "~500MB"
    },
    "shakespeare": {
        "source": "url",
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "description": "Complete works of Shakespeare",
        "size": "~1MB"
    },
    "wiki-simple": {
        "source": "huggingface",
        "dataset": "wikipedia",
        "config": "20220301.simple",
        "split": "train",
        "text_field": "text",
        "description": "Simple English Wikipedia",
        "size": "~200MB"
    },
    "openwebtext": {
        "source": "huggingface",
        "dataset": "Skylion007/openwebtext",
        "split": "train",
        "text_field": "text",
        "streaming": True,
        "max_samples": 100000,
        "description": "Web content (like GPT-2 training data)",
        "size": "~40GB full, we take 100k samples"
    },
    "code-python": {
        "source": "huggingface",
        "dataset": "bigcode/the-stack",
        "config": "data/python",
        "split": "train",
        "text_field": "content",
        "streaming": True,
        "max_samples": 50000,
        "description": "Python code from The Stack",
        "size": "Large, we take 50k samples"
    }
}

# ═══════════════════════════════════════════════════════════════════════════════
# DOWNLOAD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def download_from_url(url: str, output_path: Path) -> int:
    """Download text file from URL"""
    if not HAS_REQUESTS:
        print("❌ 'requests' not installed. Run: pip install requests")
        sys.exit(1)
    
    print(f"📥 Downloading from URL...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk.encode('utf-8'))
                if total_size:
                    pct = (downloaded / total_size) * 100
                    print(f"\r   Progress: {pct:.1f}%", end='', flush=True)
    
    print()
    return downloaded


def download_from_huggingface(
    dataset_name: str,
    output_path: Path,
    config: str = None,
    split: str = "train",
    text_field: str = "text",
    streaming: bool = False,
    max_samples: int = None
) -> int:
    """Download dataset from HuggingFace"""
    if not HAS_DATASETS:
        print("❌ 'datasets' not installed. Run: pip install datasets")
        sys.exit(1)
    
    print(f"📥 Loading from HuggingFace: {dataset_name}")
    print(f"   Config: {config or 'default'}, Split: {split}")
    
    # Load dataset
    kwargs = {"path": dataset_name, "split": split, "streaming": streaming}
    if config:
        kwargs["name"] = config
    
    try:
        dataset = load_dataset(**kwargs)
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        sys.exit(1)
    
    # Extract text
    print(f"📝 Extracting text from field: '{text_field}'")
    
    total_chars = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        if streaming:
            # Streaming mode for large datasets
            for i, item in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                text = item.get(text_field, "")
                if text:
                    f.write(text + "\n\n")
                    total_chars += len(text)
                if i % 1000 == 0:
                    print(f"\r   Processed: {i:,} samples ({total_chars:,} chars)", end='', flush=True)
        else:
            # Regular mode
            for i, item in enumerate(dataset):
                if max_samples and i >= max_samples:
                    break
                text = item.get(text_field, "")
                if text:
                    f.write(text + "\n\n")
                    total_chars += len(text)
                if i % 1000 == 0:
                    print(f"\r   Processed: {i:,} / {len(dataset):,} samples", end='', flush=True)
    
    print()
    return total_chars


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def list_presets():
    """Show available preset datasets"""
    print("\n📚 Available Preset Datasets:\n")
    print("-" * 70)
    for name, info in PRESETS.items():
        print(f"  --preset {name}")
        print(f"      {info['description']}")
        print(f"      Size: {info['size']}")
        print()
    print("-" * 70)
    print("\nExample: python gltch_data.py --preset tiny-stories")


def main():
    parser = argparse.ArgumentParser(
        description="🐝 GLTCH Data Downloader - Get training data from anywhere!",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gltch_data.py --preset tiny-stories
  python gltch_data.py --source huggingface --dataset "roneneldan/TinyStories"
  python gltch_data.py --url https://example.com/data.txt
  python gltch_data.py --list
        """
    )
    
    parser.add_argument("--preset", type=str, help="Use a preset dataset (recommended)")
    parser.add_argument("--source", type=str, choices=["huggingface", "kaggle"], help="Data source")
    parser.add_argument("--dataset", type=str, help="Dataset name")
    parser.add_argument("--config", type=str, help="Dataset config (for HuggingFace)")
    parser.add_argument("--url", type=str, help="Direct URL to text file")
    parser.add_argument("--output", type=str, default="data/training_data.txt", help="Output file path")
    parser.add_argument("--max-samples", type=int, help="Max samples to download (for large datasets)")
    parser.add_argument("--list", action="store_true", help="List available presets")
    
    args = parser.parse_args()
    
    # Show presets
    if args.list:
        list_presets()
        return
    
    # Validate args
    if not any([args.preset, args.source, args.url]):
        parser.print_help()
        print("\n💡 Try: python gltch_data.py --preset tiny-stories")
        return
    
    # Setup output path
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print("\n🐝 GLTCH Data Downloader")
    print("=" * 50)
    
    # Download based on source
    if args.preset:
        if args.preset not in PRESETS:
            print(f"❌ Unknown preset: {args.preset}")
            list_presets()
            return
        
        preset = PRESETS[args.preset]
        print(f"📦 Using preset: {args.preset}")
        print(f"   {preset['description']}")
        
        if preset["source"] == "url":
            total = download_from_url(preset["url"], output_path)
        elif preset["source"] == "huggingface":
            total = download_from_huggingface(
                dataset_name=preset["dataset"],
                output_path=output_path,
                config=preset.get("config"),
                split=preset.get("split", "train"),
                text_field=preset.get("text_field", "text"),
                streaming=preset.get("streaming", False),
                max_samples=preset.get("max_samples") or args.max_samples
            )
    
    elif args.url:
        total = download_from_url(args.url, output_path)
    
    elif args.source == "huggingface":
        if not args.dataset:
            print("❌ Please specify --dataset for HuggingFace source")
            return
        total = download_from_huggingface(
            dataset_name=args.dataset,
            output_path=output_path,
            config=args.config,
            max_samples=args.max_samples
        )
    
    elif args.source == "kaggle":
        print("❌ Kaggle support coming soon! Use HuggingFace for now.")
        return
    
    # Done!
    file_size = output_path.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 50)
    print(f"✅ Dataset saved to: {output_path}")
    print(f"   File size: {file_size:.2f} MB")
    print(f"\n🚀 Train with: python train_with_ui.py --data {output_path}")


if __name__ == "__main__":
    main()
