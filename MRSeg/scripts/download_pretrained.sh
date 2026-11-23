#!/bin/bash
# Download pretrained backbones for MRSeg
set -e

cd "$(dirname "$0")/.."
mkdir -p pretrained_models

echo "=== Downloading ConvNeXt-Tiny (facebook/convnext-tiny-224) ==="
huggingface-cli download facebook/convnext-tiny-224 \
    --local-dir pretrained_models/convnext-tiny-224

echo ""
echo "=== Downloading PubMedBERT (microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext) ==="
huggingface-cli download microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
    --local-dir pretrained_models/pubmedbert

echo ""
echo "=== Done. Models saved to pretrained_models/ ==="
echo "  pretrained_models/convnext-tiny-224/"
echo "  pretrained_models/pubmedbert/"
