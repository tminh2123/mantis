"""Count trainable parameters of VoxTell model by component.

Run from the VoxTell directory after installing:
    cd VoxTell && pip install -e .
    python ../scripts/count_voxtell_params.py
"""

import sys
import os

# Allow running from workspace root or from VoxTell dir
VOXTELL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'VoxTell')
if os.path.isdir(VOXTELL_DIR):
    sys.path.insert(0, os.path.abspath(VOXTELL_DIR))

import pydoc
import torch
from voxtell.utils.download import download_voxtell_model
from voxtell.model.voxtell_model import VoxTellModel
from nnunetv2.utilities.json_export import recursive_fix_for_json_export
from batchgenerators.utilities.file_and_folder_operations import load_json, join


def count_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def fmt(n: int) -> str:
    if n >= 1e6:
        return f"{n/1e6:.2f}M"
    if n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def main():
    print("Downloading VoxTell model (cached after first run)...")
    model_dir = download_voxtell_model()

    plans = load_json(join(model_dir, 'plans.json'))
    arch_kwargs = dict(**plans['configurations']['3d_fullres']['architecture']['arch_kwargs'])
    for key in plans['configurations']['3d_fullres']['architecture']['_kw_requires_import']:
        if arch_kwargs[key] is not None:
            arch_kwargs[key] = pydoc.locate(arch_kwargs[key])

    print("Instantiating model...")
    model = VoxTellModel(
        input_channels=1,
        **arch_kwargs,
        decoder_layer=4,
        text_embedding_dim=2560,
        num_maskformer_stages=5,
        num_heads=32,
        query_dim=2048,
        project_to_decoder_hidden_dim=2048,
        deep_supervision=False,
    )

    # --- Per-component breakdown ---
    components = {
        "encoder":              model.encoder,
        "decoder":              model.decoder,
        "transformer_decoder":  model.transformer_decoder,
    }

    total = count_params(model)
    print(f"\n{'Component':<30} {'Trainable params':>18}")
    print("-" * 50)
    for name, mod in components.items():
        n = count_params(mod)
        print(f"{name:<30} {fmt(n):>18}  ({100*n/total:.1f}%)")

    # Catch any params not under the 3 main components
    accounted = sum(count_params(m) for m in components.values())
    other = total - accounted
    if other > 0:
        print(f"{'other':<30} {fmt(other):>18}  ({100*other/total:.1f}%)")

    print("-" * 50)
    print(f"{'TOTAL':<30} {fmt(total):>18}")


if __name__ == "__main__":
    main()
