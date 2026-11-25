"""Check trainable parameters breakdown."""
import sys
sys.path.insert(0, '.')

from models.mrseg import build_mrseg

cfg = {'model': {
    'convnext_path': 'pretrained_models/convnext-tiny-224',
    'bert_path': 'pretrained_models/pubmedbert'
}}

model = build_mrseg(cfg)

enc = sum(p.numel() for n, p in model.named_parameters() 
          if p.requires_grad and ('visual_enc' in n or 'text_enc' in n))
new = sum(p.numel() for n, p in model.named_parameters() 
          if p.requires_grad and ('visual_enc' not in n and 'text_enc' not in n))

print(f'Encoder LoRA: {enc:,}')
print(f'New modules: {new:,}')
print(f'Total trainable: {enc+new:,}')
print(f'Paper reports: 7,110,000')
