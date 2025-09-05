import yaml, torch, os
from pathlib import Path

def load_config(path):
    with open(path, "r") as f: return yaml.safe_load(f)

def save_image_grid(images, out_path, nrow=4):
    import torchvision.utils as vutils
    os.makedirs(Path(out_path).parent, exist_ok=True)
    vutils.save_image(torch.stack(images), out_path, nrow=nrow, normalize=True, value_range=(-1,1))
