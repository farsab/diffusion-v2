from pathlib import Path
from PIL import Image
import csv
from torch.utils.data import Dataset
from torchvision import transforms

class CaptionImageDataset(Dataset):
    def __init__(self, images_dir, captions_csv, image_size=512):
        self.images_dir = Path(images_dir)
        self.samples = []
        with open(captions_csv, "r", encoding="utf-8") as f:
            import csv as _csv
            reader = _csv.DictReader(f)
            for row in reader:
                self.samples.append((row["image"], row["prompt"]))

        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        fname, prompt = self.samples[idx]
        img = Image.open(self.images_dir / fname).convert("RGB")
        return {"pixel_values": self.tf(img), "prompt": prompt}
