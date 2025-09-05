import argparse, torch, os
from src.dataset import CaptionImageDataset
from src.trainer import DiffusionTrainer
from src.utils import load_config

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    os.makedirs(cfg["output_dir"], exist_ok=True)

    dataset = CaptionImageDataset(cfg["data"]["images_dir"], cfg["data"]["captions_csv"], image_size=cfg["train"]["image_size"])
    trainer = DiffusionTrainer(cfg, dataset, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    trainer.train()

if __name__ == "__main__":
    main()
