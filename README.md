# Diffusion Fine-Tune Pro

A modular repo for fine-tuning Stable Diffusion with **LoRA** or full adapters. Includes logging, config-driven training, and inference.


## Data
- Place images in `./data/images`
- Create `captions.csv` with format:
```
image,prompt
car1.jpg,"a photo of a red car"
car2.png,"a sports car drifting on track"
```

## Train
```bash
python train.py --config configs/lora_cars.yaml
```

Logs + samples are saved in `./outputs/...`

## Inference
```bash
python infer.py --lora_dir ./outputs/cars_lora --prompt "a futuristic concept car at night" --n 4
```

Images saved in `./samples`.

## 📊 Features
- Config-driven YAML setup
- TensorBoard + CSV logging
- Save intermediate image samples
- Easy inference script
- Ready for GitHub portfolio
