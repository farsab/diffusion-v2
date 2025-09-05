import argparse, torch, os
from diffusers import StableDiffusionPipeline

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lora_dir", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--outdir", default="./samples")
    p.add_argument("--n", type=int, default=4)
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
    pipe.unet.load_attn_procs(args.lora_dir)

    for i in range(args.n):
        img = pipe(args.prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
        img.save(os.path.join(args.outdir, f"gen_{i}.png"))

if __name__ == "__main__":
    main()
