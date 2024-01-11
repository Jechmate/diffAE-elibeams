import torch
import time
import pandas as pd
from src.diffusion import GaussianDiffusion, prepare_noise_schedule
from src.modules import UNet_conditional

# Your setup code
device = "cuda:3"
E = 25
P = 15
ms = 20

model = UNet_conditional(img_width=128, img_height=64, feat_num=3, device=device).to(device)
ckpt = torch.load("models/physinf_tenth_850ns/ema_ckpt.pt", map_location=device)
model.load_state_dict(ckpt)

# Values for noise_steps and n
noise_steps_values = [700, 850, 1000]
n_values = range(1, 21)

# Data collection
results = []

for noise_step in noise_steps_values:
    betas = prepare_noise_schedule(noise_steps=noise_step, beta_start=1e-4, beta_end=0.02)
    diffusion = GaussianDiffusion(betas=betas, img_width=128, img_height=64, device=device, noise_steps=noise_step)

    for n in n_values:
        y = torch.Tensor([E, P, ms]).to(device).float().unsqueeze(0)  # parameter vector

        # Timing the execution
        start_time = time.time()
        x = diffusion.sample_ddpm(model, n, y, cfg_scale=5, resize=[256, 512])
        end_time = time.time()

        # Calculating time per image
        time_per_image = (end_time - start_time) / n
        results.append([noise_step, n, time_per_image])

# Convert results to a DataFrame and save as CSV
df = pd.DataFrame(results, columns=['Noise Steps', 'N', 'Time per Image (seconds)'])
df.to_csv('execution_time_analysis_diffusion.csv', index=False)
print("CSV file saved: execution_time_analysis_diffusion.csv")
