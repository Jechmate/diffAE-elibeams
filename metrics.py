import pandas as pd
from ddpm_conditional import *
import ignite
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    args.device = "cuda:3"
    args.features = ["E", "P", "ms"]
    args.resize = (256, 512)
    args.epochs = 301
    args.noise_steps = 700
    args.beta_end = 0.02
    args.batch_size = 8
    args.image_height = 64
    args.image_width = 128
    args.features = ["E","P","ms"]
    args.dataset_path = r"with_gain"
    args.csv_path = "params.csv"
    args.lr = 1e-3
    args.grad_acc = 1
    args.sample_freq = 0
    args.sample_size = 0

    settings = pd.read_csv(args.csv_path, engine='python')[args.features]

    experiments = os.listdir(args.dataset_path)

    for experiment in sorted(experiments, key=lambda x: int(x)):
        args.exclude = [os.path.join(args.dataset_path), experiment]
        args.run_name = "no_" + experiment
        row = settings.loc[[int(experiment) - 1], args.features]
        args.sample_settings = row.values.tolist()[0]

        model = UNet_conditional(img_width=128, img_height=64, feat_num=3, device=args.device).to(args.device)
        ckpt = torch.load("models/transfered.pt", map_location=args.device)
        model.load_state_dict(ckpt)
        train(args, model)


if __name__ == "__main__":
    main()