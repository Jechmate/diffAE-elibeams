import copy

from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR
from src.spectrum_dataset import *

from src.modules import EMA, EDMPrecond
from src.diffusion import *
from src.utils import *
from src.dataset import *
from src.loss import EDMLoss

def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)

def train(args, model=None, finetune=False):
    setup_logging(args.run_name)
    device = args.device

    data_dir = "data/spectra"  # Directory containing the spectrum CSV files
    params_file = "data/params.csv"
    
    # Create dataset
    dataset = SpectrumDataset(data_dir, params_file)
    
    # Create dataloader
    train_dataloader = get_spectrum_dataloader(data_dir, params_file, batch_size=32, features=args.features)

    gradient_acc = args.grad_acc
    l = len(train_dataloader)
    steps_per_epoch = l / gradient_acc

    #---------------------------------------------------------------------------
    if not model:
        print("Training from scratch")
        model = EDMPrecond(img_resolution   = args.length,
                            img_channels    = 1,
                            label_dim       = len(args.features),
                            use_fp16        = False,
                            sigma_min       = 0,
                            sigma_max       = float('inf'),
                            sigma_data      = 0.5,
                            model_type      = 'UNet_conditional',
                            device          = device
                            ).to(device)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    #---------------------------------------------------------------------------

    sampler = EdmSampler(net=model, num_steps=100)

    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs * steps_per_epoch)
    loss_fn = EDMLoss()

    logger = SummaryWriter(os.path.join("runs", args.run_name))

    ema = EMA(0.995)
    ema_model = copy.deepcopy(model).eval().requires_grad_(False).to(device)

    model.train().requires_grad_(True)
    # Training loop
    for epoch in range(1, args.epochs+1):
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{args.epochs}")
        # wavelengths = np.load('../data/wavelengths.npy')

        for i, data in enumerate(pbar):
            intensities = data['intensity'].to(device)
            settings = data['settings'].to(device)

            if np.random.random() < 0.1:
                settings = None
            loss = loss_fn(net=model, images=intensities, labels=settings)

            # Accumulate gradients
            optimizer.zero_grad()
            loss.mean().backward()

            # Update weights
            optimizer.step()
            scheduler.step()

            # TODO
            ema.step_ema(ema_model, model)

            pbar.set_postfix({"_Loss": "{:.4f}".format(loss.mean())})
            logger.add_scalar("Loss", loss.mean(), global_step=epoch * l + i)

        # Save samples periodically
        # if args.sample_freq and epoch % args.sample_freq == 0:
        #     settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
        #     ema_sampled_vectors = sampler.sample(
        #                                     length=args.length,
        #                                     device=device,
        #                                     class_labels=settings,
        #                                     n_samples=args.n_samples
        #                                 )
        #     # print(ema_sampled_vectors[0, :, :].shape)
        #     # print(data['energy'][0].unsqueeze(0).shape)
        #     save_images(torch.Tensor(ema_sampled_vectors[0, :, :]).to('cpu'),
        #         torch.Tensor(ema_sampled_vectors[0, :, :]).to('cpu'),
        #         torch.Tensor(args.sample_settings).to('cpu'),
        #         torch.Tensor(data['energy'][0].squeeze().to(device)).to('cpu'),
        #         os.path.join("results",
        #                      args.run_name,
        #                      f"{epoch}_final_ema.jpg"),
        #         epoch)
        #     torch.save(ema_model.state_dict(), os.path.join("models",
        #                                                     args.run_name,
        #                                                     f"ema_ckpt.pt"))
        #     torch.save(optimizer.state_dict(), os.path.join("models",
        #                                                     args.run_name,
        #                                                     f"optim.pt"))
        # Save final samples and checkpoints
        torch.save(ema_model.state_dict(), os.path.join("models",
                                                        args.run_name,
                                                        f"ema_ckpt.pt"))
        torch.save(optimizer.state_dict(), os.path.join("models",
                                                        args.run_name,
                                                        f"optim.pt"))

    # settings = torch.Tensor(args.sample_settings).to(device).unsqueeze(0)
    # ema_sampled_vectors = sampler.sample(
    #                                 length=args.length,
    #                                 device=device,
    #                                 class_labels=settings,
    #                                 n_samples=args.n_samples
    #                             )

    # save_images(torch.Tensor(ema_sampled_vectors[0, :, :]).to('cpu'),
    #             torch.Tensor(ema_sampled_vectors[0, :, :]).to('cpu'),
    #             torch.Tensor(args.sample_settings).to('cpu'),
    #             torch.Tensor(data['energy'][0].unsqueeze(0).to(device)).to('cpu'),
    #             os.path.join("results",
    #                          args.run_name,
    #                          f"{epoch}_final_ema.jpg"),
    #             epoch)


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "edm_loss_1000e_test"
    args.epochs = 1000
    #args.epochs = 3
    args.n_samples = 1
    args.batch_size = 16
    # length of the input
    args.length = 448 # original has 450 but 450/2 is 225 which is odd
    args.features = ["E","P","ms"]
    #args.length = 512
    args.device = "cuda:1"
    #args.device = "cpu"
    args.lr = 1e-3
    args.grad_acc = 1
    args.sample_freq = 10
    #args.sample_freq = 2
    #data_path = "../data/test_data_1024_[-1,1]_0.csv"
    args.data_path = "data/spectra"
    args.csv_path = "data/params.csv"
    args.sample_settings = [20,15,30]
    # args.x_train, args.y_train = get_data(data_path)

    # cond vector pro zkusebni datapoint behem prubezneho ukladani v trenovani
    # sample_spectrum_path = '../data/sample_spectrum.csv'
    # data = pd.read_csv(sample_spectrum_path)
    # args.sample_spectrum_real = np.array(data['intensities'].apply(lambda x: eval(x) if isinstance(x, str) else x).iloc[0])
    # args.sample_settings = np.array(data['cond_vector'].apply(lambda x: eval(x) if isinstance(x, str) else x).iloc[0])
    train(args, model=None)

if __name__ == '__main__':
    launch()