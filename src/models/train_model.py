from math import log2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter  # print to tensorboard
from torch.utils.data import DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import datetime
import re
import argparse


from src.models.utils.create_batch import EarthDataTrain
from src.models.model.model import Generator, Discriminator, init_weights
from src.models.loss.wasserstein import gradient_penalty

#######################################################
# Set Hyperparameters
#######################################################

device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100
BATCH_SIZE = 1
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
GEN_PRETRAIN_EPOCHS = 5  # number of epochs to pretrain generator


########################################################
# Helper functions
#######################################################


def plot_fake_truth(fake, x_truth, epoch_i, batch_i):
    plt.switch_backend("agg")
    with torch.no_grad():
        fake = gen(x_input).cpu()
        x_truth = x_truth.cpu()
        color_scheme = "inferno"
        no_col = 10
        fig, ax = plt.subplots(2, no_col, figsize=(12, 6))
        b, v, r, h, w = fake.shape
        for i in range(no_col):
            bi = torch.randint(b, (1,)).item()
            vi = torch.randint(v, (1,)).item()
            ri = torch.randint(r, (1,)).item()
            ax[0, i].pcolormesh(fake[bi, vi, ri, :, :].cpu(), cmap=color_scheme)
            ax[0, i].get_xaxis().set_visible(False)
            ax[0, i].get_yaxis().set_visible(False)
            ax[0, i].set_title(f"v={vi}, r={ri}", fontsize=10)
            ax[1, i].pcolormesh(x_truth[bi, vi, ri, :, :].cpu(), cmap=color_scheme)
            ax[1, i].get_xaxis().set_visible(False)
            ax[1, i].get_yaxis().set_visible(False)
        plt.suptitle(f'epoch {epoch_i}, batch_idx {batch_i}')
        plt.subplots_adjust(wspace=0, hspace=0)

    return fig


def find_most_recent_checkpoint(path_prev_checkpoint):
    """Finds the most recent checkpoint in a checkpoint folder
    and returns the path to that .pt file.
    """

    ckpt_list = list(path_prev_checkpoint.rglob("*.pt"))
    max_epoch = sorted(list(int(re.findall("[0-9]+", str(i))[-1]) for i in ckpt_list))[
        -1
    ]
    return Path(path_prev_checkpoint / f"train_{max_epoch}.pt")


#######################################################


#######################################################
# Set Directories
#######################################################

# check if "scratch" path exists in the home directory
# if it does, assume we are on HPC
scratch_path = Path.home() / "scratch"
if scratch_path.exists():
    print("Assume on HPC")
else:
    print("Assume on local compute")

# parse arguments
parser = argparse.ArgumentParser()

parser.add_argument(dest="path_data", type=str, help="Path to processed data")

parser.add_argument(
    "-c",
    "--checkpoint",
    dest="ckpt_name",
    type=str,
    help="Name of chekpoint folder to load previous checkpoint from",
)
args = parser.parse_args()

path_processed_data = Path(args.path_data)

# if loading the model from a checkpoint, a checkpoint folder name
# should be passed as an argument, like: -c 2021_07_14_185903
# the various .pt files will be inside the checkpoint folder
if args.ckpt_name:
    prev_checkpoint_folder_name = args.ckpt_name
    print("argparse works!", prev_checkpoint_folder_name)
else:
    # set dummy name for path_prev_checkpoint
    path_prev_checkpoint = Path('no_prev_checkpoint_needed')


# set time
model_start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S")

if scratch_path.exists():
    # for HPC
    root_dir = scratch_path / "earth-mantle-surrogate"
    print(root_dir)

    if args.ckpt_name:
        path_prev_checkpoint = (
            root_dir / "models/interim/checkpoints" / prev_checkpoint_folder_name
        )
        if Path(path_prev_checkpoint).exists():
            print("Previous checkpoints exist. Training from most recent checkpoint.")

            path_prev_checkpoint = find_most_recent_checkpoint(path_prev_checkpoint)

        else:
            print("Could not find previous checkpoint folder. Training from beginning.")

    path_input_folder = path_processed_data / "input"
    path_truth_folder = path_processed_data / "truth"
    path_checkpoint_folder = root_dir / "models/interim/checkpoints" / model_start_time
    Path(path_checkpoint_folder).mkdir(parents=True, exist_ok=True)

else:

    # for local compute
    root_dir = Path.cwd()  # set the root directory as a Pathlib path
    print(root_dir)

    if args.ckpt_name:
        path_prev_checkpoint = (
            root_dir / "models/interim/checkpoints" / prev_checkpoint_folder_name
        )
        if Path(path_prev_checkpoint).exists():
            print("Previous checkpoints exist. Training from most recent checkpoint.")

            path_prev_checkpoint = find_most_recent_checkpoint(path_prev_checkpoint)

        else:
            print("Could not find previous checkpoint folder. Training from beginning.")

    path_input_folder = path_processed_data / "input"
    path_truth_folder = path_processed_data / "truth"
    path_checkpoint_folder = root_dir / "models/interim/checkpoints" / model_start_time
    Path(path_checkpoint_folder).mkdir(parents=True, exist_ok=True)


#######################################################
# Prep Model and Data
#######################################################

earth_dataset = EarthDataTrain(path_input_folder, path_truth_folder)

loader = DataLoader(
    earth_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# set summary writer for Tensorboard
writer_results = SummaryWriter(root_dir / "models/interim/logs/" / model_start_time)

gen = Generator(
    in_chan=4,
    out_chan=4,
    scale_factor=8,
    chan_base=64,
    chan_min=64,
    chan_max=128,
    cat_noise=True,
).to(device)

critic = Discriminator(
    in_chan=8, out_chan=8, scale_factor=8, chan_base=64, chan_min=64, chan_max=128
).to(device)

# initialize weights
gen.apply(init_weights)
critic.apply(init_weights)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))

# load from checkpoint if wanted
if path_prev_checkpoint.exists():
    print('Loading from previous checkpoint')
    checkpoint = torch.load(path_prev_checkpoint)
    epoch_start = checkpoint['epoch']+1
    gen.load_state_dict(checkpoint['gen'])
    critic.load_state_dict(checkpoint['critic'])
    opt_gen.load_state_dict(checkpoint['opt_gen'])
    opt_critic.load_state_dict(checkpoint['opt_critic'])

else:
    epoch_start = 0

#######################################################
# Training Loop
#######################################################

step = 0
for epoch in range(epoch_start, epoch_start+ NUM_EPOCHS):
    gen.train()
    critic.train()
    print("epoch", epoch)

    for batch_idx, data in enumerate(loader):
        x_truth = data["truth"].to(device)
        x_up = data["upsampled"].to(device)
        x_input = data["input"].to(device)

        # pre-train the generator with simple MSE loss
        if epoch < GEN_PRETRAIN_EPOCHS:
            criterion = nn.MSELoss()
            gen_fake = gen(x_input)
            loss_mse = criterion(gen_fake, x_truth)
            gen.zero_grad()
            loss_mse.backward()
            opt_gen.step()

        # after pre-training of generator, enter the
        # full training loop and train critic (e.g. discriminator) too
        else:
            # train critic
            for _ in range(CRITIC_ITERATIONS):
                fake = gen(x_input)

                critic_real = critic(torch.cat([x_truth, x_up], dim=1)).view(-1)
                critic_fake = critic(torch.cat([fake, x_up], dim=1)).view(-1)

                gp = gradient_penalty(
                    critic,
                    torch.cat([x_truth, x_up], dim=1),  # real
                    torch.cat([fake, x_up], dim=1),  # fake
                    device=device,
                )

                loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake))
                    + LAMBDA_GP * gp
                )

                critic.zero_grad()
                loss_critic.backward(retain_graph=True)
                opt_critic.step()

            # train generator after every N critic iterations
            gen_fake = critic(torch.cat([fake, x_up], dim=1)).reshape(-1)
            loss_gen = -torch.mean(gen_fake)
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
        if epoch > GEN_PRETRAIN_EPOCHS:
            if batch_idx % 3 == 0:

                with torch.no_grad():
                    gen.eval() # does this need to be included???
                    fake = gen(x_input)
                    fig = plot_fake_truth(fake, x_truth, epoch, batch_idx)
                    writer_results.add_figure("Results", fig, global_step=step)

                step += 1

                # save checkpoint
                torch.save(
                    {
                        "gen": gen.state_dict(),
                        "critic": critic.state_dict(),
                        "opt_gen": opt_gen.state_dict(),
                        "opt_critic": opt_critic.state_dict(),
                        "epoch": epoch,
                    },
                    path_checkpoint_folder / f"train_{epoch}.pt",
                )
        else:
            if batch_idx % 10 == 0:

                with torch.no_grad():
                    gen.eval() # does this need to be included???
                    fake = gen(x_input)
                    fig = plot_fake_truth(fake, x_truth, epoch, batch_idx)
                    writer_results.add_figure("Results", fig, global_step=step)

                step += 1

                # save checkpoint
                torch.save(
                    {
                        "gen": gen.state_dict(),
                        "critic": critic.state_dict(),
                        "opt_gen": opt_gen.state_dict(),
                        "opt_critic": opt_critic.state_dict(),
                        "epoch": epoch,
                    },
                    path_checkpoint_folder / f"train_{epoch}.pt",
                )

