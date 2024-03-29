from math import log2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.tensorboard import SummaryWriter  # print to tensorboard
from torch.utils.data import DataLoader
import torch.utils.data.distributed
from pathlib import Path
import matplotlib.pyplot as plt
import datetime
import re
import argparse
import shutil
import logging


from src.models.utils.create_batch import EarthDataTrain
from src.models.model.model import Generator, Discriminator, init_weights
from src.models.loss.wasserstein import gradient_penalty

#######################################################
# Argparse
#######################################################

parser = argparse.ArgumentParser()

parser.add_argument(
    "--path_data", dest="path_data", type=str, help="Path to processed data"
)

parser.add_argument(
    "-c",
    "--checkpoint",
    dest="ckpt_name",
    type=str,
    help="Name of chekpoint folder to load previous checkpoint from",
)

parser.add_argument(
    "--model_time_suffix",
    dest="model_time_suffix",
    type=str,
    help="Optional suffix string to append at the end of the model start time identifier",
)

parser.add_argument(
    "-p",
    "--proj_dir",
    dest="proj_dir",
    type=str,
    help="Location of project folder",
)


parser.add_argument(
    "--var_to_include",
    dest="var_to_include",
    type=int,
    default=1,
    help="Number of variables to be trained on. var_to_include=1 \
        is only the temperature data. \
        var_to_include=4 is the temperature, vx, vy, and vz.",
)

parser.add_argument(
    "--batch_size",
    dest="batch_size",
    type=int,
    default=1,
    help="Mini-batch size for each GPU",
)

parser.add_argument(
    "--cat_noise",
    action="store_true",
    help="Will concatenate noise if argument used (sets cat_noise=True).",
)

parser.add_argument(
    "--learning_rate",
    dest="learning_rate",
    type=float,
    default=1e-4,
    help="Learning rate for optimizer",
)

parser.add_argument(
    "--critic_iterations",
    dest="critic_iterations",
    type=int,
    default=5,
    help="Number of critic iterations for every 1 generator iteration",
)

parser.add_argument(
    "--num_epochs",
    dest="num_epochs",
    type=int,
    default=500,
    help="Number of epochs",
)

parser.add_argument(
    "--lambda_gp",
    dest="lambda_gp",
    type=int,
    default=10,
    help="Lambda modifier for gradient penalty",
)

parser.add_argument(
    "--gen_pretrain_epochs",
    dest="gen_pretrain_epochs",
    type=int,
    default=5,
    help="Epochs to train generator alone at the beginning",
)

args = parser.parse_args()

# hyperparameters
LEARNING_RATE = args.learning_rate
NUM_EPOCHS = args.num_epochs
BATCH_SIZE = args.batch_size
CRITIC_ITERATIONS = args.critic_iterations
LAMBDA_GP = args.lambda_gp
GEN_PRETRAIN_EPOCHS = args.gen_pretrain_epochs
VAR_TO_INCLUDE = args.var_to_include


########################################################
# Functions
#######################################################


def plot_fake_truth(fake, x_truth, x_up, epoch_i, batch_i, time_i):
    """Create the ground-truth, upscaled, and fake images used
    in Tensorboard"""

    plt.switch_backend("agg")  # needed for HPC

    with torch.no_grad():
        fake = fake.cpu()
        x_truth = x_truth.cpu()
        color_scheme = "inferno"
        no_col = 10
        fig, ax = plt.subplots(3, no_col, figsize=(12, 9))
        b, v, r, h, w = fake.shape
        for i in range(no_col):
            bi = torch.randint(b, (1,)).item()
            vi = torch.randint(v, (1,)).item()
            ri = torch.randint(r, (1,)).item()
            ax[0, i].pcolormesh(fake[bi, vi, ri, :, :].cpu(), cmap=color_scheme)
            ax[0, i].get_xaxis().set_visible(False)
            ax[0, i].get_yaxis().set_visible(False)
            ax[0, i].set_title(f"v={vi}, r={ri}", fontsize=10)
            ax[1, i].pcolormesh(x_up[bi, vi, ri, :, :].cpu(), cmap=color_scheme)
            ax[1, i].get_xaxis().set_visible(False)
            ax[1, i].get_yaxis().set_visible(False)
            ax[2, i].pcolormesh(x_truth[bi, vi, ri, :, :].cpu(), cmap=color_scheme)
            ax[2, i].get_xaxis().set_visible(False)
            ax[2, i].get_yaxis().set_visible(False)
        plt.suptitle(f"Epoch {epoch_i}, Batch Index {batch_i}, Time Step {time_i[bi]}")
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


def set_directories():
    """Sets the directory paths used for data, checkpoints, etc."""

    # check if "scratch" path exists in the home directory
    # if it does, assume we are on HPC
    scratch_path = Path.home() / "scratch"
    if scratch_path.exists():
        print("Assume on HPC")
    else:
        print("Assume on local compute")

    path_processed_data = Path(args.path_data)

    # if loading the model from a checkpoint, a checkpoint folder name
    # should be passed as an argument, like: -c 2021_07_14_185903
    # the various .pt files will be inside the checkpoint folder
    if args.ckpt_name:
        prev_checkpoint_folder_name = args.ckpt_name
    else:
        # set dummy name for path_prev_checkpoint
        path_prev_checkpoint = Path("no_prev_checkpoint_needed")

    if args.proj_dir:
        proj_dir = Path(args.proj_dir)
    else:
        # proj_dir assumed to be cwd
        proj_dir = Path.cwd()

    # set time
    if args.model_time_suffix:
        model_start_time = datetime.datetime.now().strftime("%Y_%m_%d_%H%M%S") + "_" + args.model_time_suffix
    else:
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
                print(
                    "Previous checkpoints exist. Training from most recent checkpoint."
                )

                path_prev_checkpoint = find_most_recent_checkpoint(path_prev_checkpoint)

            else:
                print(
                    "Could not find previous checkpoint folder. Training from beginning."
                )

        path_input_folder = path_processed_data / "input"
        path_truth_folder = path_processed_data / "truth"
        path_checkpoint_folder = (
            root_dir / "models/interim/checkpoints" / model_start_time
        )
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
                print(
                    "Previous checkpoints exist. Training from most recent checkpoint."
                )

                path_prev_checkpoint = find_most_recent_checkpoint(path_prev_checkpoint)

            else:
                print(
                    "Could not find previous checkpoint folder. Training from beginning."
                )

        path_input_folder = path_processed_data / "input"
        path_truth_folder = path_processed_data / "truth"
        path_checkpoint_folder = (
            root_dir / "models/interim/checkpoints" / model_start_time
        )
        Path(path_checkpoint_folder).mkdir(parents=True, exist_ok=True)

    # save src directory as a zip into the checkpoint folder
    shutil.make_archive(
        path_checkpoint_folder / f"src_files_{model_start_time}",
        "zip",
        proj_dir / "src",
    )
    shutil.copy(
        proj_dir / "bash_scripts/train_model_hpc.sh",
        path_checkpoint_folder / "train_model_hpc.sh",
    )

    return (
        root_dir,
        path_input_folder,
        path_truth_folder,
        path_checkpoint_folder,
        path_prev_checkpoint,
        model_start_time,
    )


def save_checkpoint(epoch, path_checkpoint_folder, gen, critic, opt_gen, opt_critic):
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


def create_tensorboard_fig(
    gen, x_input, x_truth, x_up, epoch, batch_idx, time_i, step, writer_results
):
    with torch.no_grad():
        gen.eval()  # does this need to be included???
        fake = gen(x_input)
        fig = plot_fake_truth(fake, x_truth, x_up, epoch, batch_idx, time_i)
        writer_results.add_figure("Results", fig, global_step=step)


def main():
    """Establish models and run training loop"""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = EarthDataTrain(
        path_input_folder, path_truth_folder, var_to_include=VAR_TO_INCLUDE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    gen = Generator(
        in_chan=VAR_TO_INCLUDE,
        out_chan=VAR_TO_INCLUDE,
        scale_factor=8,
        chan_base=128,
        chan_min=64,
        chan_max=512,
        cat_noise=args.cat_noise,
    ).to(device)

    critic = Discriminator(
        in_chan=VAR_TO_INCLUDE*2, out_chan=VAR_TO_INCLUDE*2, scale_factor=8, chan_base=512, chan_min=64, chan_max=512
    ).to(device)

    # initialize weights
    gen.apply(init_weights)
    critic.apply(init_weights)

    # initializate optimizer
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))
    opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.9))

    train(
        gen,
        critic,
        opt_gen,
        opt_critic,
        device,
        train_loader,
    )


def train(
    gen,
    critic,
    opt_gen,
    opt_critic,
    device,
    train_loader,
):
    """Training scrip"""

    # set summary writer for Tensorboard
    writer_results = SummaryWriter(root_dir / "models/interim/logs/" / model_start_time)

    # load from checkpoint if wanted
    if path_prev_checkpoint.exists():
        print("Loading from previous checkpoint")
        checkpoint = torch.load(path_prev_checkpoint)
        epoch_start = checkpoint["epoch"] + 1
        gen.load_state_dict(checkpoint["gen"])
        critic.load_state_dict(checkpoint["critic"])
        opt_gen.load_state_dict(checkpoint["opt_gen"])
        opt_critic.load_state_dict(checkpoint["opt_critic"])

    else:
        epoch_start = 0

    step = 0
    for epoch in range(epoch_start, epoch_start + NUM_EPOCHS):
        gen.train()
        critic.train()
        print("epoch", epoch)

        for batch_idx, data in enumerate(train_loader):
            x_truth = data["truth"].to(device)
            x_up = data["upsampled"].to(device)
            x_input = data["input"].to(device)
            time_i = data["time_step_index"]

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
                if batch_idx % 10 == 0:

                    create_tensorboard_fig(
                        gen,
                        x_input,
                        x_truth,
                        x_up,
                        epoch,
                        batch_idx,
                        time_i,
                        step,
                        writer_results,
                    )
                    save_checkpoint(
                        epoch, path_checkpoint_folder, gen, critic, opt_gen, opt_critic
                    )
                    step += 1
            else:
                if batch_idx % 10 == 0:

                    create_tensorboard_fig(
                        gen,
                        x_input,
                        x_truth,
                        x_up,
                        epoch,
                        batch_idx,
                        time_i,
                        step,
                        writer_results,
                    )
                    save_checkpoint(
                        epoch, path_checkpoint_folder, gen, critic, opt_gen, opt_critic
                    )
                    step += 1

        # save checkpoint at end of epoch
        save_checkpoint(epoch, path_checkpoint_folder, gen, critic, opt_gen, opt_critic)


if __name__ == "__main__":

    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    (
        root_dir,
        path_input_folder,
        path_truth_folder,
        path_checkpoint_folder,
        path_prev_checkpoint,
        model_start_time,
    ) = set_directories()

    main()
