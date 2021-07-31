import torch
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from src.models.model.model import Generator
import logging
import warnings
warnings.filterwarnings("ignore")


def gen_output_slice_for_concatenate(x_input, gen, roll_increment=0, width=32):
    """
    Args:
        x_input (tensor): Input (downsampled) sample.
        gen (pytorch model): Loaded pytorch model
        roll_increment (int): Amount to roll the input by. Standard x_input has width of 27
    """
    x_input = torch.roll(x_input, roll_increment, 4)  # roll if used
    x_input = x_input[:, :, :, :, :10]  # select first 10 points for width
    x_input = pad_data(x_input, pad_top_bot=3, pad_sides=0)  # pad top/bottom

    with torch.no_grad():
        gen.eval()
        x_fake = gen(x_input)

        # x_fake shape: (1, 4, 198, 118, 38)
        return x_fake[:, :, :, :, 3 : width + 3].detach().cpu()


def pad_data(x, pad_top_bot=0, pad_sides=0):
    "pad top/bot or sides of tensor"
    if pad_sides > 0:
        x = torch.cat(
            (x[:, :, :, :, -pad_sides:], x, x[:, :, :, :, :pad_sides]), axis=-1
        )

    if pad_top_bot > 0:
        x = torch.cat(
            (
                torch.flip(x, [3])[
                    :, :, :, -pad_top_bot:, :
                ],  # mirror array and select top rows
                x,
                torch.flip(x, [3])[:, :, :, :pad_top_bot, :],
            ),  # mirror array and select bottom rows
            axis=-2,
        )  # append along longitudinal (left-right) axis
    return x


def crop_data(
    x,
    crop_height=1,
    crop_width=1,
):
    "symetrically crop tensor"
    w = crop_width
    h = crop_height

    if crop_width == 0 and crop_height == 0:
        return x
    elif crop_width == 0:
        return x[:, :, :, h:-h, :]
    elif crop_height == 0:
        return x[:, :, :, :, w:-w]
    else:
        return x[:, :, :, h:-h, w:-w]


def create_combined_map(
    x_input, gen, device, r_index, v, path_save_name, dpi=150, final_roll=0, crop=True, save_fig=False
):
    fake_dict = {}
    for i, roll_n in enumerate(range(0, 32, 4)):
        fake = gen_output_slice_for_concatenate(
            x_input.to(device), gen, roll_increment=roll_n, width=32
        )
        fake_dict[i] = fake

    for i, k in enumerate(fake_dict):
        print(i)
        if i == 0:
            a = fake_dict[i]
        else:
            a = torch.cat((fake_dict[i], a), axis=-1)

    print(a.shape)

    if crop:
        a = crop_data(
            a,
            crop_height=5,
            crop_width=20,
        )
    print(a.shape)

    color_scheme = "inferno"
    fig, ax = plt.subplots(1, 1, figsize=(18, 8))
    a = torch.roll(a, final_roll, 4)
    ax.pcolormesh(a[0, v, r_index, :, :], cmap=color_scheme)
    ax.set_aspect(1)
    if save_fig:
        plt.savefig(path_save_name, dpi=dpi, bbox_inches="tight")
    plt.cla()
    plt.close()
    return a


def create_truth_map(
    truth_data, fake, r_index, v,  path_save_name1, path_save_name2, dpi=150, final_roll=0
):
    color_scheme = "inferno"
    fig, ax = plt.subplots(2, 1, figsize=(18, 16))

    ax[0].set_title("Ground-Truth", fontsize=16)
    ax[0].pcolormesh(torch.roll(truth_data[0, v, r_index, :, :],final_roll, 1), cmap=color_scheme)
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    ax[0].set_aspect(1)

    ax[1].set_title("Fake", fontsize=16)
    ax[1].pcolormesh(fake[0, v, r_index, :, :], cmap=color_scheme)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    ax[1].set_aspect(1)

    plt.savefig(path_save_name1, dpi=dpi, bbox_inches="tight")
    plt.cla()
    plt.close()

    # create mollweide representation
    fig, ax = plt.subplots(1, 2, figsize=(18, 16))
    ax[0] = plt.subplot(121, projection="mollweide")
    ax[1] = plt.subplot(122, projection="mollweide")
    # ax[1] = plt.subplot(111, projection="mollweide")

    lon = np.linspace(-np.pi, np.pi, 216)
    lat = np.linspace(-np.pi / 2.0, np.pi / 2.0, 108)
    Lon, Lat = np.meshgrid(lon, lat)

    ax[0].set_title("Ground-Truth", fontsize=16)
    ax[0].pcolormesh(Lon, Lat, torch.roll(truth_data[0, v, r_index, :, :],final_roll, 1), cmap=color_scheme)
    ax[0].get_xaxis().set_visible(False)
    ax[0].get_yaxis().set_visible(False)
    # ax[0].set_aspect(1)

    ax[1].set_title("Fake", fontsize=16)
    ax[1].pcolormesh(Lon, Lat, fake[0, 0, r_index, :, :], cmap=color_scheme)
    ax[1].get_xaxis().set_visible(False)
    ax[1].get_yaxis().set_visible(False)
    # ax[1].set_aspect(1)

    plt.savefig(path_save_name2, dpi=dpi, bbox_inches="tight")
    plt.cla()
    plt.close()


def main():
    logger = logging.getLogger(__name__)
    logger.info("making figures from results")

    path_results = root_dir / "models/interim/"
    path_save_loc = root_dir / "reports/figures/"

    path_processed_data = root_dir / "data/processed"
    path_input_folder = path_processed_data / "input"
    path_truth_folder = path_processed_data / "truth"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)

    # iterate through checkpoint folders and get the paths for each epoch's serialized model
    root_checkpoint_dir = root_dir / "models/interim/checkpoints_2021_07_21_152707"
    checkpoint_file_list = list(
        root_checkpoint_dir.rglob("*.pt")
    )  # get list of all .pt file paths

    # dict containing each epoch (as dict key) and associated .pt file path
    checkpoint_file_dict = {
        int(str(i).split(".")[-2].split("_")[-1]): i for i in checkpoint_file_list
    }

    gen = Generator(
        in_chan=4,
        out_chan=4,
        scale_factor=8,
        chan_base=64,
        chan_min=64,
        chan_max=128,
        cat_noise=True,
    ).to(device)

    EPOCH_LOAD = 36
    R_INDEX = 165
    VARIABLE_INDEX = 0
    ITERATE_EPOCHS = True

    checkpoint = torch.load(checkpoint_file_dict[EPOCH_LOAD])
    gen.load_state_dict(checkpoint["gen"])

    # save gen checkpoint to check size
    # torch.save(
    #     {
    #         "gen": gen.state_dict(),
    #     },
    #     path_save_loc / "gen.pt",
    # )

    input_data = torch.tensor(np.load(path_input_folder / "x_001.npy"))
    index_keep = np.round(
        np.arange(0, input_data.shape[2], input_data.shape[2] / 30.0)
    ).astype(int)
    input_data = input_data[:, :, index_keep, :, :]

    truth_data = torch.tensor(np.load(path_truth_folder / "x_truth_001.npy"))
    print("shape truth:", truth_data.shape)

    x_comb = create_combined_map(
        input_data,
        gen,
        device,
        R_INDEX, VARIABLE_INDEX,
        path_save_name=path_save_loc / "combined.png",
        dpi=150,
        final_roll=0,
    )



    create_truth_map(
        truth_data,
        x_comb,
        R_INDEX, VARIABLE_INDEX, 
        path_save_name1=path_save_loc / f"compare_epoch{EPOCH_LOAD}_rindex{R_INDEX}.png",
        path_save_name2=path_save_loc / f"compare_epoch{EPOCH_LOAD}_rindex{R_INDEX}_moll.png",
        dpi=150, final_roll=-35
    )

    if ITERATE_EPOCHS:
        Path(path_save_loc / f"epoch_rindex{R_INDEX}").mkdir(parents=True, exist_ok=True)
        Path(path_save_loc / f"epoch_rindex{R_INDEX}" / "moll").mkdir(parents=True, exist_ok=True)
        path_epoch_iter = Path(path_save_loc / f"epoch_rindex{R_INDEX}")

        for epoch in range(0,46):
            checkpoint = torch.load(checkpoint_file_dict[epoch])
            gen.load_state_dict(checkpoint["gen"])

            x_comb = create_combined_map(
                input_data,
                gen,
                device,
                R_INDEX, VARIABLE_INDEX,
                path_save_name=path_epoch_iter / "combined.png",
                dpi=150,
                final_roll=0,
            )


            create_truth_map(
                truth_data,
                x_comb,
                R_INDEX, VARIABLE_INDEX, 
                path_save_name1=path_epoch_iter / f"compare_epoch{epoch}_rindex{R_INDEX}.png",
                path_save_name2=path_epoch_iter / "moll" / f"compare_epoch{epoch}_rindex{R_INDEX}_moll.png",
                dpi=150, final_roll=-35
            )




if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    root_dir = Path(__file__).resolve().parents[2]

    main()
