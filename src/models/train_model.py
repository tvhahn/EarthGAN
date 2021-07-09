from math import log2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path


from src.models.utils.create_batch import EarthDataTrain
from src.models.model.model import Generator, Discriminator, init_weights

#######################################################
# Set Hyperparameters
#######################################################
device = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 2
BATCH_SIZE = 2
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10
#######################################################


root_dir = Path.cwd().parent # set the root directory as a Pathlib path
print(root_dir)

path_input_folder = root_dir / 'data/processed/input'
path_truth_folder = root_dir / 'data/processed/truth'

earth_dataset = EarthDataTrain(path_input_folder, path_truth_folder)

loader = DataLoader(
    earth_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

gen = Generator(in_chan=4, out_chan=4, scale_factor=8, chan_base=32, chan_min=32, chan_max=64, cat_noise=True).to(device)
critic = Discriminator(in_chan=8, out_chan=8, scale_factor=8, chan_base=32, chan_min=32, chan_max=64).to(device)

# initialize weights
gen.apply(init_weights)
critic.apply(init_weights)

# initializate optimizer
opt_gen = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))

gen.train()
critic.train()


for epoch in range(NUM_EPOCHS):
    print('epoch', epoch)

    for batch_idx, data in enumerate(loader):
        x_truth = data['truth'].to(device)
        x_up = data['upsampled'].to(device)

        for _ in range(CRITIC_ITERATIONS):
            fake = gen(x_input)
            
    #         loss = criterion(output, x_target)
            
            critic_real = critic(torch.cat([x_truth, x_up], dim=1)).view(-1)
            critic_fake = critic(torch.cat([fake, x_up], dim=1)).view(-1)
            
            gp = gradient_penalty(critic, 
                                torch.cat([x_truth, x_up], dim=1), # real
                                torch.cat([fake, x_up], dim=1),  # fake
                                device=device)

    #         gp = wgan_grad_penalty(critic, torch.cat([fake, x_up], dim=1), torch.cat([x_truth, x_up], dim=1))
            
            loss_critic = (
                    -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
                )
            
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()