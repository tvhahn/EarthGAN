# Devlog

> Thoughts on what I'm working on, struggles, and ideas. Could be an interesting retrospective, one day...
>
> Posts are in descending order (newest to oldest).

## October 15, 2021
Training is slow, but perhaps I'm seeing progress? Really, I have to ge the multi-GPU working.

Here are some samples from TensorBoard:

![sample training](./img/2021-10-15_09-29-00.png)

![sample training 2](./img/2021-10-15_09-30-48.png)

I've also been doing some research on how noise is used in StyleGAN2. Understanding this better should help me tune the model. Add it to the list.

## October 11, 2021
Over the weekend I tried training with slightly larger models. However, the GPUs always maxed out on memory (16 GB VRAM max). So, I've begun training on only one variable at a time. Let's see how this goes....

![new training only on one variable](./img/2021-10-11_08-34-17.png)

## October 8, 2021

It's been a while since I last posted! I've been on holidays and have been moving, so other things have taken some priority. However, now I've tried to get multi-GPU training going. Turns out, it's difficult to use Horovod on the HPC (some sort of dependency hell and I don't want to use Docker/Singularity yet). I'll ask some experts to see if they have any idea. Instead, I've decided to go ahead with PyTorch's [DistributedDataParallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html#comparison-between-dataparallel-and-distributeddataparallel). Wish me luck!

![dependency_hell](./img/dependency_hell.png)




## July 30, 2021

Making visualizations for the submission to the IEEE SciVis contest. So much work, so little time...

![line](./img/line.gif)

## July 19, 2021

The HPC was down for most of the weekend due to maintenance. Training is back up and running today though.

Before I begin my next iteration of model training, I need to include the discriminator loss. It would be nice to see the trend, in TenorBoard, over time.

### Results So Far

Some radial slices produce better "fake" results than others. This is the case when there aren't as many sharp features, like the results below.

![28_216](./img/28_216.png)

However, the sharp features continue to be an issue:

![28_174](./img/28_174.png)

I'll be stopping the training soon, and then implement several changes (like only training on the temperature layer and measuring the WGAN-GP loss). Each "fake" results, even from the same input, is slightly different from each other due to the stochasticity of the noise injected into the generator. This creates perceptible line in the stitched together outputs -- see image below.

![stitch_1](./img/stitch_1.png)

I'm not too concerned with this, yet. An average across several samples would likely improve things, and final implementation may even involve some sort of super-sampling layer.

### Stitching Together Results

I've started work on stitching together several "slices". This is proving to be slightly more challenging than I thought (no surprise, again). 



## July 17, 2021

Training continues... slowly. And interestingly, the "stripes" seemed to show up again (see the picture below).

![16_27](/home/tim/Documents/earth-mantle-surrogate/devlog/img/16_27.png)

The stripes  disappear after several mini-batches, as shown below in epoch 16, mini-batch 138. The stripes seem to come-and-go in an oscillating fashion.

![16_138](/home/tim/Pictures/16_138.png)

## July 16, 2021

The training has begun on the HPC! And watching TensorBoard is addictive.

First thing I noticed, which was no surprise, was how long the training takes. I'm training on all 251 time steps, and it takes 4 hours to train one epoch. Yikes!

The training is slow for a number of reasons. First, I'm training a GAN (like the [stylegan2](https://github.com/NVlabs/stylegan2), at least that's the idea). For each step where the generator is trained the discriminator takes 5 steps to train. 

Second, we're dealing with 3D data, so the models need to be larger. I'm primarily training on P100 GPUs, so about 12 GB of VRAM. The VRAM can quickly become the rate-limiting-step. 

Third, because of the 3D data and the VRAM constraint, I'm making the GAN replicate only a slice of the "Earth" at a time. Below is an image of the truth, input (input to the generator), and upsampled (concatenated onto the inputs to the discriminator). The width of the "truth" data is 38, whereas the full truth data would be 180.

![truth_input](./img/truth_input.png)

Finally, I have to train the model with only one sample in each mini-batch. Yup, it's slow.

### Results So Far

I am getting results, but I think it's too early to tell if the model will begin to produce realistic results. 

Here's an random sampling of slices from a recent mini-batch.

![13_96](./img/13_96.png)

And another one:

![13_144](./img/13_144.png)

At some point, during epoch 13, "stripes" started showing up (see the image below). I think(?) this is positive. Noise is injected into the convolutional layers of the generator. Perhaps, with time, the generator will use this additional noise, and the "stripes", to create sharper features.

![13_246](./img/13_246.png)



In the above to images, the "v" represents the variable and "r" is the radial layer (r=0 closest to core of Earth).

| Variable Index | Variable Name |
| -------------- | ------------- |
| 0              | Temperature   |
| 1              | Velocity x    |
| 2              | Velocity y    |
| 3              | Velocity z    |

So far, the model cannot generate the fine features. But again, the training is slow, and at only 13 epochs I haven't trained much.

### Next Steps

Regardless of what I do next, I think I need faster feedback on whether the model is training. I'll probably let the model train some more (sunk cost fallacy be damned! 😂) and then implement some methods to speed up training.

I can speed up training a number of ways.

1. Train on more GPUs. I could do this, but I'm concerned that if my methodology is flawed, I'll be wasting all my valuable GPU time!
2. Train on less data. Either do this by using fewer time steps (say only train on the first 50 time steps), or by using fewer variables in each sample (e.g. only train on the temperature variable).

I think I'll go for option 2 for now, and then slowly move towards scaling up the training (hello Horovod). 

### Thoughts

This is my first true foray into GANs, and GANs seem even finickier than your "normal" deep learning techniques. But like so many things in life, you learn by doing. 

![machine_learning](https://imgs.xkcd.com/comics/machine_learning.png)

Here are some of my concerns about my current approach:

* Concern that the input to the generator (making this a conditional-GAN, aka, cGAN) is not informative enough.... Do I need to make the input larger?
* I've had to tune the number of channels, in the discriminator and generator, down from what I originally wanted (from 512 channels, max, to 128 channels, max) in order to allow the model to fit into GPU memory. I'm concerned that their won't be enough "power" in the network now.
* The article by Li et al. on "[AI-assisted superresolution cosmological simulations](https://www.pnas.org/content/118/19/e2022038118)"  is a *huge* source of inspiration and info. Plus, I've used chunks of the code from Yin Li's [map2map repo](https://github.com/eelregit/map2map). In their work, they concatenate the density fields (whatever those are -- I'm no astronomer) onto the inputs to their discriminator. They say that "this addition [was] crucial to generating visually sharp images and accurate predictions of the small-scale power spectra." I'm not doing something like that, but I wonder if I need to? *But what type of data to concatenate?*
* One of my ideas was that it would be better to train with multiple variables at once, like I'm currently doing; that is, train on temperature, and the xyz velocities. Since these variables are all somewhat related, I was hoping that there would be a sort-of [multi-task learning](https://en.wikipedia.org/wiki/Multi-task_learning) benefit, whereby commonalities between the variables could be exploited. I will have to keep this in mind, but for now I think I need faster training.

That's all I have for now. Time to watch my TensorBoard...

![tensorboard_meme](./img/tensorboard_meme.jpg)



