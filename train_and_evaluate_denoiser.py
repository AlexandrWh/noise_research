from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets import DenoiseNet

from utils import val_batch_gen_denoiser, train_batch_gen_denoiser, total_mse, train


def train_denoiser():

    print('Training denoiser...')

    train_denoiser_gen = train_batch_gen_denoiser("train/noisy/*/*.npy", steps=1, batch_size=1)
    denoiser = DenoiseNet()

    for lr in [0.05, 0.01, 0.005, 0.001, 0.0001][:1]:
        optimizer = torch.optim.SGD(denoiser.parameters(), lr=lr, momentum=0.9)
        train(denoiser, 1, 1, train_denoiser_gen, optimizer, nn.MSELoss())

    torch.save(denoiser.state_dict(), 'new_denoiser.pth')

    print('\nEvaluating...')

    val_denoiser_gen = val_batch_gen_denoiser("val/*/*/*.npy", steps=1, batch_size=1)

    #print(total_mse(denoiser, val_denoiser_gen, 1, 1))

#train_denoiser()



