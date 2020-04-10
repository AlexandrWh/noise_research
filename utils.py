import numpy as np
from glob import glob
import cv2
from tqdm import tqdm

from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F


def train_batch_gen_denoiser(folder_path, steps, batch_size):
    step = 0
    train_files = list(glob(folder_path))
    np.random.shuffle(train_files)
    size = 80

    while True:
        X_train, y_train = [], []

        for train_file in train_files[step * batch_size:(step + 1) * batch_size]:
            x_mel = np.load(train_file).astype(np.float32).T
            y_mel = np.load(train_file.replace('noisy', 'clean')).astype(np.float32).T
            # idx = np.random.randint(x_mel.shape[1] - size - 1)

            # x_mel = x_mel[:, idx:idx+size]
            # y_mel = y_mel[:, idx:idx+size]
            x_mel = x_mel.reshape((1, *x_mel.shape))
            y_mel = y_mel.reshape((1, *y_mel.shape))

            X_train.append(x_mel)
            # X_train.append(y_mel)
            y_train.append(y_mel)
            # y_train.append(y_mel)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        X_batch = Variable(torch.FloatTensor(X_train))
        y_batch = Variable(torch.FloatTensor(y_train))

        step += 1
        if step == steps:
            step = 0
        yield X_batch, y_batch


def val_batch_gen_denoiser(folder_path, steps, batch_size):
    step = 0
    val_files = list(glob(folder_path))

    while True:
        X_val, y_val = [], []

        for val_file in val_files[step * batch_size:(step + 1) * batch_size]:
            x_mel = np.load(val_file).astype(np.float32).T
            x_mel = x_mel.reshape((1, *x_mel.shape))
            X_val.append(x_mel)

            y_mel = np.load(val_file.replace('noisy', 'clean')).astype(np.float32).T
            y_mel = y_mel.reshape((1, *y_mel.shape))
            y_val.append(y_mel)

        X_val = np.array(X_val)
        y_val = np.array(y_val)

        step += 1
        if step == steps:
            step = 0
        yield X_val, y_val


def total_mse(net, batch_gen, batch_size, val_size):
    steps = val_size // batch_size
    total_loss = 0
    for batch_i in tqdm(range(steps)):
        X_batch, y_batch = next(batch_gen)
        predict = net(Variable(torch.FloatTensor(X_batch))).detach().cpu().numpy()
        total_loss += np.mean((predict - y_batch) ** 2)

    return total_loss / steps


def train_batch_gen_classifier(folder_path, steps, batch_size):
    step = 0
    train_files = list(glob(folder_path))
    np.random.shuffle(train_files)
    size = 80

    while True:
        X_train, y_train = [], []

        for train_file in train_files[step * batch_size:(step + 1) * batch_size]:
            mel = np.load(train_file).astype(np.float32).T
            if mel.shape[1] > size:
                idx = np.random.randint(mel.shape[1] - size)
                mel = mel[:, idx:idx + size].reshape((1, 80, size))
            else:
                mel = cv2.resize(mel, (80, size)).reshape((1, 80, size))

            X_train.append(mel)
            if train_file.split('/')[1] == 'clean':
                y_train.append(1)
            else:
                y_train.append(0)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        step += 1
        if step == steps:
            step = 0

        X_batch = Variable(torch.FloatTensor(X_train))
        y_batch = Variable(torch.LongTensor(y_train))

        yield X_batch, y_batch


def val_batch_gen_classifier(folder_path, steps, batch_size):
    step = 0
    val_files = list(glob(folder_path))
    np.random.shuffle(val_files)
    size = 80

    while True:
        X_val, y_val = [], []

        for val_file in val_files[step * batch_size:(step + 1) * batch_size]:
            mel = np.load(val_file).astype(np.float32).T
            x = []

            if mel.shape[1] > size:
                for i in range((mel.shape[1] - size) // 40):
                    x.append(mel[:, i * 40:i * 40 + size].reshape(1, 80, size))
                x.append(mel[:, -size:].reshape(1, 80, size))
                X_val.append(np.array(x))
            else:
                X_val.append(cv2.resize(mel, (80, size)).reshape((1, 1, 80, size)))
            if val_file.split('/')[1] == 'clean':
                y_val.append(1)
            else:
                y_val.append(0)

        # X_val = np.array(X_val)
        y_val = np.array(y_val)

        step += 1
        if step == steps:
            step = 0
        yield X_val, y_val


def strange_accuracy(net, batch_gen, batch_size, val_size):
    steps = val_size // batch_size
    total_correct_sum = 0
    for batch_i in tqdm(range(steps)):
        X_batch, y_batch = next(batch_gen)
        predict = np.array(
            [np.sum(net(Variable(torch.FloatTensor(pack))).detach().cpu().numpy(), axis=0).argmax() for
             pack in X_batch])
        total_correct_sum += np.sum(predict == y_batch)

    return total_correct_sum / val_size


def train(net, num_epochs, steps, train_batch_gen, optimizer, criterion):
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_i in range(steps):
            X_batch, y_batch = next(train_batch_gen)

            optimizer.zero_grad()
            net_out = net(X_batch)
            loss = criterion(net_out, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss
            print('Train Epoch: {}({}/{}), Loss: {:.6f}'.format(epoch + 1, batch_i+1, steps, total_loss / (batch_i+1)), end="\r")


def process_classification(model, path_to_mel):
    mel = np.load(path_to_mel).astype(np.float32).T
    size = 80

    x = []

    if mel.shape[1] > size:
        for i in range((mel.shape[1] - size) // 40):
            x.append(mel[:, i * 40:i * 40 + size].reshape(1, 80, size))
        x.append(mel[:, -size:].reshape(1, 80, size))
        x = np.array(x)
    else:
        x = cv2.resize(mel, (80, size)).reshape((1, 1, 80, size))

    predict = np.sum(model(Variable(torch.FloatTensor(x))).detach().cpu().numpy(), axis=0).argmax()
    if predict == 1:
        return 'It\'s clean!'
    else:
        return 'It\'s noisy!'


def process_denoising(model, path_to_mel):
    mel = np.load(path_to_mel).astype(np.float32).T

    cv2.imshow('not cleaned', mel)
    cv2.waitKey(0)

    mel = mel.reshape((1, 1, *mel.shape))
    cleaned_mel = model(Variable(torch.FloatTensor(mel))).detach().cpu().numpy()[0][0]

    cv2.imshow('cleaned', cleaned_mel)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return cleaned_mel
