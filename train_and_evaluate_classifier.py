from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

from nets import ClassificationNet

from utils import val_batch_gen_classifier, train_batch_gen_classifier, strange_accuracy, train


def train_classifier():

    print('Training classifier...')

    train_classifier_gen = train_batch_gen_classifier("train/*/*/*.npy", steps=1, batch_size=1)
    classifier = ClassificationNet()

    for epochs, lr in [(1, 0.001), (1, 0.0005)]:
        optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
        train(classifier, epochs, 1, train_classifier_gen, optimizer, nn.CrossEntropyLoss())

    torch.save(classifier.state_dict(), 'new_classifier.pth')

    print('\nEvaluating...')

    val_classifier_gen = val_batch_gen_classifier("test/*/*/*.npy", steps=1, batch_size=1)

    #print(strange_accuracy(classifier, val_classifier_gen, 1, 1))

#train_classifier()



