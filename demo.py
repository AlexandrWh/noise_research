import argparse
from nets import DenoiseNet, ClassificationNet
from utils import process_classification, process_denoising
import cv2
import torch


def run_demo():
    print('running demo...')
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument('--mel_file_path', type=str, default='clean.npy',
                        help='path to the file you want to be processed')
    parser.add_argument('--action', help='classification or denoising', type=str, required=False, default='classification')

    args = parser.parse_args()

    path_to_mel = args.mel_file_path

    if args.action == 'classification':
        classifier = ClassificationNet()
        classifier.load_state_dict(torch.load('classifier.pth'))
        classifier.eval()

        print(process_classification(classifier, path_to_mel))


    elif args.action == 'denoising':
        denoiser = DenoiseNet()
        denoiser.load_state_dict(torch.load('denoiser.pth'))
        denoiser.eval()

        cleaned_mel = process_denoising(denoiser, path_to_mel)

    else:

        print('You may do onlу \'denoising\' or \'classification\' !!!')









