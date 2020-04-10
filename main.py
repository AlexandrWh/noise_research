from download import download
from train_and_evaluate_classifier import train_classifier
from train_and_evaluate_denoiser import train_denoiser
from demo import run_demo
from subarray import test_subarray


download()
print("*"*40)
train_classifier()
print("*"*40)
train_denoiser()
print("*"*40)
run_demo()
print("*"*40)
test_subarray()
