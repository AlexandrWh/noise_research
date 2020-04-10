FROM ubuntu:16.04
FROM python:3.6

COPY download.py /download.py
COPY demo.py /demo.py
COPY nets.py /nets.py
COPY train_and_evaluate_classifier.py /train_and_evaluate_classifier.py
COPY train_and_evaluate_denoiser.py /train_and_evaluate_denoiser.py
COPY utils.py /utils.py
COPY main.py /main.py
COPY subarray.py /subarray.py
COPY classifier.pth /classifier.pth
COPY denoiser.pth /denoiser.pth
COPY clean.npy /clean.npy
COPY noisy.npy /noisy.npy

ADD requirements.txt /req/
RUN pip install -r /req/requirements.txt

CMD python /main.py