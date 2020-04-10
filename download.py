import requests
from tqdm import tqdm
import os
import zipfile

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768
    iter = 0

    with open(destination, "wb") as f:
        print('Downloading ' + destination + '...')
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                iter += 1
                if not iter % 100:
                    print('Downloaded {} MBs'.format(iter*CHUNK_SIZE/(1024*1024)), end="\r")

def download():
    files = [('1jVeoX3yNGL3IqycQKwLb8Hs2N49Advuu', 'train.zip'),
             ('1c8a9xlgThXiX4_zxAOwXkqcooz_MeSQf', 'val.zip'),
             ('1O1L3T_pRp52NZ3A23hBi0LuSg1dqCTCq', 'explore_ata.zip')]

    for id, name in files[::-1]:
        download_file_from_google_drive(id, name)
        with zipfile.ZipFile(name, 'r') as zip_ref:
            zip_ref.extractall('./')
            os.remove(name)
