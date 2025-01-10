import argparse
import os
import time
from math import log10
import pandas
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import pandas as pd
import torch
from torch.autograd import Variable
from torchvision.transforms import ToTensor,ToPILImage
from model import Generator
from utils import read_h5,save_h5,frequency_distance,ssim,cal_psnr,normal



def main(opt):
    try:
        print("Starting the main function...")
        UPSCALE_FACTOR = opt.upscale_factor
        TEST_DATA_PATH = opt.test_data_path
        MODEL_PATH = opt.model_path
        SAVE_PATH = opt.save_path

        if not os.path.exists(os.path.join(SAVE_PATH, "predicted")):
            os.makedirs(os.path.join(SAVE_PATH, "predicted"))

        #print("Loading model...")
        model = Generator(scale_factor=UPSCALE_FACTOR).eval()
        model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        #print("Reading files...")
        file_list = [file for file in os.listdir(TEST_DATA_PATH) if file.endswith(".h5")]
        #print(f"File list: {file_list}")

        to_tensor = ToTensor()

        for file in file_list:
            #print(f"Processing file: {file}")
            data = read_h5(os.path.join(TEST_DATA_PATH, file))
            data_max = cp.max(data)
            data_min = cp.min(data)
            data_normal = normal(data)
            data_normal = to_tensor(data_normal).type(torch.FloatTensor)
            input = torch.unsqueeze(data_normal, dim=0).to(device)
            out = model(input).detach().cpu()

            # Verify output
            #print(f"Output shape: {out.shape}")

            out_renormal = (data_max - data_min) * out + data_min
            out_renormal = np.squeeze(out_renormal.numpy(), axis=(0, 1))

            save_path = os.path.join(SAVE_PATH, "predicted", file)
            #print(f"Saving file to: {save_path}")
            save_h5(out_renormal, save_path)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    path = "drive/MyDrive/Seis_Gan_Model/SeisGAN-Improving-Seismic-Image-Resolution-and-Reducing-Noise/"
    parser = argparse.ArgumentParser(description="Test the model performance")
    parser.add_argument("--upscale_factor", default=2, type=int, help="Super resolution upscale factor")
    parser.add_argument("--test_data_path", default= path + "data/SRF_2/test2/high", type=str, help="Path of test low seismic images with noise")
    parser.add_argument("--model_path", default= path + "result/SRF_2/model/netG_bestmodel.pth", type=str, help="Pre-trained model used for test")
    parser.add_argument("--save_path", default= path + "result/test_results/predicted3", type=str, help="Path to save the test results")
    opt = parser.parse_args()
    main(opt)
