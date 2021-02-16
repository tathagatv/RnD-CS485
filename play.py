import matplotlib.pyplot as plt
import torch
import scipy
import numpy as np
import os
import math
import time
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torchsummary import summary
from PIL import Image
import util
from pytorch_msssim import ssim
import random
import argparse
from unet import UNet

def fftshift2d(x, ifft=False):
    assert (len(x.shape) == 2) and all([(s % 2 == 1) for s in x.shape])
    s0 = (x.shape[0] // 2) + (0 if ifft else 1)
    s1 = (x.shape[1] // 2) + (0 if ifft else 1)
    x = np.concatenate([x[s0:, :], x[:s0, :]], axis=0)
    x = np.concatenate([x[:, s1:], x[:, :s1]], axis=1)
    return x

def corrupt_data_gaussian(img, spec, params):
    rows = spec.shape[0]
    central_band = params['central_band']
    undersample_fraction_central = params['undersample_fraction_central']
    undersample_fraction_side = params['undersample_fraction_side']
    variance = params['variance']
    low_band_idx = int(rows*(1-central_band)/2)
    hi_band_idx = int(rows*(1+central_band)/2)
    ## lower band
    undersample_rows_lower = random.sample(range(0, low_band_idx), int(low_band_idx*undersample_fraction_side))
    ## central band
    undersample_rows_central = random.sample(range(low_band_idx, hi_band_idx), int((hi_band_idx-low_band_idx)*undersample_fraction_central))
    ## upper band
    undersample_rows_upper = random.sample(range(hi_band_idx, rows), int((rows-hi_band_idx)*undersample_fraction_side))
    
    undersample_rows = undersample_rows_lower + undersample_rows_central + undersample_rows_upper
    noise = np.random.randn(*spec.shape) + np.random.randn(*spec.shape)*1j
    noise *= np.sqrt(variance/2)

    corrupt_spec = spec + noise
    corrupt_spec[undersample_rows] = 0.1 + 0.1j
    corrupt_spec = fftshift2d(corrupt_spec, ifft=True)
    corrupt_img = np.real(np.fft.ifft2(corrupt_spec)).astype(np.float32)[:,:-1]

    return corrupt_img, corrupt_spec

def rrmse(arrX, arrY):
    v = np.square(arrX - arrY).sum()
    v /= np.square(arrX - arrX.mean(axis=0)).sum()
    v = np.sqrt(v)
    return v

def load_dataset(fn, num_images=None, shuffle=False):
    # datadir = submit.get_path_from_template(config_mri.data_dir)
    datadir = '../mri-pkl'
    if fn.lower().endswith('.pkl'):
        abspath = os.path.join(datadir, fn)
        print ('Loading dataset from', abspath)
        img, spec = util.load_pkl(abspath)
    else:
        assert False

    if shuffle:
        perm = np.arange(img.shape[0])
        np.random.shuffle(perm)
        if num_images is not None:
            perm = perm[:num_images]
        img = img[perm]
        spec = spec[perm]

    if num_images is not None:
        img = img[:num_images]
        spec = spec[:num_images]

    # Remove last row/column of the images, we're officially 255x255 now.
    img = img[:, :-1, :-1]

    # Convert to float32.
    assert img.dtype == np.uint8
    img = img.astype(np.float32) / 255.0 - 0.5

    return img, spec

dataset_train, dataset_test = dict(), dict()
train_img, train_spec = load_dataset('ixi_train.pkl', num_images=1000)
test_img, test_spec = load_dataset('ixi_valid.pkl', num_images=200)


def create_noisy_data(cor_params):
    t1 = time.time()

    train_X, train_Y = np.zeros_like(train_img), np.zeros_like(train_img)
    test_X, test_Y = np.zeros_like(test_img), np.zeros_like(test_img)
    train_X_spec, train_Y_spec = np.zeros_like(train_spec), np.zeros_like(train_spec)
    test_X_spec, test_Y_spec = np.zeros_like(test_spec), np.zeros_like(test_spec)
    train_output, test_output = np.zeros_like(train_X), np.zeros_like(test_X)

    for i in range(train_img.shape[0]):
        train_X[i], train_X_spec[i] = corrupt_data_gaussian(0, train_spec[i], cor_params)
        train_Y[i], train_Y_spec[i] = corrupt_data_gaussian(0, train_spec[i], cor_params)

    for i in range(test_img.shape[0]):
        test_X[i], test_X_spec[i] = corrupt_data_gaussian(0, test_spec[i], cor_params)
        test_Y[i], test_Y_spec[i] = corrupt_data_gaussian(0, test_spec[i], cor_params)

    print("RRMSE train inputs = %.4f" % rrmse(train_img, train_X))
    print("RRMSE test inputs = %.4f" % rrmse(test_img, test_X))
    print('time for corrupting data = %.2f sec' % (time.time()-t1))
    print('data generation done')
    return train_X, train_Y, test_X, test_Y

device = None
def define_load_model(model_path, gpu_id):
    model = UNet(init_features=16)
    global device
    device = torch.device("cuda:%s" % gpu_id)
    torch.cuda.set_device(device)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.train()
    return model

def predict(model, input_imgs):
    preds = np.zeros_like(input_imgs)
    batch_size = 16
    for i in range(0, len(input_imgs), batch_size):
        inputs = input_imgs[i : i+batch_size][:, np.newaxis, :, :].astype(np.float32)
        inputs = torch.from_numpy(inputs).to(device)
        outputs = model(inputs).cpu().detach().numpy()
        preds[i : i+batch_size, :, :] = outputs[:, 0, :, :]
    return preds

def rrmse_list(arrX, arrY):
    assert arrX.shape == arrY.shape
    l = []
    for i in range(arrX.shape[0]):
        l.append(rrmse(arrX[i], arrY[i]))
    return l

def create_rrmse_data(file_num, cor_params, plot_title, gpu_id=5):
    train_X, train_Y, test_X, test_Y = create_noisy_data(cor_params)
    rrmse_train_input = rrmse_list(train_img, train_X)
    rrmse_test_input = rrmse_list(test_img, test_X)
    
    model = define_load_model("n2n_unet%s.pth" % file_num, gpu_id)
    train_output = predict(model, train_X)
    test_output = predict(model, test_X)
    rrmse_train_n2n = rrmse_list(train_img, train_output)
    rrmse_test_n2n = rrmse_list(test_img, test_output)
    print("rrmse train n2n = %.4f" % rrmse(train_img, train_X))
    print("rrmse test n2n = %.4f" % )

    model = define_load_model("n2c_unet%s.pth" % file_num, gpu_id)
    train_output = predict(model, train_X)
    test_output = predict(model, test_X)
    rrmse_train_n2c = rrmse_list(train_img, train_output)
    rrmse_test_n2c = rrmse_list(test_img, test_output)
    print("rrmse train n2c = %.4f" % )
    print("rrmse test n2c = %.4f" % )
    
    plt.boxplot((rrmse_train_input, rrmse_train_n2n, rrmse_train_n2c, rrmse_test_n2n, rrmse_test_n2c),
    labels=('train input', 'n2n train', 'n2c train', 'n2n test', 'n2c test'))
    plt.title(plot_title)
    plt.ylabel('RRMSE')
    plt.savefig("boxplot%s.png" % file_num)

cor_params1 = {
    'central_band' : 1/4,
    'undersample_fraction_central' : 0,
    'undersample_fraction_side' : 1/2,
    'variance' : 20
}
cor_params2 = {
    'central_band' : 1/4,
    'undersample_fraction_central' : 0,
    'undersample_fraction_side' : 0,
    'variance' : 40
}
cor_params3 = {
    'central_band' : 1/5.5,
    'undersample_fraction_central' : 0,
    'undersample_fraction_side' : 1/2,
    'variance' : 0
}

create_rrmse_data(1, cor_params1, 'Undersample + Noise')
create_rrmse_data(2, cor_params2, 'Noise only')
create_rrmse_data(3, cor_params3, 'Undersample only')