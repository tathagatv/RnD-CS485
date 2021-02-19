# %%
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
import util
from pytorch_msssim import ssim
import random
import argparse
from unet import UNet

np.random.seed(0)
random.seed(0)

def fftshift2d(x, ifft=False):
    assert (len(x.shape) == 2) and all([(s % 2 == 1) for s in x.shape])
    s0 = (x.shape[0] // 2) + (0 if ifft else 1)
    s1 = (x.shape[1] // 2) + (0 if ifft else 1)
    x = np.concatenate([x[s0:, :], x[:s0, :]], axis=0)
    x = np.concatenate([x[:, s1:], x[:, :s1]], axis=1)
    return x

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
model_folder = 'trained_model'
img_folder = 'img'

# %%

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
    corrupt_spec[undersample_rows] = (1 + 1j)*(1e-5)
    corrupt_spec = fftshift2d(corrupt_spec, ifft=True)
    corrupt_img = np.real(np.fft.ifft2(corrupt_spec)).astype(np.float32)[:,:-1]
    corrupt_img = np.clip(corrupt_img, -0.5, 0.5)

    return corrupt_img, corrupt_spec

def rrmse(arrX, arrY):
    v = np.square(arrX - arrY).sum()
    v /= np.square(arrX + 0.5).sum()
    v = np.sqrt(v)
    return v

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

    print('time for making corrupted data = %.2f sec' % (time.time()-t1))
    print("rrmse input train = %.4f, test = %.4f" % (rrmse(train_img, train_X), rrmse(test_img, test_X)))
    return train_X, train_Y, test_X, test_Y

device = None
def define_load_model(model_path, gpu_id):
    model = UNet(init_features=16)
    global device
    device = torch.device("cuda:%s" % gpu_id)
    torch.cuda.set_device(device)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def predict(model, input_imgs):
    preds = np.zeros_like(input_imgs)
    batch_size = 32
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

def plot_spec(spec):
    spec = abs(spec)
    spec = np.log10(spec)
    spec -= spec.min()
    spec *= 255/spec.max()
    plt.figure()
    plt.imshow(spec, cmap='gray')

def plot_img(img_arr, img_name_arr, name, idx, title, save_img=False, cmap='gray'):

    rrmse_arr = [0]*4
    for i in range(1,4):
        rrmse_arr[i] = rrmse(img_arr[0], img_arr[i])
    for i in range(4):
        img_arr[i] = (img_arr[i]+0.5)*255

    combined_data = np.array(img_arr)
    _min, _max = np.amin(combined_data), np.amax(combined_data)
    fig = plt.figure(figsize=(18,15))

    def plot_ax(ax, im, sub_title):
        h = ax.imshow(im, cmap=cmap,vmin=_min, vmax=_max)
        ax.autoscale(False)
        ax.set_title(sub_title)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)

    ax = plt.subplot("221")
    plot_ax(ax, img_arr[0], '%s' % img_name_arr[0])
    ax = plt.subplot("222")
    plot_ax(ax, img_arr[1], '%s, rrmse = %.4f' % (img_name_arr[1], rrmse_arr[1]))
    ax = plt.subplot("223")
    plot_ax(ax, img_arr[2], '%s, rrmse = %.4f'% (img_name_arr[2], rrmse_arr[2]))
    ax = plt.subplot("224")
    plot_ax(ax, img_arr[3], '%s, rrmse = %.4f'% (img_name_arr[3], rrmse_arr[3]))
    # fig.suptitle(title)
    if save_img:
        plt.savefig(os.path.join(img_folder, "%s_%d.png" % (name, idx)), bbox_inches='tight')
    plt.close(fig)


def create_rrmse_data(cor_params, model_file, plot_title, gpu_id):
    print('\n------------------------')
    print("Corruption type: %s" % plot_title)
    train_X, train_Y, test_X, test_Y = create_noisy_data(cor_params)
    rrmse_train_input, rrmse_test_input = rrmse_list(train_img, train_X), rrmse_list(test_img, test_X)
    
    ## n2n predictions
    model = define_load_model(os.path.join(model_folder,"n2n_%s.pth" % model_file), gpu_id)
    train_output_n2n, test_output_n2n = predict(model, train_X), predict(model, test_X)
    rrmse_train_n2n, rrmse_test_n2n = rrmse_list(train_img, train_output_n2n), rrmse_list(test_img, test_output_n2n)
    print("rrmse n2n train = %.4f, test = %.4f" % (rrmse(train_img, train_output_n2n), rrmse(test_img, test_output_n2n)))
    
    ## n2c predictions
    model = define_load_model(os.path.join(model_folder,"n2c_%s.pth" % model_file), gpu_id)
    train_output_n2c, test_output_n2c = predict(model, train_X), predict(model, test_X)
    rrmse_train_n2c, rrmse_test_n2c = rrmse_list(train_img, train_output_n2c), rrmse_list(test_img, test_output_n2c)
    print("rrmse n2c train = %.4f, test = %.4f" % (rrmse(train_img, train_output_n2c), rrmse(test_img, test_output_n2c)))

    ## sample prediction images
    for idx in [34, 97, 120]:
        img_arr = [test_img[idx], test_X[idx], test_output_n2n[idx], test_output_n2c[idx]]
        img_name_arr = ['Clean', 'Noisy input', 'N2N', 'N2C']
        plot_img(img_arr, img_name_arr, 'pred_%s' % model_file,
            idx, 'Predictions on %s corruption' % model_file, True)
    
    ## boxplot
    f = plt.figure()
    plt.boxplot((rrmse_test_input, rrmse_test_n2n, rrmse_test_n2c),
        labels=('test input', 'n2n test', 'n2c test'), notch=True)
    plt.title(plot_title)
    plt.ylabel('RRMSE')
    plt.savefig(os.path.join(img_folder,"boxplot_%s.png" % model_file))
    plt.close(f)

# %%
## corruption parameters

corruption_high_params1 = {
    'central_band' : 1/10,
    'undersample_fraction_central' : 0,
    'undersample_fraction_side' : 1/1.3,
    'variance' : 150
}
corruption_low_params1 = {
    'central_band' : 1/6,
    'undersample_fraction_central' : 0,
    'undersample_fraction_side' : 1/1.8,
    'variance' : 45
}
corruption_high_params2 = {
    'central_band' : 0,
    'undersample_fraction_central' : 0,
    'undersample_fraction_side' : 0,
    'variance' : 220
}
corruption_low_params2 = {
    'central_band' : 0,
    'undersample_fraction_central' : 0,
    'undersample_fraction_side' : 0,
    'variance' : 80
}
corruption_high_params3 = {
    'central_band' : 1/12,
    'undersample_fraction_central' : 0,
    'undersample_fraction_side' : 1/1.2,
    'variance' : 0
}
corruption_low_params3 = {
    'central_band' : 1/7,
    'undersample_fraction_central' : 0,
    'undersample_fraction_side' : 1/1.6,
    'variance' : 0
}

gpu_id = 1
create_rrmse_data(corruption_high_params1, 'high_1', 'Undersample + Noise high', gpu_id=gpu_id)
create_rrmse_data(corruption_low_params1, 'low_1', 'Undersample + Noise low', gpu_id=gpu_id)
create_rrmse_data(corruption_high_params2, 'high_2', 'Noise high', gpu_id=gpu_id)
create_rrmse_data(corruption_low_params2, 'low_2', 'Noise low', gpu_id=gpu_id)
create_rrmse_data(corruption_high_params3, 'high_3', 'Undersample high', gpu_id=gpu_id)
create_rrmse_data(corruption_low_params3, 'low_3', 'Undersample low', gpu_id=gpu_id)

## create data to check rrmse and check plots
# train_X, test_X = [0]*3, [0]*3
# train_X[0], _, test_X[0], _ = create_noisy_data(corruption_high_params1)
# train_X[1], _, test_X[1], _ = create_noisy_data(corruption_high_params2)
# train_X[2], _, test_X[2], _ = create_noisy_data(corruption_high_params3)


# for idx in [34,97,120]:
#     img_arr = [test_img[idx], test_X[0][idx], test_X[1][idx], test_X[2][idx]]
#     plot_img(img_arr, name='sample_high_noise', idx=idx)


# %%
