# %%
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
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
from skimage.metrics import structural_similarity as ssim
import random
import argparse
from unet import UNet
from util import corrupt_data_gaussian, rrmse, load_dataset, predict

np.random.seed(0)
random.seed(0)

def fftshift2d(x, ifft=False):
    assert (len(x.shape) == 2) and all([(s % 2 == 1) for s in x.shape])
    s0 = (x.shape[0] // 2) + (0 if ifft else 1)
    s1 = (x.shape[1] // 2) + (0 if ifft else 1)
    x = np.concatenate([x[s0:, :], x[:s0, :]], axis=0)
    x = np.concatenate([x[:, s1:], x[:, :s1]], axis=1)
    return x

train_sz, valid_sz, test_sz = 800, 200, 200
train_sz, valid_sz, test_sz = 600, 120, 120
train_img, train_spec = load_dataset('ixi_train.pkl', num_images=train_sz+valid_sz)
test_img, test_spec = load_dataset('ixi_valid.pkl', num_images=test_sz)
valid_img, valid_spec = train_img[train_sz:, :, :], train_spec[train_sz:, :, :]
train_img, train_spec = train_img[:train_sz, :, :], train_spec[:train_sz, :, :]
model_folder = 'trained_model_perceptual'
# model_folder = 'trained_perceptual_backup'
img_folder = 'img_perceptual'
if not os.path.isdir(img_folder):
    os.mkdir(img_folder)

# %%

def create_noisy_data(cor_params):
    t1 = time.time()

    train_X, train_Y = np.zeros_like(train_img), np.zeros_like(train_img)
    test_X, test_Y = np.zeros_like(test_img), np.zeros_like(test_img)
    train_X_spec, train_Y_spec = np.zeros_like(train_spec), np.zeros_like(train_spec)
    test_X_spec, test_Y_spec = np.zeros_like(test_spec), np.zeros_like(test_spec)
    train_output, test_output = np.zeros_like(train_X), np.zeros_like(test_X)

    for i in range(train_img.shape[0]):
        train_X[i], train_X_spec[i], _ = corrupt_data_gaussian(0, train_spec[i], cor_params)
        train_Y[i], train_Y_spec[i], _ = corrupt_data_gaussian(0, train_spec[i], cor_params)
    r = None
    for i in range(test_img.shape[0]):
        test_X[i], test_X_spec[i], r = corrupt_data_gaussian(0, test_spec[i], cor_params)
        test_Y[i], test_Y_spec[i], r = corrupt_data_gaussian(0, test_spec[i], cor_params)

    print('time for making corrupted data = %.2f sec' % (time.time()-t1))
    print('fraction of k-space maintained = %.4f' % (np.sum(r)/np.sum(r>=0)))
    print("rrmse input train = %.4f, test = %.4f" % (rrmse(train_img, train_X), rrmse(test_img, test_X)))
    print("ssim input train = %.4f, test = %.4f" % (ssim(train_img, train_X), ssim(test_img, test_X)))
    return train_X, train_Y, test_X, test_Y

device = 'cpu' ## default device
def define_load_model(model_path, device_name):
    model = UNet(init_features=16)
    global device
    device = torch.device(device_name)
    if device_name.startswith('cuda'):
        torch.cuda.set_device(device)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

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
    plt.figure()
    plt.imshow(spec, cmap='gray')

def plot_img(img_arr, img_name_arr, name, idx, title, save_img=False, cmap='gray'):

    rrmse_arr = [0]*4
    ssim_arr = [0]*4
    _max = np.max(max(img_arr, key=lambda o: np.max(o)))
    for i in range(1,4):
        rrmse_arr[i] = rrmse(img_arr[0], img_arr[i])
        ssim_arr[i] = ssim(img_arr[0], img_arr[i])
        img_arr[i] = img_arr[i] / _max
    img_arr[0] = img_arr[0] / _max

    fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(18,15))

    def plot_ax(ax, im, sub_title):
        im_rotated = np.rot90(im)
        h = ax.imshow(im_rotated, cmap=cmap,vmin=0, vmax=1)
        ax.autoscale(False)
        ax.set_title(sub_title)
        ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig.colorbar(h, cax=cax)

    plot_ax(axs[0,0], img_arr[0], '%s' % img_name_arr[0])
    plot_ax(axs[0,1], img_arr[1], '%s, rrmse = %.4f, ssim = %.4f'% (img_name_arr[1], rrmse_arr[1], ssim_arr[1]))
    plot_ax(axs[1,0], img_arr[2], '%s, rrmse = %.4f, ssim = %.4f'% (img_name_arr[2], rrmse_arr[2], ssim_arr[2]))
    plot_ax(axs[1,1], img_arr[3], '%s, rrmse = %.4f, ssim = %.4f'% (img_name_arr[3], rrmse_arr[3], ssim_arr[3]))
    # fig.suptitle(title)
    if save_img:
        plt.savefig(os.path.join(img_folder, "%s_%d.png" % (name, idx)), bbox_inches='tight')
    plt.close(fig)

def create_rrmse_data(device_name):

    df = {
        'noise_level':[],
        'content_wt':[],
        'style_wt':[],
        'tv_wt':[],
        'rrmse':[]
    }
    style_wts = [0, 1e2, 1e3, 1e4, 1e5]

    for corr_type in ['low_3', 'high_3']:
        cor_params = corruption_params[corr_type]
        train_X, train_Y, test_X, test_Y = create_noisy_data(cor_params)
        df['noise_level'].append(corr_type)
        df['content_wt'].append(None)
        df['style_wt'].append(None)
        df['tv_wt'].append(None)
        df['rrmse'].append(rrmse(test_img, test_X))
        for sty in style_wts:
            model_file = 'n2c_%s_%s_%s_%s.pth' % (corr_type, 1e2, sty, 0)
            model = define_load_model(os.path.join(model_folder, model_file), device_name)
            test_output = predict(model, test_X, 32, device)
            df['noise_level'].append(corr_type)
            df['content_wt'].append(1e2)
            df['style_wt'].append(sty)
            df['tv_wt'].append(0)
            df['rrmse'].append(rrmse(test_img, test_output))
    df = pd.DataFrame(df)
    df.to_csv('n2c_rrmse_perceptual.csv')
                    

    ## sample prediction images
    # for idx in [34]:
    #     img_arr = [test_img[idx], test_X[idx], test_output_n2n[idx], test_output_n2c[idx]]
    #     img_name_arr = ['Clean', 'Noisy input', 'N2N', 'N2C']
    #     plot_img(img_arr, img_name_arr, 'pred_%s' % model_file,
    #         idx, 'Predictions on %s corruption, lambda = %s' % (noise_desc, lamb), True)
    
corruption_params = {
    'high_1' : {
        'central_band' : 1/10,
        'undersample_fraction_central' : 0,
        'undersample_fraction_side' : 1/1.3,
        'variance' : 150
    },
    'low_1' : {
        'central_band' : 1/6,
        'undersample_fraction_central' : 0,
        'undersample_fraction_side' : 1/1.8,
        'variance' : 45
    },
    'high_2' : {
        'central_band' : 0,
        'undersample_fraction_central' : 0,
        'undersample_fraction_side' : 0,
        'variance' : 220
    },
    'low_2' : {
        'central_band' : 0,
        'undersample_fraction_central' : 0,
        'undersample_fraction_side' : 0,
        'variance' : 80
    },
    'high_3' : {
        'central_band' : 1/12,
        'undersample_fraction_central' : 0,
        'undersample_fraction_side' : 1/1.2,
        'variance' : 0
    },
    'low_3' : {
        'central_band' : 1/7,
        'undersample_fraction_central' : 0,
        'undersample_fraction_side' : 1/1.6,
        'variance' : 0
    }
}

# %%
# create_rrmse_data('cuda:6')

# %%
corr_type = 'low_3'
train_X, train_Y, test_X, test_Y = create_noisy_data(corruption_params[corr_type])
device_name = 'cuda:5'

# %%
model_file = '%s_%s_%s_%s_%s' % (corr_type, 1e3, 1e-3, 1e-1, 32)
model = define_load_model(os.path.join(model_folder,"n2c_%s.pth" % model_file), device_name)
test_output = predict(model, test_X, 32, device)
train_output = predict(model, train_X, 32, device)

# model_file = '%s_%s_%s_%s' % (corr_type, 0, 1, 1e-2)
# model = define_load_model(os.path.join(model_folder,"n2c_%s.pth" % model_file), device_name)
# test_output_dt = predict(model, test_X, 32, device)

# print(rrmse(test_img, test_output_dt))
print('test rrmse: %.4f, train rrmse: %.4f' % (rrmse(test_img, test_output), rrmse(train_img, train_output)))
print('test ssim: %.4f, train ssim: %.4f' % (ssim(test_img, test_output), ssim(train_img, train_output)))
exit(0)

# %%
# %matplotlib inline
def p1(r):
    return r
    # return r[60:200, 40:170]
    return r[100:220, 180:300]
plt.figure()
plt.imshow(p1(test_X[34]), cmap='gray')
plt.axis('off')
plt.show()
print(rrmse(test_img[34], test_X[34]))
plt.figure()
plt.imshow(p1(test_output[34]), cmap='gray')
plt.axis('off')
plt.show()
print(rrmse(test_img[34], test_output[34]))
# plt.figure()
# plt.imshow(p1(test_output[34]), cmap='gray')
# plt.axis('off')
# plt.show()
# print(rrmse(test_img[34], test_output_dt[34]))
plt.figure()
plt.imshow(p1(test_img[34]), cmap='gray')
plt.axis('off')
plt.show()


img_arr = [test_img[34], test_X[34], test_output[34], test_output[34]]
img_arr = list(map(p1, img_arr))
img_name_arr = ['Clean', 'Noisy input', 'content+style loss', 'content+style loss']
plot_img(img_arr, img_name_arr, 'pred_%s' % model_file,
    34, 'Predictions on k-space subsample corruption, content_wt = 100, style_wt = 1000', True)


# %%

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, help="free gpu - (cuda:id) or cpu")
parser.add_argument("--lamb", type=float, help="regularization weight")
args = parser.parse_args()

device_name = args.device
lamb = args.lamb

if device_name is None or lamb is None:
    print("device, lamb needed")
    exit()

create_rrmse_data('high_1', 'Undersample + Noise high', lamb, device_name)
create_rrmse_data('low_1', 'Undersample + Noise low', lamb, device_name)
create_rrmse_data('high_2', 'Noise high', lamb, device_name)
create_rrmse_data('low_2', 'Noise low', lamb, device_name)
create_rrmse_data('high_3', 'Undersample high', lamb, device_name)
create_rrmse_data('low_3', 'Undersample low', lamb, device_name)

# %%
