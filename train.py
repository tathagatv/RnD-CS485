# %%
import torch
import scipy
import numpy as np
import os
import math
import time
import torch.nn as nn
import torch.optim as optim
import torch.fft
from collections import OrderedDict
from torchsummary import summary
import util
from skimage.metrics import structural_similarity as ssim
import random
import argparse
import matplotlib.pyplot as plt
from unet import UNet
from util import corrupt_data_gaussian, rrmse, load_dataset

np.random.seed(0)
random.seed(0)


def fftshift2d_torch(x, ifft=False):
    assert (len(x.shape) == 2) and all([(s % 2 == 1) for s in x.shape])
    s0 = (x.shape[0] // 2) + (0 if ifft else 1)
    s1 = (x.shape[1] // 2) + (0 if ifft else 1)
    x = torch.cat([x[s0:, :], x[:s0, :]], dim=0)
    x = torch.cat([x[:, s1:], x[:, :s1]], dim=1)
    return x

# %%
#----------------------------------------------------------------------------
# Dataset loader

dataset_train, dataset_test = dict(), dict()
train_img, train_spec = load_dataset('ixi_train.pkl', num_images=1000)
test_img, test_spec = load_dataset('ixi_valid.pkl', num_images=200)
model_folder = 'trained_model'

# %%
# Runner function

def plot_spec(spec):
    spec = abs(spec)
    spec = np.log10(spec)
    spec -= spec.min()
    plt.figure()
    plt.imshow(spec, cmap='gray')

def runner(corr_type, gpu_id, model_type, lamb=0, epochs=300, batch_size=64):

    if model_type not in ['n2n', 'n2c']:
        print("model_type should be 'n2n' or 'n2c'")
        return
    cor_params = corruption_params[corr_type]
    
    train_X, train_Y = np.zeros_like(train_img), np.zeros_like(train_img)
    test_X, test_Y = np.zeros_like(test_img), np.zeros_like(test_img)
    train_X_spec, train_Y_spec = np.zeros_like(train_spec), np.zeros_like(train_spec)
    test_X_spec, test_Y_spec = np.zeros_like(test_spec), np.zeros_like(test_spec)
    train_X_rows, train_Y_rows = [0]*train_img.shape[0], [0]*train_img.shape[0]
    test_X_rows, test_Y_rows = [0]*test_img.shape[0], [0]*test_img.shape[0]

    t1 = time.time()
    for i in range(train_img.shape[0]):
        train_X[i], train_X_spec[i], train_X_rows[i] = corrupt_data_gaussian(0, train_spec[i], cor_params)
        train_Y[i], train_Y_spec[i], train_Y_rows[i] = corrupt_data_gaussian(0, train_spec[i], cor_params)

    for i in range(test_img.shape[0]):
        test_X[i], test_X_spec[i], test_X_rows[i] = corrupt_data_gaussian(0, test_spec[i], cor_params)
        test_Y[i], test_Y_spec[i], test_Y_rows[i] = corrupt_data_gaussian(0, test_spec[i], cor_params)

    train_output, test_output = np.zeros_like(train_X), np.zeros_like(test_X)

    print("RRMSE train, test inputs = %.4f, %.4f" % (rrmse(train_img, train_X), rrmse(test_img, test_X)))
    print("SSIM train, test inputs = %.4f, %.4f" % (ssim(train_img, train_X), ssim(test_img, test_X)))
    print('time for corrupting data = %.2f sec' % (time.time()-t1))
    print('data loading done')

    model = UNet(init_features=16)
    model_path = os.path.join(model_folder, '%s_%s_%s.pth' % (model_type, corr_type, lamb))

    optimizer = optim.Adam(model.parameters(), lr=0.02)

    device = torch.device("%s" % gpu_id)
    if gpu_id.startswith('cuda'):
        torch.cuda.set_device(device)
        print(device, torch.cuda.current_device())
    model.to(device)

    ## load model if existing
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("loaded existing model")
    except Exception as e:
        print(e)
    model.train()

    target_img, target_spec, target_rows = 0, 0, 0
    if model_type == 'n2c':
        target_img, target_spec, target_rows = train_img, train_spec, [[]]*len(train_img)
    elif model_type == 'n2n':
        target_img, target_spec, target_rows = train_Y, train_Y_spec, train_Y_rows
    t1 = time.time()
    for ep in range(epochs):

        running_loss = 0.0
        running_freq_loss = 0.0
        i = 0
        t2 = time.time()
        while i < len(train_X):
            inputs = train_X[i : i+batch_size]
            inputs = inputs[:, np.newaxis, :, :].astype(np.float32)
            inputs = torch.from_numpy(inputs)
            labels = target_img[i : i+batch_size]
            labels = labels[:, np.newaxis, :, :].astype(np.float32)
            labels = torch.from_numpy(labels)

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = torch.mean((outputs - labels)**2) * (1-lamb)
            output_spec = torch.fft.fftn(outputs, dim=(-2, -1)).type(torch.complex64)
            outputs = outputs.cpu().detach().numpy()[:,0,:,:]
            inputs = inputs.cpu().detach().numpy()[:,0,:,:]

            ## compute regularization term
            for j in range(i, min(i+batch_size, len(train_X))):
                # output_spec[j-i] = torch.fft.fftshift(output_spec[j-i], dim=(-2, -1))
                output_spec[j-i, 0] = fftshift2d_torch(output_spec[j-i, 0])
                output_spec[j-i, 0][target_rows[j]] = (1 + 1j)*(1e-5)
                # output_spec[j-i] = torch.fft.ifftshift(output_spec[j-i], dim=(-2, -1))
                output_spec[j-i, 0] = fftshift2d_torch(output_spec[j-i, 0], ifft=True)
            
            target_spec_labels = target_spec[i : i+batch_size]
            target_spec_labels = target_spec_labels[:, np.newaxis, :, :]
            target_spec_labels = torch.from_numpy(target_spec_labels)
            target_spec_labels = target_spec_labels.to(device)
            v = torch.mean(torch.absolute(output_spec - target_spec_labels)**2)
            running_freq_loss += v*len(inputs)
            loss += v*lamb
            loss.backward()
            optimizer.step()
            output_spec = output_spec.cpu().detach().numpy()[:,0,:,:]
            target_spec_labels = target_spec_labels.cpu().detach().numpy()[:,0,:,:]

            train_output[i : i+batch_size] = outputs
            running_loss += loss.item()*len(inputs)
            i += batch_size

        running_loss = running_loss / len(train_X)
        rrmse_train = rrmse(train_img, train_output)
        running_freq_loss /= len(train_X)
        
        print("epoch %s loss = %.5f, time = %.2f sec, rrmse = %.4f, k-space mse = %.4f" %
        (ep, running_loss, time.time()-t2, rrmse_train, running_freq_loss))
        if (ep+1) % 10 == 0:
            torch.save(model.state_dict(), model_path)
                

    print('Finished training, total time = %.1f min' % ((time.time()-t1)/60))
    torch.save(model.state_dict(), model_path)

    #------------------------------------------------------
    # result metrics

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    def predict(input_imgs):
        preds = np.zeros_like(input_imgs)
        for i in range(0, len(input_imgs), batch_size):
            inputs = input_imgs[i : i+batch_size][:, np.newaxis, :, :].astype(np.float32)
            inputs = torch.from_numpy(inputs).to(device)
            outputs = model(inputs).cpu().detach().numpy()
            preds[i : i+batch_size, :, :] = outputs[:, 0, :, :]
        return preds

    train_output = predict(train_X)

    rrmse_inp = rrmse(train_img, train_X)
    rrmse_out = rrmse(train_img, train_output)
    ssim_inp = ssim(train_img, train_X)
    ssim_out = ssim(train_img, train_output)

    print("Train data")
    print('rrmse input = %.4f, output = %.4f' % (rrmse_inp, rrmse_out))
    print('ssim input = %.4f, output = %.4f' % (ssim_inp, ssim_out))

    test_output = predict(test_X)

    rrmse_inp = rrmse(test_img, test_X)
    rrmse_out = rrmse(test_img, test_output)
    ssim_inp = ssim(test_img, test_X)
    ssim_out = ssim(test_img, test_output)

    print("Test data")
    print('rrmse input = %.4f, output = %.4f' % (rrmse_inp, rrmse_out))
    print('ssim input = %.4f, output = %.4f' % (ssim_inp, ssim_out))

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

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, help="n2n or n2c")
parser.add_argument("--gpu", type=str, help="free gpu - (cuda:id) or cpu")
parser.add_argument("--lamb", type=float, help="regularization weight")
args = parser.parse_args()

gpu_id = args.gpu
model_type = args.model_type
lamb = args.lamb

if gpu_id is None or model_type is None or lamb is None:
    print("gpu, model_type, lamb needed")
    exit()
batch_size = 60

### under + noise
print('\n---------------------------')
runner(corr_type='low_1', gpu_id=gpu_id, model_type=model_type, lamb=lamb, batch_size=batch_size)
print('\n---------------------------')
runner(corr_type='high_1', gpu_id=gpu_id, model_type=model_type, lamb=lamb, batch_size=batch_size)

### noise
print('\n---------------------------')
runner(corr_type='low_2', gpu_id=gpu_id, model_type=model_type, lamb=lamb, batch_size=batch_size)
print('\n---------------------------')
runner(corr_type='high_2', gpu_id=gpu_id, model_type=model_type, lamb=lamb, batch_size=batch_size)

## under
print('\n---------------------------')
runner(corr_type='low_3', gpu_id=gpu_id, model_type=model_type, lamb=lamb, batch_size=batch_size)
print('\n---------------------------')
runner(corr_type='high_3', gpu_id=gpu_id, model_type=model_type, lamb=lamb, batch_size=batch_size)



