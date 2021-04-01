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
from util import corrupt_data_gaussian, rrmse, load_dataset, predict, fftshift2d_torch
from pytorch_wavelets import DTCWTForward, DTCWTInverse
import pandas as pd
from skopt import gp_minimize
from skopt.space import Real

np.random.seed(0)
random.seed(0)

# %%
#----------------------------------------------------------------------------
# Dataset loader

train_sz, valid_sz, test_sz = 800, 200, 200
train_img, train_spec = load_dataset('ixi_train.pkl', num_images=train_sz+valid_sz)
test_img, test_spec = load_dataset('ixi_valid.pkl', num_images=test_sz)
valid_img, valid_spec = train_img[train_sz:, :, :], train_spec[train_sz:, :, :]
train_img, train_spec = train_img[:train_sz, :, :], train_spec[:train_sz, :, :]
model_folder = 'trained_model'

## create wavelet transform
dtcwt_fn = DTCWTForward()
def dtcwt_from_np(imgs):
    torch_imgs = imgs[:, np.newaxis, :, :].astype(np.float32)
    torch_imgs = torch.from_numpy(torch_imgs)
    yl, yh = dtcwt_fn.forward(torch_imgs)
    return yl, yh

train_dtcwt = dtcwt_from_np(train_img)
valid_dtcwt = dtcwt_from_np(valid_img)
test_dtcwt = dtcwt_from_np(test_img)

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
## runner functions

def create_data(corr_type):
    cor_params = corruption_params[corr_type]
        
    global train_X, train_Y, train_X_spec, train_Y_spec, train_X_rows, train_Y_rows
    train_X, train_Y = np.zeros_like(train_img), np.zeros_like(train_img)
    train_X_spec, train_Y_spec = np.zeros_like(train_spec), np.zeros_like(train_spec)
    train_X_rows, train_Y_rows = [0]*train_img.shape[0], [0]*train_img.shape[0]

    global valid_X, valid_Y, valid_X_spec, valid_Y_spec, valid_X_rows, valid_Y_rows
    valid_X, valid_Y = np.zeros_like(valid_img), np.zeros_like(valid_img)
    valid_X_spec, valid_Y_spec = np.zeros_like(valid_spec), np.zeros_like(valid_spec)
    valid_X_rows, valid_Y_rows = [0]*valid_img.shape[0], [0]*valid_img.shape[0]

    global test_X, test_X_spec, test_X_rows
    test_X = np.zeros_like(test_img)
    test_X_spec = np.zeros_like(test_spec)
    test_X_rows = [0]*test_img.shape[0]

    t1 = time.time()
    for i in range(train_img.shape[0]):
        train_X[i], train_X_spec[i], train_X_rows[i] = corrupt_data_gaussian(0, train_spec[i], cor_params)
        train_Y[i], train_Y_spec[i], train_Y_rows[i] = corrupt_data_gaussian(0, train_spec[i], cor_params)

    for i in range(valid_img.shape[0]):
        valid_X[i], valid_X_spec[i], valid_X_rows[i] = corrupt_data_gaussian(0, valid_spec[i], cor_params)
        valid_Y[i], valid_Y_spec[i], valid_Y_rows[i] = corrupt_data_gaussian(0, valid_spec[i], cor_params)

    for i in range(test_img.shape[0]):
        test_X[i], test_X_spec[i], test_X_rows[i] = corrupt_data_gaussian(0, test_spec[i], cor_params)

    global train_output, valid_output, test_output
    train_output, valid_output, test_output = np.zeros_like(train_X), np.zeros_like(valid_X), np.zeros_like(test_X)

    print('time for corrupting data = %.2f sec' % (time.time()-t1))
    print("RRMSE train, test inputs = %.4f, %.4f" % (rrmse(train_img, train_X), rrmse(test_img, test_X)))
    print('data creation done')


def run_model(param_list):
    spatial_wt, fft_wt, dtcwt_wt = param_list[0], param_list[1], param_list[2]

    ## define and load model
    model = UNet(init_features=16)
    model_path = os.path.join(model_folder, '%s_%s_%s_%s_%s.pth' %
        (model_type, corr_type, spatial_wt, fft_wt, dtcwt_wt))
    optimizer = optim.Adam(model.parameters(), lr=0.02)
    device = torch.device("%s" % device_name)
    if device_name.startswith('cuda'):
        torch.cuda.set_device(device)
        print(device, torch.cuda.current_device())
    model.to(device)

    ## n2c model only
    target_img, target_spec, target_rows = train_img, train_spec, [[]]*len(train_img)
    valid_target_img, valid_target_spec, valid_target_rows = valid_img, valid_spec, [[]]*len(valid_img)
    target_dtcwt, valid_target_dtcwt = train_dtcwt, valid_dtcwt

    ## add channel dimension
    global train_X, train_Y, train_X_spec, train_Y_spec, train_X_rows, train_Y_rows
    global valid_X, valid_Y, valid_X_spec, valid_Y_spec, valid_X_rows, valid_Y_rows
    global test_X, test_X_spec, test_X_rows
    train_X, valid_X = train_X[:, np.newaxis, :, :], valid_X[:, np.newaxis, :, :]
    target_img, valid_target_img = target_img[:, np.newaxis, :, :], valid_target_img[:, np.newaxis, :, :]
    
    min_valid_loss, min_loss_ep = 0, 0
    early_stopping_tolerance = 10
    dtcwt_fn = DTCWTForward().to(device)

    for ep in range(epochs):
        model.train()
        running_loss, running_fft_loss, running_dtcwt_loss = 0.0, 0.0, 0.0
        i = 0
        t2 = time.time()
        while i < len(train_X):
            
            inputs = torch.from_numpy(train_X[i : i+batch_size]).to(device)
            labels = torch.from_numpy(target_img[i : i+batch_size]).to(device)
            sz = len(inputs)

            optimizer.zero_grad()
            outputs = model(inputs)

            ## spatial loss
            loss = torch.mean((outputs - labels)**2) * spatial_wt
            del labels, inputs ## free up gpu

            ## dtcwt loss
            output_dtcwt = dtcwt_fn.forward(outputs)
            target_dtcwt_labels = target_dtcwt[0][i: i+batch_size].to(device)
            dtcwt_mse = torch.mean(torch.absolute(output_dtcwt[0] - target_dtcwt_labels)**2)
            for j in range(len(output_dtcwt[1])):
                target_dtcwt_labels = target_dtcwt[1][j][i: i+batch_size].to(device)
                dtcwt_mse += torch.mean(torch.absolute(output_dtcwt[1][j] - target_dtcwt_labels)**2)
            running_dtcwt_loss += dtcwt_mse * sz
            loss += dtcwt_mse * dtcwt_wt
            del output_dtcwt, target_dtcwt_labels ## free up gpu

            outputs = torch.squeeze(outputs, dim=1)
            ## fft loss
            output_spec = torch.fft.fftn(outputs, dim=(-2, -1)).type(torch.complex64)
            target_spec_labels = torch.from_numpy(target_spec[i : i+batch_size]).to(device)
            fft_mse = torch.mean(torch.absolute(output_spec - target_spec_labels)**2)
            running_fft_loss += fft_mse*sz
            loss += fft_mse * fft_wt
            del output_spec, target_spec_labels ## free up gpu
            del outputs
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item()*sz
            i += batch_size

        running_loss /= len(train_X)
        running_fft_loss /= len(train_X)
        running_dtcwt_loss /= len(train_X)

        ## validation loss
        running_valid_loss = 0.0
        i = 0
        model.eval()
        with torch.no_grad():
            while i < len(valid_X):
                inputs = torch.from_numpy(valid_X[i : i+batch_size]).to(device)
                labels = torch.from_numpy(valid_target_img[i : i+batch_size]).to(device)
                sz = len(inputs)
                outputs = model(inputs)
                ## spatial loss
                loss = torch.mean((outputs - labels)**2) * spatial_wt
                del labels, inputs ## free up gpu
                ## dtcwt loss
                output_dtcwt = dtcwt_fn.forward(outputs)
                target_dtcwt_labels = valid_target_dtcwt[0][i: i+batch_size].to(device)
                dtcwt_loss = torch.mean(torch.absolute(output_dtcwt[0] - target_dtcwt_labels)**2)
                for j in range(len(output_dtcwt[1])):
                    target_dtcwt_labels = valid_target_dtcwt[1][j][i: i+batch_size].to(device)
                    dtcwt_loss += torch.mean(torch.absolute(output_dtcwt[1][j] - target_dtcwt_labels)**2)
                loss += dtcwt_loss * dtcwt_wt
                del output_dtcwt, target_dtcwt_labels ## free up gpu
                ## fft loss
                outputs = torch.squeeze(outputs, dim=1)
                output_spec = torch.fft.fftn(outputs, dim=(-2, -1)).type(torch.complex64)
                target_spec_labels = torch.from_numpy(valid_target_spec[i : i+batch_size]).to(device)
                fft_mse = torch.mean(torch.absolute(output_spec - target_spec_labels)**2)
                loss += fft_mse * fft_wt
                del output_spec, target_spec_labels ## free up gpu
                del outputs

                running_valid_loss += loss.item()*sz
                i += batch_size

        running_valid_loss /= len(valid_X)
        
        ## save best model
        if ep==0 or running_valid_loss < min_valid_loss:
            min_valid_loss = running_valid_loss
            min_loss_ep = ep
            torch.save(model.state_dict(), model_path)
        ## early stopping
        if ep>60 and ep-min_loss_ep > early_stopping_tolerance:
            # print('Stopping early...total epochs = %d' % ep)
            break

    model.eval()
    test_output = predict(model, test_X, batch_size, device)
    rrmse_output = rrmse(test_img, test_output)
    print('rrmse at params %s : %.4f', (param_list, rrmse_output))
    return rrmse_output

# %%
## create data
epochs = 300
batch_size = 64
device_name = 'cuda:7'
model_type = 'n2c'
corr_type = 'high_2'
create_data(corr_type)

# %%
## load initial data
spatial_wts = [0.0, 3.0, 6.0, 9.0]
fft_wts = [0.0, 1e-6, 1e-4, 1e-2, 1.0]
dtcwt_wts = [0.0, 3.0, 6.0, 9.0]
df = pd.read_csv('n2c_rrmse.csv')

spatial_dim = Real(0.0, 10.0, prior='uniform')
fft_dim = Real(1e-6, 1, prior='log-uniform')
dtcwt_dim = Real(0.0, 10.0, prior='uniform')
dimensions = [spatial_dim, fft_dim, dtcwt_dim]

x0, y0 = [], []
for idx, row in df[df['noise_level']=='high_2'].iterrows():
    if not row.isnull().any():
        if row['fft_wt']>0:
            x0.append([row['spatial_wt'], row['fft_wt'], row['dtcwt_wt']])
            y0.append(row['rrmse'])

# %%
search_result = gp_minimize(
    func=run_model,
    dimensions=dimensions,
    n_calls=80,
    x0=x0,
    y0=y0,
    random_state=0
)

print('best params : %s, rrmse = %.4f' % (search_result.x, search_result.fun))
