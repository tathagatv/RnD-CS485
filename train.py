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
from util import corrupt_data_gaussian, rrmse, load_dataset, predict
from pytorch_wavelets import DTCWTForward, DTCWTInverse

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

# %%
# Runner function

def plot_spec(spec):
    spec = abs(spec)
    spec = np.log10(spec)
    spec -= spec.min()
    plt.figure()
    plt.imshow(spec, cmap='gray')

def runner(corr_type, device_name, model_type, spatial_wt=1, fft_wt=0, dtcwt_wt=0, epochs=300, batch_size=64):

    if model_type not in ['n2n', 'n2c']:
        print("model_type should be 'n2n' or 'n2c'")
        return
    cor_params = corruption_params[corr_type]
    
    train_X, train_Y = np.zeros_like(train_img), np.zeros_like(train_img)
    train_X_spec, train_Y_spec = np.zeros_like(train_spec), np.zeros_like(train_spec)
    train_X_rows, train_Y_rows = [0]*train_img.shape[0], [0]*train_img.shape[0]
    
    valid_X, valid_Y = np.zeros_like(valid_img), np.zeros_like(valid_img)
    valid_X_spec, valid_Y_spec = np.zeros_like(valid_spec), np.zeros_like(valid_spec)
    valid_X_rows, valid_Y_rows = [0]*valid_img.shape[0], [0]*valid_img.shape[0]

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

    train_output, valid_output, test_output = np.zeros_like(train_X), np.zeros_like(valid_X), np.zeros_like(test_X)

    print('time for corrupting data = %.2f sec' % (time.time()-t1))
    print("RRMSE train, test inputs = %.4f, %.4f" % (rrmse(train_img, train_X), rrmse(test_img, test_X)))
    # print("SSIM train, test inputs = %.4f, %.4f" % (ssim(train_img, train_X), ssim(test_img, test_X)))
    print('data loading done')

    ## create model
    model = UNet(init_features=16)
    model_path = os.path.join(model_folder, '%s_%s_%s_%s_%s.pth' %
        (model_type, corr_type, spatial_wt, fft_wt, dtcwt_wt))

    optimizer = optim.Adam(model.parameters(), lr=0.02)

    device = torch.device("%s" % device_name)
    if device_name.startswith('cuda'):
        torch.cuda.set_device(device)
        print(device, torch.cuda.current_device())
    model.to(device)

    ## load model if existing
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("loaded existing model")
    except Exception as e:
        print('no pre-existing model')
    print('Model file name: %s' % model_path)

    ## set target images and spectrum
    target_img, target_spec, target_rows = 0, 0, 0
    valid_target_img, valid_target_spec, valid_target_rows = 0, 0, 0
    if model_type == 'n2c':
        target_img, target_spec, target_rows = train_img, train_spec, [[]]*len(train_img)
        valid_target_img, valid_target_spec, valid_target_rows = valid_img, valid_spec, [[]]*len(valid_img)
        target_dtcwt, valid_target_dtcwt = train_dtcwt, valid_dtcwt
    elif model_type == 'n2n':
        target_img, target_spec, target_rows = train_Y, train_Y_spec, train_Y_rows
        valid_target_img, valid_target_spec, valid_target_rows = valid_Y, valid_Y_spec, valid_Y_rows
        target_dtcwt, valid_target_dtcwt = train_dtcwt, valid_dtcwt

    t1 = time.time()
    train_loss_history, valid_loss_history = [], []
    
    ## add channel dimension
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
            dtcwt_loss = torch.mean(torch.absolute(output_dtcwt[0] - target_dtcwt_labels)**2)
            for j in range(len(output_dtcwt[1])):
                target_dtcwt_labels = target_dtcwt[1][j][i: i+batch_size].to(device)
                dtcwt_loss += torch.mean(torch.absolute(output_dtcwt[1][j] - target_dtcwt_labels)**2)
            running_dtcwt_loss += dtcwt_loss * sz
            loss += dtcwt_loss * dtcwt_wt
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
        train_loss_history.append(running_loss)

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
        valid_loss_history.append(running_valid_loss)

        print("epoch %s, time = %.2f sec, losses: train = %.4f, fft = %.4f, dtcwt = %.4f, val = %.4f" %
        (ep, time.time()-t2, running_loss, running_fft_loss, running_dtcwt_loss, running_valid_loss))
        
        ## save best model
        if ep==0 or running_valid_loss < min_valid_loss:
            min_valid_loss = running_valid_loss
            min_loss_ep = ep
            torch.save(model.state_dict(), model_path)
        ## early stopping
        if ep>20 and ep-min_loss_ep > early_stopping_tolerance:
            print('Stopping early...')
            break
                
    train_X, valid_X = train_X[:, 0, :, :], valid_X[:, 0, :, :]

    print('\nFinished training, total time = %.1f min' % ((time.time()-t1)/60))
    print('Model saved in %s\n' % model_path)

    #------------------------------------------------------
    # result metrics

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print('Corruption type : %s, model type : %s, weights: spatial = %s, fft = %s, dtcwt = %s'
    % (corr_type, model_type, spatial_wt, fft_wt, dtcwt_wt))

    rrmse_data = {'train':{}, 'test':{}, 'valid':{}}
    
    train_output = predict(model, train_X, batch_size, device)
    valid_output = predict(model, valid_X, batch_size, device)
    test_output = predict(model, test_X, batch_size, device)

    def print_metrics(f, metric_name):
        metric_data = {'train':{}, 'test':{}, 'valid':{}}
        metric_data['train']['input'] = f(train_img, train_X)
        metric_data['train']['output'] = f(train_img, train_output)
        
        metric_data['valid']['input'] = f(valid_img, valid_X)
        metric_data['valid']['output'] = f(valid_img, valid_output)

        metric_data['test']['input'] = f(test_img, test_X)
        metric_data['test']['output'] = f(test_img, test_output)

        print('\t%s data' % metric_name)
        print('\tInput\tOutput')
        for data in ['Train', 'Valid', 'Test']:
            print('%s\t%.4f\t%.4f' % (data, metric_data[data.lower()]['input'],
                metric_data[data.lower()]['output']))
    
    print_metrics(rrmse, 'RRMSE')
    print_metrics(ssim, 'SSIM')


    ## plot loss history
    fig = plt.figure()
    epoch_arr = list(np.arange(1, epochs+1).astype(np.int))
    plt.plot(epoch_arr[20:], train_loss_history[20:])
    plt.plot(epoch_arr[20:], valid_loss_history[20:])
    plt.legend(['train loss', 'valid loss'])
    plt.xlabel('epoch')
    plt.xticks(epoch_arr[20:])
    plt.ylabel('loss')
    plt.title('%s %s weights - spatial:%s, fft:%s, dtcwt:%s' %
        (model_type, corr_type, spatial_wt, fft_wt, dtcwt_wt))
    plt.tight_layout()
    plt.savefig('%s_loss.png' % (model_path[:-4]))
    plt.close(fig)

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
# runner('high_2', 'cuda:0', 'n2c', 0, 0, 1, 3, 64)

# %%

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, help="n2n or n2c")
parser.add_argument("--device", type=str, help="free gpu - (cuda:id) or cpu")
parser.add_argument("--spatial_wt", type=float, help="spatial mse loss weight")
parser.add_argument("--fft_wt", type=float, help="fourier mse loss weight")
parser.add_argument("--dtcwt_wt", type=float, help="wavelet mse loss weight")
args = parser.parse_args()

device_name = args.device
model_type = args.model_type
spatial_wt = args.spatial_wt
fft_wt = args.fft_wt
dtcwt_wt = args.dtcwt_wt

if device_name is None or model_type is None:
    print("device, model_type, lamb needed")
    exit()
batch_size = 64
epochs = 300

### under + noise
# print('\n---------------------------')
# runner('low_1', device_name, model_type, lamb, epochs, batch_size)
# print('\n---------------------------')
# runner('high_1', device_name, model_type, lamb, epochs, batch_size)

### noise
print('\n---------------------------')
runner('low_2', device_name, model_type, 100, 0, 0, epochs, batch_size)
print('\n---------------------------')
runner('high_2', device_name, model_type, 100, 0, 0, epochs, batch_size)

print('\n---------------------------')
runner('low_2', device_name, model_type, 0, 1, 0, epochs, batch_size)
print('\n---------------------------')
runner('high_2', device_name, model_type, 0, 1, 0, epochs, batch_size)

## under
# print('\n---------------------------')
# runner('low_3', device_name, model_type, lamb, epochs, batch_size)
# print('\n---------------------------')
# runner('high_3', device_name, model_type, lamb, epochs, batch_size)



