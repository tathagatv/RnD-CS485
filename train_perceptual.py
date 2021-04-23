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
from util import corrupt_data_gaussian, rrmse, load_dataset, predict, fftshift2d_torch, gram
from vgg import Vgg16

np.random.seed(0)
random.seed(0)

# %%
#----------------------------------------------------------------------------
# Dataset loader

train_sz, valid_sz, test_sz = 800, 200, 200
train_sz, valid_sz, test_sz = 600, 120, 120
train_sz, valid_sz, test_sz = 200, 100, 100
train_img, train_spec = load_dataset('ixi_train.pkl', num_images=train_sz+valid_sz)
test_img, test_spec = load_dataset('ixi_valid.pkl', num_images=test_sz)
valid_img, valid_spec = train_img[train_sz:, :, :], train_spec[train_sz:, :, :]
train_img, train_spec = train_img[:train_sz, :, :], train_spec[:train_sz, :, :]
model_folder = 'trained_model_perceptual'
if not os.path.isdir(model_folder):
    os.mkdir(model_folder)

# %%
# Runner function

def runner(corr_type, device_name, model_type, spatial_wt=1.0, content_wt=1.0, style_wt=1.0, epochs=300, batch_size=64):

    if model_type not in ['n2n', 'n2c']:
        print("model_type should be 'n2n' or 'n2c'")
        return
    cor_params = corruption_params[corr_type]
    
    train_X, train_Y = np.zeros_like(train_img), np.zeros_like(train_img)
    train_X_spec, train_Y_spec = np.zeros_like(train_spec), np.zeros_like(train_spec)
    train_X_mask, train_Y_mask = np.zeros_like(train_spec), np.zeros_like(train_spec)
    
    valid_X, valid_Y = np.zeros_like(valid_img), np.zeros_like(valid_img)
    valid_X_spec, valid_Y_spec = np.zeros_like(valid_spec), np.zeros_like(valid_spec)
    valid_X_mask, valid_Y_mask = np.zeros_like(valid_spec), np.zeros_like(valid_spec)

    test_X = np.zeros_like(test_img)
    test_X_spec = np.zeros_like(test_spec)
    test_X_mask = np.zeros_like(test_spec)

    t1 = time.time()
    for i in range(train_img.shape[0]):
        train_X[i], train_X_spec[i], train_X_mask[i] = corrupt_data_gaussian(0, train_spec[i], cor_params)
        train_Y[i], train_Y_spec[i], train_Y_mask[i] = corrupt_data_gaussian(0, train_spec[i], cor_params)

    for i in range(valid_img.shape[0]):
        valid_X[i], valid_X_spec[i], valid_X_mask[i] = corrupt_data_gaussian(0, valid_spec[i], cor_params)
        valid_Y[i], valid_Y_spec[i], valid_Y_mask[i] = corrupt_data_gaussian(0, valid_spec[i], cor_params)

    for i in range(test_img.shape[0]):
        test_X[i], test_X_spec[i], test_X_mask[i] = corrupt_data_gaussian(0, test_spec[i], cor_params)

    train_output, valid_output, test_output = np.zeros_like(train_X), np.zeros_like(valid_X), np.zeros_like(test_X)

    print('time for corrupting data = %.2f sec' % (time.time()-t1))
    print("RRMSE train, test inputs = %.4f, %.4f" % (rrmse(train_img, train_X), rrmse(test_img, test_X)))
    # print("SSIM train, test inputs = %.4f, %.4f" % (ssim(train_img, train_X), ssim(test_img, test_X)))
    print('data loading done')

    ## create model
    model = UNet(init_features=16)
    model_path = os.path.join(model_folder, '%s_%s_%s_%s_%s.pth' %
        (model_type, corr_type, spatial_wt, content_wt, style_wt))

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
    target_img, target_spec, target_mask = 0, 0, 0
    valid_target_img, valid_target_spec, valid_target_mask = 0, 0, 0
    if model_type == 'n2c':
        target_img, target_spec, target_mask = train_img, train_spec, np.ones_like(train_spec)
        valid_target_img, valid_target_spec, valid_target_mask = valid_img, valid_spec, np.ones_like(valid_spec)
    elif model_type == 'n2n':
        target_img, target_spec, target_mask = train_Y, train_Y_spec, train_Y_mask
        valid_target_img, valid_target_spec, valid_target_mask = valid_Y, valid_Y_spec, valid_Y_mask

    t1 = time.time()
    train_loss_history, valid_loss_history = [], []
    
    ## add channel dimension
    train_X, valid_X = train_X[:, np.newaxis, :, :], valid_X[:, np.newaxis, :, :]
    target_img, valid_target_img = target_img[:, np.newaxis, :, :], valid_target_img[:, np.newaxis, :, :]
    
    min_valid_loss, min_loss_ep = 0, 0
    early_stopping_tolerance = 10
    vgg = Vgg16()
    vgg.to(device)

    for ep in range(epochs):

        model.train()
        running_loss, running_spatial_loss, running_style_loss , running_content_loss = 0.0, 0.0, 0.0, 0.0
        i = 0
        t2 = time.time()
        while i < len(train_X):
            
            inputs = torch.from_numpy(train_X[i : i+batch_size]).to(device)
            labels = torch.from_numpy(target_img[i : i+batch_size]).to(device)
            sz = len(inputs)
            optimizer.zero_grad()

            outputs = model(inputs)
            del inputs
            spatial_loss = nn.MSELoss(reduction="mean")(outputs, labels)*spatial_wt
            running_spatial_loss += spatial_loss.item()*sz

            # # calculate total variation regularization (anisotropic version)
            # diff_i = torch.mean(torch.abs(outputs[:, :, :, 1:] - outputs[:, :, :, :-1]))
            # diff_j = torch.mean(torch.abs(outputs[:, :, 1:, :] - outputs[:, :, :-1, :]))
            # tv_loss = (diff_i + diff_j)*tv_wt
            # running_tv_loss += tv_loss.item()*sz
            
            ## vgg features
            outputs_fea = vgg(outputs)
            del outputs
            labels_fea = vgg(labels)
            del labels

            # calculate content loss
            content_loss = nn.MSELoss(reduction="mean")(outputs_fea[1], labels_fea[1])*content_wt
            running_content_loss += content_loss.item()*sz

            # calculate style loss
            style_loss = torch.tensor([0.0]).to(device)
            for fid in [0,1,2,3]:
                if style_wt == 0:
                    break
                gr1 = gram(outputs_fea)
                gr1 = gram(labels_fea)
                style_loss += nn.MSELoss(reduction="mean")(gr1, gr2)*style_wt
            del outputs_fea, labels_fea
            running_style_loss += style_loss.item()*sz

            loss =  spatial_loss + content_loss + style_loss
            loss.backward()
            optimizer.step()

            running_loss += loss.item()*sz
            i += batch_size

        running_loss /= len(train_X)
        running_spatial_loss /= len(train_X)
        running_content_loss /= len(train_X)
        running_style_loss /= len(train_X)
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
                spatial_loss = nn.MSELoss(reduction="mean")(outputs, labels)*spatial_wt
                # # total variation regularization (anisotropic version)
                # diff_i = torch.sum(torch.abs(outputs[:, :, :, 1:] - outputs[:, :, :, :-1]))
                # diff_j = torch.sum(torch.abs(outputs[:, :, 1:, :] - outputs[:, :, :-1, :]))
                # tv_loss = (diff_i + diff_j)*tv_wt
                ## vgg features
                outputs_fea = vgg(outputs)
                del outputs
                labels_fea = vgg(labels)
                del labels
                # calculate content loss
                content_loss = nn.MSELoss(reduction="mean")(outputs_fea[1], labels_fea[1])*content_wt
                style_loss = torch.tensor([0.0]).to(device)
                for fid in [0,1,2,3]:
                    if style_wt == 0:
                        break
                    gr1 = gram(outputs_fea)
                    gr1 = gram(labels_fea)
                    style_loss += nn.MSELoss(reduction="mean")(gr1, gr2)*style_wt
                del outputs_fea, labels_fea
                loss = spatial_loss + content_loss + style_loss
                running_valid_loss += loss.item()*sz
                i += batch_size

        running_valid_loss /= len(valid_X)
        valid_loss_history.append(running_valid_loss)

        if ep%30 == 0 or True:
            print("epoch %s, time = %.2f sec, losses: train = %.4f, spatial = %.4f, content = %.4f, style = %.4f, val = %.4f" %
            (ep, time.time()-t2, running_loss, running_spatial_loss, running_content_loss, running_style_loss, running_valid_loss))
        
        ## save best model
        if ep==0 or running_valid_loss < min_valid_loss:
            min_valid_loss = running_valid_loss
            min_loss_ep = ep
            torch.save(model.state_dict(), model_path)
        ## early stopping
        # if ep>60 and ep-min_loss_ep > early_stopping_tolerance:
        #     print('Stopping early...total epochs = %d' % ep)
        #     break
                
    train_X, valid_X = train_X[:, 0, :, :], valid_X[:, 0, :, :]

    print('\nFinished training, total time = %.1f min' % ((time.time()-t1)/60))
    print('Model saved in %s\n' % model_path)

    #------------------------------------------------------
    # result metrics

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print('Corruption type : %s, model type : %s, weights: spatial = %s, content = %s, style = %s'
    % (corr_type, model_type, spatial_wt, content_wt, style_wt))
    
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
    # print_metrics(ssim, 'SSIM')

    ## plot loss history
    loss_history_img = 'loss_history_img_perceptual'
    if not os.path.isdir(loss_history_img):
        os.mkdir(loss_history_img)
    fig = plt.figure()
    epoch_arr = list(range(1, len(train_loss_history)+1))
    init_offset = 0
    plt.plot(epoch_arr[init_offset:], train_loss_history[init_offset:])
    plt.plot(epoch_arr[init_offset:], valid_loss_history[init_offset:])
    plt.legend(['train loss', 'valid loss'])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('%s %s weights - spatial:%s, content:%s, style:%s' % (model_type, corr_type, spatial_wt, content_wt, style_wt))
    plt.tight_layout()
    plt.savefig(os.path.join(loss_history_img, '%s_loss.png' % (os.path.split(model_path)[-1][:-4])))
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
runner('high_3', 'cuda:6', 'n2c', 1e1, 1e-2, 0, 100, 24)

# %%

parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, help="n2n or n2c")
parser.add_argument("--device", type=str, help="free gpu - (cuda:id) or cpu")
parser.add_argument("--content_wt", type=float, help="content loss weight")
parser.add_argument("--style_wt", type=float, help="style loss weight")
parser.add_argument("--spatial_wt", type=float, help="spatial loss weight")
parser.add_argument("--noise_type", type=str, help="[high/low]_[1/2/3]")
args = parser.parse_args()

device_name = args.device
model_type = args.model_type
content_wt = args.content_wt
style_wt = args.style_wt
spatial_wt = args.spatial_wt
noise_type = args.noise_type

if device_name is None or model_type not in ['n2n', 'n2c']:
    print("device, model_type, noise_type needed")
    exit()
batch_size = 24
epochs = 300
content_wt = 1e1
spatial_wt = 1e1
style_wts = [1e2, 1e3, 1e4, 1e5]
style_wts = [0]

for style_wt in style_wts:
    print('\n---------------------------')
    runner(noise_type, device_name, model_type, content_wt, style_wt, 1e-6, epochs, batch_size)

