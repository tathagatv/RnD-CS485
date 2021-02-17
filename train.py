# %%
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

# %%
#----------------------------------------------------------------------------
# Dataset noising functions

def fftshift2d(x, ifft=False):
    assert (len(x.shape) == 2) and all([(s % 2 == 1) for s in x.shape])
    s0 = (x.shape[0] // 2) + (0 if ifft else 1)
    s1 = (x.shape[1] // 2) + (0 if ifft else 1)
    x = np.concatenate([x[s0:, :], x[:s0, :]], axis=0)
    x = np.concatenate([x[:, s1:], x[:, :s1]], axis=1)
    return x

augment_translate_cache = dict()
def augment_data(img, spec, params):
    t = params.get('translate', 0)
    if t > 0:
        global augment_translate_cache
        trans = np.random.randint(-t, t + 1, size=(2,))
        key = (trans[0], trans[1])
        if key not in augment_translate_cache:
            x = np.zeros_like(img)
            x[trans[0], trans[1]] = 1.0
            augment_translate_cache[key] = fftshift2d(np.fft.fft2(x).astype(np.complex64))
        img = np.roll(img, trans, axis=(0, 1))
        spec = spec * augment_translate_cache[key]
    return img, spec

bernoulli_mask_cache = dict()
def corrupt_data(img, spec, params):
    ctype = params['type']
    assert ctype == 'bspec'
    p_at_edge = params['p_at_edge']
    global bernoulli_mask_cache
    if bernoulli_mask_cache.get(p_at_edge) is None:
        h = [s // 2 for s in spec.shape]
        r = [np.arange(s, dtype=np.float32) - h for s, h in zip(spec.shape, h)]
        r = [x ** 2 for x in r]
        r = (r[0][:, np.newaxis] + r[1][np.newaxis, :]) ** .5
        m = (p_at_edge ** (1./h[1])) ** r
        bernoulli_mask_cache[p_at_edge] = m
        print('Bernoulli probability at edge = %.5f' % m[h[0], 0])
        print('Average Bernoulli probability = %.5f' % np.mean(m))
    mask = bernoulli_mask_cache[p_at_edge]
    keep = (np.random.uniform(0.0, 1.0, size=spec.shape)**2 < mask)
    keep = keep & keep[::-1, ::-1]
    sval = spec * keep
    smsk = keep.astype(np.float32)
    spec = fftshift2d(sval / (mask + ~keep), ifft=True) # Add 1.0 to not-kept values to prevent div-by-zero.
    img = np.real(np.fft.ifft2(spec)).astype(np.float32)[:,:-1]
    return img, sval, smsk


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



# %%
#----------------------------------------------------------------------------
# Dataset loader

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


def runner(cor_params, model_file, gpu_id, model_type):

    if model_type not in ['n2n', 'n2c']:
        print("model_type should be 'n2n' or 'n2c'")
        return
    
    train_X, train_Y = np.zeros_like(train_img), np.zeros_like(train_img)
    test_X, test_Y = np.zeros_like(test_img), np.zeros_like(test_img)
    train_X_spec, train_Y_spec = np.zeros_like(train_spec), np.zeros_like(train_spec)
    test_X_spec, test_Y_spec = np.zeros_like(test_spec), np.zeros_like(test_spec)

    t1 = time.time()
    for i in range(train_img.shape[0]):
        train_X[i], train_X_spec[i] = corrupt_data_gaussian(0, train_spec[i], cor_params)
        train_Y[i], train_Y_spec[i] = corrupt_data_gaussian(0, train_spec[i], cor_params)

    for i in range(test_img.shape[0]):
        test_X[i], test_X_spec[i] = corrupt_data_gaussian(0, test_spec[i], cor_params)
        test_Y[i], test_Y_spec[i] = corrupt_data_gaussian(0, test_spec[i], cor_params)

    train_output, test_output = np.zeros_like(train_X), np.zeros_like(test_X)

    print("RRMSE train inputs = %.4f" % rrmse(train_img, train_X))
    print("RRMSE test inputs = %.4f" % rrmse(test_img, test_X))
    print('time for corrupting data = %.2f sec' % (time.time()-t1))
    print('data loading done')

    model = UNet(init_features=16)
    model_path = os.path.join(model_folder, '%s_%s.pth' % (model_type, model_file))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.02)

    device = torch.device("cuda:%d" % gpu_id)
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


    t1 = time.time()
    epochs = 300
    batch_size = 64
    for ep in range(epochs):

        running_loss = 0.0
        i = 0
        t2 = time.time()
        while i < len(train_X):
            inputs = train_X[i : i+batch_size]
            inputs = inputs[:, np.newaxis, :, :].astype(np.float32)
            inputs = torch.from_numpy(inputs)
            if model_type == 'n2c':
                labels = train_img[i : i+batch_size]
            else:
                labels = train_Y[i : i+batch_size]
            labels = labels[:, np.newaxis, :, :].astype(np.float32)
            labels = torch.from_numpy(labels)

            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_output[i : i+batch_size] = outputs.cpu().detach().numpy()[:,0,:,:]
            running_loss += loss.item()*len(inputs)
            i += batch_size

        running_loss = running_loss / len(train_X)
        rrmse_train = rrmse(train_img, train_output)
        
        print("epoch %s loss = %.5f, time = %.2f sec, rrmse = %.4f" % (ep, running_loss, time.time()-t2, rrmse_train))
        if ep>0 and ep % 10 == 0:
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

    print("Train data")
    print('rrmse input = %.4f, output = %.4f' % (rrmse_inp, rrmse_out))

    test_output = predict(test_X)

    rrmse_inp = rrmse(test_img, test_X)
    rrmse_out = rrmse(test_img, test_output)

    print("Test data")
    print('rrmse input = %.4f, output = %.4f' % (rrmse_inp, rrmse_out))



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


parser = argparse.ArgumentParser()
parser.add_argument("--model_type", type=str, help="n2n or n2c")
parser.add_argument("--gpu", type=int, help="free gpu id")
args = parser.parse_args()

gpu_id = args.gpu
model_type = args.model_type

if gpu_id is None or model_type is None:
    print("gpu_id and model_type needed")
    exit()


### under + noise
print('\n---------------------------')
runner(corruption_low_params1, 'low_1', gpu_id, model_type)
print('\n---------------------------')
runner(corruption_high_params1, 'high_1', gpu_id, model_type)

### noise
print('\n---------------------------')
runner(corruption_low_params2, 'low_2', gpu_id, model_type)
print('\n---------------------------')
runner(corruption_high_params2, 'high_2', gpu_id, model_type)

## under
print('\n---------------------------')
runner(corruption_low_params3, 'low_3', gpu_id, model_type)
print('\n---------------------------')
runner(corruption_high_params3, 'high_3', gpu_id, model_type)



