import os
import numpy as np
import pickle
import random
import PIL.Image
import torch

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

    corrupt_spec = np.fft.fftshift(spec) + noise
    corrupt_spec[undersample_rows] = (1 + 1j)*(1e-5)
    corrupt_spec = np.fft.ifftshift(corrupt_spec)
    corrupt_img = np.real(np.fft.ifft2(corrupt_spec)).astype(np.float32)
    corrupt_img = np.clip(corrupt_img, 0.0, 1.0)

    return corrupt_img, corrupt_spec, undersample_rows

def predict(model, input_imgs, batch_size, device):
    preds = np.zeros_like(input_imgs)
    with torch.no_grad():
        input_arr = input_imgs[:, np.newaxis, :, :]
        for i in range(0, len(input_imgs), batch_size):
            inputs = torch.from_numpy(input_arr[i : i+batch_size]).to(device)
            outputs = model(inputs).cpu().detach().numpy()
            preds[i : i+batch_size, :, :] = outputs[:, 0, :, :]
    return preds

def rrmse(arrX, arrY):
    v = np.square(arrX - arrY).sum()
    v /= np.square(arrX).sum()
    v = np.sqrt(v)
    return v

def load_dataset(fn, num_images=None, shuffle=False):
    # datadir = submit.get_path_from_template(config_mri.data_dir)
    datadir = '../mri-pkl'
    if fn.lower().endswith('.pkl'):
        abspath = os.path.join(datadir, fn)
        print ('Loading dataset from', abspath)
        img, spec = load_pkl(abspath)
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

    img = img[:, :-1, :]
    spec = np.fft.fft2(img).astype(np.complex64)
    print('image, spec shape from pkl file:', img.shape, spec.shape)
    return img, spec

# save_pkl, load_pkl are used by the mri code to save datasets
def save_pkl(obj, filename):
    with open(filename, 'wb') as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(filename):
    with open(filename, 'rb') as file:
        return pickle.load(file)

# save_snapshot, load_snapshot are used save/restore trained networks
def save_snapshot(submit_config, net, fname_postfix):
    dump_fname = os.path.join(submit_config.run_dir, "network_%s.pickle" % fname_postfix)
    with open(dump_fname, "wb") as f:
        pickle.dump(net, f)

def crop_np(img, x, y, w, h):
    return img[:, y:h, x:w]

# Run an image through the network (apply reflect padding when needed
# and crop back to original dimensions.)
def infer_image(net, img):
    w = img.shape[2]
    h = img.shape[1]
    pw, ph = (w+31)//32*32-w, (h+31)//32*32-h
    padded_img = img
    if pw!=0 or ph!=0:
        padded_img  = np.pad(img, ((0,0),(0,ph),(0,pw)), 'reflect')
    inferred = net.run(np.expand_dims(padded_img, axis=0), width=w+pw, height=h+ph)
    return clip_to_uint8(crop_np(inferred[0], 0, 0, w, h))
