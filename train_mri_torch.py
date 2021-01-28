import torch
import scipy
import numpy as np
import os
import math
import time
import util
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from torchsummary import summary
from PIL import Image
#----------------------------------------------------------------------
# the network

class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, init_features=24):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.encoder2 = UNet._block(features, features, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.encoder3 = UNet._block(features, features, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.encoder4 = UNet._block(features, features, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=1)

        self.bottleneck = UNet._block(features, features, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features, features, kernel_size=2, stride=1
        )
        self.decoder4 = UNet._block(features*2, features, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features, features, kernel_size=2, stride=1
        )
        self.decoder3 = UNet._block(features*2, features, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features, features, kernel_size=2, stride=1
        )
        self.decoder2 = UNet._block(features*2, features, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features, features, kernel_size=2, stride=1
        )
        self.decoder1 = UNet._block(features*2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )

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
train_img, train_spec = load_dataset('ixi_train.pkl')
test_img, test_spec = load_dataset('ixi_valid.pkl')

train_X, train_Y = np.zeros(train_img.shape), np.zeros(train_img.shape)
test_X, test_Y = np.zeros(test_img.shape), np.zeros(test_img.shape)
noise_std = 0.08
for i in range(train_img.shape[0]):
    img = train_img[i]
    noise_X = np.random.normal(0, noise_std, img.shape)
    noise_Y = np.random.normal(0, noise_std, img.shape)
    train_X[i] = (img + noise_X)
    train_Y[i] = (img + noise_Y)

for i in range(test_img.shape[0]):
    img = test_img[i]
    noise_X = np.random.normal(0, noise_std, img.shape)
    noise_Y = np.random.normal(0, noise_std, img.shape)
    test_X[i] = (img + noise_X)
    test_Y[i] = (img + noise_Y)

img = Image.fromarray((train_img[2]+0.5)*255)
img = img.convert('L')
img.save('sample_clean.png')
img = Image.fromarray((train_X[2]+0.5)*255)
img = img.convert('L')
img.save('sample_X.png')
img = Image.fromarray((train_Y[2]+0.5)*255)
img = img.convert('L')
img.save('sample_Y.png')

model = UNet(init_features=24)
epochs = 30
batch_size = 64

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for ep in range(epochs):

    running_loss = 0.0
    i = 0
    while i < len(train_X):

        inputs = train_X[i : i+batch_size]
        inputs = inputs[:, np.newaxis, :, :]
        inputs = torch.from_numpy(inputs)
        labels = train_Y[i : i+batch_size]
        labels = labels[:, np.newaxis, :, :]
        labels = torch.from_numpy(labels)
        print(inputs.dtype, labels.dtype)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()*len(inputs)
        i += batch_size

    running_loss = running_loss / len(train_X)
    
    print("epoch %s loss = %.3f" % (ep, running_loss))

print('Finished training')