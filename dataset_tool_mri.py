import re
import argparse
import glob
import os
import PIL.Image
import numpy as np
import sys
import matplotlib.pyplot as plt
import util

import nibabel as nib

OUT_RESOLUTION = (260, 311)

# Select z-slices from [25,124] in hops of 5
slice_min = 25
slice_max = 125
slice_hops = 5

# Select train and validation subsets from IXI-T1 (these two lists shouldn't overlap)
all_basenames = [
    '100206_T1w_restore', '105014_T1w_restore', '109830_T1w_restore', '114419_T1w_restore', '118932_T1w_restore',
    '100307_T1w_restore', '105115_T1w_restore', '110007_T1w_restore', '114621_T1w_restore', '119126_T1w_restore',
    '100408_T1w_restore', '105216_T1w_restore', '110411_T1w_restore', '114823_T1w_restore', '119732_T1w_restore',
    '100610_T1w_restore', '105620_T1w_restore', '110613_T1w_restore', '114924_T1w_restore', '119833_T1w_restore',
    '101006_T1w_restore', '105923_T1w_restore', '111009_T1w_restore', '115017_T1w_restore', '120111_T1w_restore',
    '101107_T1w_restore', '106016_T1w_restore', '111312_T1w_restore', '115219_T1w_restore', '120212_T1w_restore',
    '101309_T1w_restore', '106319_T1w_restore', '111413_T1w_restore', '115320_T1w_restore', '120515_T1w_restore',
    '101410_T1w_restore', '106521_T1w_restore', '111514_T1w_restore', '115825_T1w_restore', '120717_T1w_restore',
    '101915_T1w_restore', '107018_T1w_restore', '111716_T1w_restore', '116120_T1w_restore', '121315_T1w_restore',
    '102008_T1w_restore', '107220_T1w_restore', '112112_T1w_restore', '116221_T1w_restore', '121416_T1w_restore',
    '102311_T1w_restore', '107321_T1w_restore', '112314_T1w_restore', '116524_T1w_restore', '121618_T1w_restore',
    '102513_T1w_restore', '107422_T1w_restore', '112516_T1w_restore', '116726_T1w_restore', '121820_T1w_restore',
    '102816_T1w_restore', '107725_T1w_restore', '112819_T1w_restore', '117122_T1w_restore', '121921_T1w_restore',
    '103111_T1w_restore', '108121_T1w_restore', '112920_T1w_restore', '117324_T1w_restore', '122317_T1w_restore',
    '103414_T1w_restore', '108222_T1w_restore', '113215_T1w_restore', '117930_T1w_restore', '122620_T1w_restore',
    '103515_T1w_restore', '108323_T1w_restore', '113619_T1w_restore', '118023_T1w_restore', '122822_T1w_restore',
    '103818_T1w_restore', '108525_T1w_restore', '113821_T1w_restore', '118124_T1w_restore', '123117_T1w_restore',
    '104012_T1w_restore', '108828_T1w_restore', '113922_T1w_restore', '118225_T1w_restore', '123420_T1w_restore',
    '104416_T1w_restore', '109123_T1w_restore', '114217_T1w_restore', '118528_T1w_restore', '123521_T1w_restore',
    '104820_T1w_restore', '109325_T1w_restore', '114318_T1w_restore', '118730_T1w_restore', '123824_T1w_restore'
]
train_basenames = all_basenames[0:80]
valid_basenames = all_basenames[80:100]


def fftshift2d(x, ifft=False):
    assert (len(x.shape) == 2) and all([(s % 2 == 1) for s in x.shape])
    s0 = (x.shape[0] // 2) + (0 if ifft else 1)
    s1 = (x.shape[1] // 2) + (0 if ifft else 1)
    x = np.concatenate([x[s0:, :], x[:s0, :]], axis=0)
    x = np.concatenate([x[:, s1:], x[:, :s1]], axis=1)
    return x

def preprocess_mri(input_files,
                   output_file):
    all_files = sorted(input_files)
    num_images = len(all_files)
    print('Input images: %d' % num_images)
    assert num_images > 0

    resolution = np.asarray(PIL.Image.open(all_files[0]), dtype=np.uint8).shape
    print(resolution)
    assert len(resolution) == 2 # Expect monochromatic images
    print('Image resolution: %s' % str(resolution))

    crop_size = tuple([((r - 1) | 1) for r in resolution])
    crop_slice = np.s_[:crop_size[0], :crop_size[1]]
    print('Crop size: %s' % str(crop_size))

    img_primal = np.zeros((num_images,) + resolution, dtype=np.float32)
    img_spectrum = np.zeros((num_images,) + crop_size, dtype=np.complex64)

    print('Processing input files..')
    for i, fn in enumerate(all_files):
        if i % 100 == 0:
            print('%d / %d ..' % (i, num_images))
        img = np.asarray(PIL.Image.open(fn), dtype=np.float32) / 255.0
        img_primal[i] = img

        img = img[crop_slice]
        spec = np.fft.fft2(img).astype(np.complex64)
        spec = fftshift2d(spec)
        img_spectrum[i] = spec

    print('Saving: %s' % output_file)
    util.save_pkl((img_primal, img_spectrum), output_file)


def genpng(args):
    if args.outdir is None:
        print ('Must specify output directory with --outdir')
        sys.exit(1)
    if args.ixi_dir is None:
        print ('Must specify input IXI-T1 directory with --ixi-dir')
        sys.exit(1)

    mri_directory = args.ixi_dir

    out_directory = args.outdir
    os.makedirs(out_directory, exist_ok=True)

    nii_files = glob.glob(os.path.join(mri_directory, "*.nii.gz"))

    for nii_file in nii_files:
        print('Processing', nii_file) 
        nii_img = nib.load(nii_file)
        name = os.path.basename(nii_file).split(".")[0]
        print("name", name)
        hborder = (np.asarray(OUT_RESOLUTION) - nii_img.shape[0:2]) // 2
        print("Img: ", nii_img.shape, " border: ", hborder)
        # Normalize image to [0,1]
        img = nii_img.get_data().astype(np.float32)
        img = img / np.max(img)
        print('Max value', np.max(img))
        # # Slice along z dimension
        ## select 1 out of 5 slices
        for s in range(slice_min, slice_max, slice_hops):
            slice = img[:, :, s]
            # Pad to output resolution by inserting zeros
            output = np.zeros(OUT_RESOLUTION)
            output[hborder[0] : hborder[0] + nii_img.shape[0], hborder[1] : hborder[1] + nii_img.shape[1]] = slice
            output = np.clip(output, 0.0, 1.0) * 255.0
            output = output.astype(np.uint8)

            # Save to png
            outname = os.path.join(out_directory, "%s_%03d.png" % (name, s))
            PIL.Image.fromarray(output).convert('L').save(outname)

def make_slice_name(basename, slice_idx):
    return basename + ('_%03d.png' % slice_idx)

def genpkl(args):
    if args.png_dir is None:
        print ('Must specify PNG directory directory with --png-dir')
        sys.exit(1)
    if args.pkl_dir is None:
        print ('Must specify PKL output directory directory with --pkl-dir')
        sys.exit(1)

    input_train_files = []
    input_valid_files = []
    for base in train_basenames:
        for sidx in range(slice_min, slice_max, slice_hops):
            input_train_files.append(os.path.join(args.png_dir, make_slice_name(base, sidx)))
    for base in valid_basenames:
        for sidx in range(slice_min, slice_max, slice_hops):
            input_valid_files.append(os.path.join(args.png_dir, make_slice_name(base, sidx)))
    print ('Num train samples', len(input_train_files))
    print ('Num valid samples', len(input_valid_files))
    preprocess_mri(input_files=input_train_files, output_file=os.path.join(args.pkl_dir, 'ixi_train.pkl'))
    preprocess_mri(input_files=input_valid_files, output_file=os.path.join(args.pkl_dir, 'ixi_valid.pkl'))

def extract_basenames(lst):
    s = set()
    name_re = re.compile('^(.*)-T1_[0-9]+.png')
    for fname in lst:
        m = name_re.match(os.path.basename(fname))
        if m:
            s.add(m[1])
    return sorted(list(s))

examples='''examples:

  # Convert the IXI-T1 dataset into a set of PNG image files:
  python %(prog)s genpng --ixi-dir=~/Downloads/IXI-T1 --outdir=datasets/ixi-png

  # Convert the PNG image files into a Python pickle for use in training:
  python %(prog)s genpkl --png-dir=datasets/ixi-png --pkl-dir=datasets
'''

def main():
    parser = argparse.ArgumentParser(
        description='Convert the IXI-T1 dataset into a format suitable for network training',
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(help='Sub-commands')
    parser_genpng = subparsers.add_parser('genpng', help='IXI nifti to PNG converter (intermediate step)')
    parser_genpng.add_argument('--ixi-dir', help='Directory pointing to unpacked IXI-T1.tar')
    parser_genpng.add_argument('--outdir', help='Directory where to save .PNG files')
    parser_genpng.set_defaults(func=genpng)

    parser_genpkl = subparsers.add_parser('genpkl', help='PNG to PKL converter (used in training)')
    parser_genpkl.add_argument('--png-dir', help='Directory containing .PNGs saved by with the genpng command')
    parser_genpkl.add_argument('--pkl-dir', help='Where to save the .pkl files for train and valid sets')
    parser_genpkl.set_defaults(func=genpkl)

    args = parser.parse_args()
    if 'func' not in args:
        print ('No command given.  Try --help.')
        sys.exit(1)
    args.func(args)

if __name__ == "__main__":
    main()
