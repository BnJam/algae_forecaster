#!/usr/bin/env python3
import os
#import argparse
from PIL import Image
import numpy as np

" PoolScaler "
" Paul Molina-Plant 2019 "
" PoolScaler is a script for converting a dataset of very large images into a"
" larger set of scaled images. The algorithm takes an image of size M x N "
" and a scaling factor S, and produces S^2 m x n subimages. Each sub image is "
" samples every Sth pixel in the X and Y directions to produce an M/S x N/S "
" image with a similar pixel distribution to the original image. When M or N "
" are not divisible by S, the appropriate 0 padding is added around the "
" borders. "

class ImgScaler():
    
    def __init__(self, scale, src, out):
        self.scale = scale
        self.src = src
        self.ext = ext
        self.out = out
        self.main(self.scale, self.src, self.out)


    def padding(self, shape, scale):
        " calculates the padding to be added for a dimension for this scale "
        " shape := dimensions of image (x, y, channels) "
        " scale := the desired scaling factor "
        " returns : tuple(x, y) of the required padding for each dimension"
        x0, y0, _ = shape
        x1, y1 = x0//scale, y0//scale
        xdiff, ydiff = x0 - x1*scale, y0 - y1*scale

        if xdiff > 0:
            x1 += max(xdiff//scale, 1)
            xpad = max(x1//2, 1)
        else:
            xpad = 0

        if ydiff > 0:
            y1 += max(ydiff//scale, 1)
            ypad = max(y1//2, 1)
        else:
            ypad = 0

        return [(0, xpad), (0, ypad), (0, 0)]


    def poolscale(self, filename, scale, out):
        " scales and augments image data by convering an M x N image into "
        " scale**2 m x n images, where m and n are M and N divided by scale "
        " with some additional padding "
        " src := the directory to rescale the contents of"
        " filename := the name of the image file"
        " scale := the pooling scale"
        " out := where to save the generated image "
        " Open an image and dump to numpy array "
        img = Image.open(filename)
        # print(pic.format, pic.size, pic.mode)
        mode = img.mode
        format = img.format
        img = np.asarray(img)

        " Calculate and add the padding necessary for this scaling factor "
        pad = padding(img.shape, scale)
        img = np.pad(img, pad, 'constant', constant_values=0)

        " Generate the scaled images by applying scale x scale pooling. "
        imgs_out = [img[i::scale, j::scale]
                    for i in range(0, scale) for j in range(0, scale)]

        " Convert our new images into the original format and save to disk "
        imgs_out = [Image.fromarray(img, mode=mode) for img in imgs_out]
        for x, p in enumerate(imgs_out):
            fname = os.path.splitext(os.path.basename(filename))
            pname = os.path.join(out, fname[0] + '_' + str(x) + fname[1])
            p.save(pname, format)


    def main(self, scale, src, out):
        " For every file in dir, apply poolscale"
        files = src#[f for f in os.listdir(src) if f.endswith('.' + ext)]
        total = len(files)

        " Process each file "
        " [TODO: DISTRIBUTE] "
        for n, file in enumerate(files):
            poolscale(file, scale, out)
            print("%d/%d : %s\n" % (n+1, total, file))
        print("done")

"""
" Command line args "
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Augment and scale the pngs in source directory.')
    parser.add_argument('scale', type=int,
                        help='the pooling-scale factor to apply.')
    parser.add_argument('-src', type=str, default='./',
                        help='the directory to sample images from')
    parser.add_argument('-out', type=str, default='./gen',
                        help='the directory to save the generated images to')
    parser.add_argument('-ext', type=str, default='png',
                        help='the extension of images to rescale.')
    args = parser.parse_args()
    main(args.scale, args.src, args.ext, args.out)
"""