import netCDF4 as nc
import numpy as np
import os
import argparse
from PIL import Image

" NcToGrey "
" This script open netCDF4 files and converts them to greyscale png images "
" by mapping non-zero values to the line interpolated between "
" (-1.0, 0) and (1.602, 255). "


def linear_sol(x_min, x_max, y_min, y_max, x_intercept=None):
    " helper function for solving linear equations on a bounded segment"
    if (x_intercept is None):
        x_intercept = x_min
    # y = mx + b
    m = (y_max - y_min) / (x_max - x_min)
    b = -m * x_intercept

    def helper(x):
        if (x <= x_min):
            return int(y_min)
        else:
            return min(int(y_max), int(m * x + b) + 1)
    return helper


def process(src, filename, out):
    " a helper function for mapping chisqr values to interval [0, 255]. "
    to_grey = linear_sol(x_min=-1.0, x_max=1.602, y_min=0, y_max=255)

    " Open the source file "
    f = nc.Dataset(os.path.join(src, filename), 'r', format="NETCDF4")

    " Load parameters from .nc "
    CHL = f.variables['logchl'][:].data
    CHL = np.asarray(CHL)
    width = CHL.shape[1]
    height = CHL.shape[0]

    " Create empty image to fill in "
    img = Image.new('L', (width, height), 0)
    pixels = img.load()

    " Fill in each pixel "
    for x in range(width):
        for y in range(height):
            val = CHL[y, x]
            pixels[x, y] = 255 - (to_grey(val) if val != 0 else 0)

    " Save the output to file "
    fname = os.path.splitext(os.path.basename(filename))
    pname = os.path.join(out, fname[0] + '.png')
    print(pname)
    img.save(pname)


def main(src, out):
    " For every file in src, process "
    files = [f for f in os.listdir(src) if f.endswith('.nc')]
    total = len(files)
    failed = 0

    " Process each file in directory "
    " [TODO: DISTRIBUTE THIS LOOP] "
    for n, filename in enumerate(files):
        try:
            process(src, filename, out)
            print("%d/%d: %s success\n" % (n+1, total, filename))
        except:
            failed += 1
            print("%d/%d: %s failed [%d]\n" % (n+1, total, filename, failed))

    print("done")
    if (failed > 0):
        print("failed to convert %d files" % (failed))


" Command line args "
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert netCDF4 to grayscale images')
    parser.add_argument('-src', type=str, default='./',
                        help='the directory to source .nc files from')
    parser.add_argument('-out', type=str, default='./gen',
                        help='the directory to save the generated images to')
    args = parser.parse_args()
    main(args.src, args.out)
