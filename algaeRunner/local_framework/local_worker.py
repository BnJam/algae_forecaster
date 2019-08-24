import time
import sys
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import numpy as np
import argparse
import netCDF4 as nc

class Worker():

    def __init__(self, name):
        self.name = "worker-"+name
        #self.port = sys.argv[1]
        print(self.name)
        #self.local_log(self.name)
        self.active = True
        self.taskBuff = []
        #self.runTask()
        
    #def __del__(self):
    #    print("End worker-"+self.port)

    def clearBuffer(self):
        self.taskBuff = []

    def loadTask(self, task):
        print("WORKER LOADING TASK "+task)
        self.taskBuff.append(task)
        #print(self.taskBuff)
        return True

    def greyscale(self, config):
        if len(self.taskBuff) == 0:
            return 0
        print("WORKER GREYSCALE")
        failed = 0
        pool = ProcessPoolExecutor(config.gnumproc)
        for f in self.taskBuff:
            try:
                pool.submit(self.process, config.gsrc, f, config.gout)
                print("SUBMITTED "+f)
            except EXception as e:
                failed += 1
                print("FAILED: "+str(e))

        print("done greyscaling")
        if (failed > 0):
            print("failed to convert %d files" % (failed))
        
        return "done greyscaling"

    def imgScale(self, config):
        print("WORKER IMGSCALE")       
        pool = ProcessPoolExecutor(config.snumproc)
        #with ProcessPoolExecutor(max_workers=4) as ex:
        for f in self.taskBuff:
            pool.submit(self.poolscale, f, config.scale, config.sout, config.sext)
            print("SUBMITTED "+f)

            
        print("done scaling images from WORKER-"+str(self.name))
        return

    def train(self):
        pass

    def predict(self):
        pass
    
    def killServer(self):
        server.close()
        self.__del__()
   
    def local_log(self, *items):
        file = open("../log/worker_log/"+self.name+".txt", "a+")
        for i in items:
            file.write(i + "\n")
        file.close()

    def padding(self, shape, scale):
        " calculates the padding to be added for a dimension for this scale "
        " shape := dimensions of image (x, y, channels) "
        " scale := the desired scaling factor "
        " returns : tuple(x, y) of the required padding for each dimension"
        x0, y0 = shape[0], shape[1]
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

        dims = len(shape)
        if (dims == 2):
            out = [(0, xpad), (0, ypad)]
        elif (dims == 3):
            out = [(0, xpad), (0, ypad), (0, 0)]
        else:
            raise

        return out


    def poolscale(self, filename, scale, out, ext):
        print("POOLSCALE")
        " scales and augments image data by convering an M x N image into "
        " scale**2 m x n images, where m and n are M and N divided by scale "
        " with some additional padding "
        " src := the directory to rescale the contents of"
        " filename := the name of the image file"
        " scale := the pooling scale"
        " out := where to save the generated image "
        " Open an image and dump to numpy array "
        try:
            img = Image.open(filename)
            print("OPENED IMG FILE")
        except:
            print("COULD NOT OPEN IMG "+filename)
        #print(pic.format, pic.size, pic.mode)
        mode = img.mode
        format = img.format
        
        img = np.asarray(img)
       
        " Calculate and add the padding necessary for this scaling factor "
        pad = self.padding(img.shape, scale)
        img = np.pad(img, pad, 'constant', constant_values=0)
        x1, y1 = img.shape[0]//scale, img.shape[1]//scale

        " Generate the scaled images by applying scale x scale pooling. "
        imgs_out = [(img[i::scale, j::scale])[:x1, :y1]
                for i in range(0, scale) for j in range(0, scale)]

        " Convert our new images into the original format and save to disk "
        imgs_out = [Image.fromarray(img, mode=mode) for img in imgs_out]

        for x, p in enumerate(imgs_out):
            fname = filename.split('/')[2]
            fname = fname.split('.')[0]
            pname = out+"/"+fname+'_'+str(x)+"."+ext
            #print(pname)
            try:
                p.save(pname, format)
            except:
                print("Could not save file "+pname)
        print("Done scaling images")

    def linear_sol(self, x_min, x_max, y_min, y_max, x_intercept=None):
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
    
    
    def process(self, src, filename, out):
        " a helper function for mapping chisqr values to interval [0, 255]. "
        to_grey = self.linear_sol(x_min=-1.0, x_max=1.602, y_min=0, y_max=255)
        " Open the source file "
        #f = nc.Dataset(os.path.join(src, filename), 'r', format="NETCDF4")
        try:
            f = nc.Dataset(filename)
        except:
            print("Could not open nc file")
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
                #print(str(x) + " // " + str(width) + " " + str(y) + " // "+str(height))
                val = CHL[y, x]
                pixels[x, y] = 255 - (to_grey(val) if val != 0 else 0)

        " Save the output to file "
        #fname = os.path.splitext(os.path.basename(filename))
        #pname = os.path.join(out, fname[0] + '.png')
        fname = filename.split('/')[2]
        fname = fname.split('.')[0]
        pname = out+"/"+fname+".png"
        print(pname)
        img.save(pname)
"""
if __name__ == "__main__":
    print("STARTING WORKER")
    parser = argparse.ArgumentParser(
        description='Worker entity')
    parser.add_argument('--port', type=int,
                        help='port which to be available on')
    args = parser.parse_args()
    server = rpyc.utils.server.ThreadedServer(Worker, port=args.port) #listener_timeout=10
    server.start()
"""