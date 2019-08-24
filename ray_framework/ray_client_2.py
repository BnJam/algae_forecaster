import ray
import ray_worker
import time
import glob
import argparse
import netCDF4 as nc
from PIL import Image
import numpy as np

class Client():
    def __init__(self, func, scale, src, out, ext, numWorker):
        self.name = "client"
        print(self.name)
        self.phase = ["GREYSCALE", "SCALE", "TRAIN", "PREDICT"]
        
        self.func = func
        self.scale = scale
        self.src = src
        self.out = out
        self.ext = ext
        self.numWorker = numWorker
       
        self.taskList = []
        self.address = "localhost"
        self.port = []
        for i in range(10):
            self.port.append(12345+i)  
        self.switch(self.func)
        #self.killWorkers()
        #self.getTasks()
        #self.loadTasks()
        #self.wait()

    def __del__(self):
        print("client ending")

    """
    Prints out connections lists
    """
    def debug(self):
        print("\nWORKER LIST\n")
        print(self.workerList)
        print("\nCONNECTED LIST\n")
        print(self.connected)
        print("\nNOT CONNECTED LIST\n")
        print(self.notConnected)

    """
    Removes a value is it exists in a list
    """
    def removeValue(self, v, l):
        if v in l:
            l.remove(v)
    
    """
    Adds a value if it is not in a list
    """
    def addValue(self, v, l):
        if v not in l:
            l.append(v)


    def switch(self, func):
        if func == "greyscale":
            print("GREYSCALING")
            self.grey_scale(self.src, self.out, self.ext)
        elif func == "scale":
            self.scaleImgs(self.scale, self.src, self.out, self.ext)
        elif func == "train":
            self.train()
        elif func == "predict":
            self.predict()
        else:
            print("invalid")
            return 0

    """
    Get input file paths
    """
    def getTasks(self, src, ext):
        print("GETTING TASKS")
        self.taskList = glob.glob(src+"/*."+ext)
        #for file in glob.glob("png/*.png"):
        #    self.taskList.append(file)
        print(self.taskList)

        
    def grey_scale(self, src, out, ext):
        print("PREPROCESSING: GREYSCALE")
        #self.local_log("PREPROCESSING: GREYSCALE")
        self.getTasks(src, "nc")
        
        futures = [self.greyscale.remote(self, f, src, out) for f in self.taskList]
        for fy in futures:
            ray.get(fy)
        
    

    def scaleImgs(self, scale, src, out, ext, numproc):
        print("PREPROCESSING: SCALE")
        #self.local_log("PREPROCESSING: SCALE")
        self.clearWorkerBuff()
        self.getTasks(src, "png")
        self.loadTasks()
        #for w in self.workerList:
        try: 
            print("issuing scaling command")
            # Try to deliver a task to a worker
            futures = [w.imgScale.remote(scale, out, ext, numproc) for w in self.workerList]
        except:
            print("Could not get worker to scale images")
        for f in futures:
            print(ray.get(f))
        
    def train(self):
        pass

    def predict(self):
        pass
        
    def wait(self):
        while 1:
            time.sleep(0.1)
            #self.checkConnect()
            #self.loadTasks()
   
    def killWorkers(self):
        for w, p in self.workerList.items():
            try: 
                rpyc.async_(w.root.killServer)
            except:
                self.addValue(p, self.notConnected)
                self.removeValue(p, self.connected)
                self.workersToRemove.append(w)
                print("notConnected port: "+str(p))
        # Bumped outside loop since dicts cannot change len at runtime
        for w in self.workersToRemove:
            self.workerList.pop(w)
        self.workersToRemove = [] # Empty list to prevent violations
            
    def local_log(self, *args):
        file = open("../log/master_log/"+self.name+".txt", "a+")
        for i in args:
            file.write(i + "\n")
        file.close()

    @ray.remote
    def imgScale(self, scale, out, ext, numproc):
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
        print("IMGSCALE")       
        #pool = ProcessPoolExecutor(numproc)
        #with ProcessPoolExecutor(max_workers=4) as ex:
        for f in self.taskList:
            #pool.submit(self.poolscale, f, scale, out, ext)
            print("SUBMITTED "+f)
            self.poolscale(f, scale, out, ext)  
        #print("done scaling image

    #@ray.remote(num_cpu=self.numWorkers)
    @ray.remote
    def greyscale(self, f, src, out):
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
            #f = nc.Dataset(os.path.join(src, filename), 'r', format="NETCDF4")
            try:
                f = nc.Dataset(filename)
            except Exception as e:
                print("Could not open nc file: "+str(e))
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
                    #print(str(x)+" \\ "+str(width)+ " || "+str(y)+ " \\ "+str(height))
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
        " For every file in src, process "
        try:
            process(src, f, out)
        except Exception as e:
            print("FAILED "+str(e))

        print("done greyscaling")
        
        
        return "done greyscaling"


if __name__ == "__main__":
    print("STARTING CLIENT")
    
    ray.init()
    
    parser = argparse.ArgumentParser(
        description='Client (edge application) to submit jobs to workers.')
    parser.add_argument('--func' ,type=str,
                        help='the function to run.')
    parser.add_argument('--scale', type=int, default=2,
                        help='the pooling-scale factor to apply.')
    parser.add_argument('--src', type=str, default='../',
                        help='the directory to sample images from')
    parser.add_argument('--out', type=str, default='../gen',
                        help='the directory to save the generated images to')
    parser.add_argument('--ext', type=str, default='png',
                        help='the extension of images to rescale.')
    parser.add_argument('--numWorker', type=int, default=4,
                        help='Number of workers to create')
    args = parser.parse_args()
    mr = Client(args.func, args.scale, args.src, args.out, args.ext, args.numWorker)
