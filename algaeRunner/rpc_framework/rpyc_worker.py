import rpyc
import time
import sys
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import numpy as np
import argparse

class Worker(rpyc.Service):

    def __init__(self, port):
        self.name = "worker"
        self.port = port
        #self.local_log(self.name)
        self.active = True
        self.taskBuff = []
        #self.runTask()
        
    #def __del__(self):
    #    print("End worker-"+self.port)

    def exposed_clearBuffer(self):
        self.taskBuff = []

    def exposed_loadTask(self, task):
        print("LOADING TASK "+task)
        self.taskBuff.append(task)
        print(self.taskBuff)
        return True

    def exposed_greyscale(self, task):
        total = len(files)
        failed = 0

        for f in self.taskBuff:
            try:
                pool.submit(self.process, src, f, out)
                print("SUBMITTED: "+str(f))
            except:
                failed += 1
                print("could not submit task to greyscale: "+str(f))
        print("done submitting tasks")
        if (failed > 0):
            print("failed to convert %d files" % (failed))

    def exposed_imgScale(self, scale, out, ext, numproc):
        print("IMGSCALE")       
        pool = ProcessPoolExecutor(numproc)
        #with ProcessPoolExecutor(max_workers=4) as ex:
        for f in self.taskBuff:
            pool.submit(self.poolscale, f, scale, out, ext)
            print("SUBMITTED "+f)

            
        print("done scaling images from WORKER-"+str(self.port))
        return

    def exposed_train(self):
        pass

    def exposed_predict(self):
        pass
    

    # Exported function for RPC calls
    def exposed_dotask(self, task):
        self.taskBuff.append(task)
        print(self.taskBuff)
        with ProcessPoolExecutor(max_workers=4) as ex:
            ex.submit(self.runTask(), None)
    
    def exposed_killServer(self):
        server.close()
        self.__del__()
    
    # Tester function
    def runTask(self):
        time.sleep(0.1)
        if len(self.taskBuff) != 0:
            while len(self.taskBuff) != 0:
                print("THIS TASK: " + self.taskBuff.pop(0))
                for i in range(5):
                    print(i)
                    time.sleep(1)
    
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
            print(pname)
            try:
                p.save(pname, format)
            except:
                print("Could not save file "+pname)
        print("Done scaling images")

if __name__ == "__main__":
    print("STARTING WORKER")
    parser = argparse.ArgumentParser(
        description='Worker entity')
    parser.add_argument('--port', type=int,
                        help='port which to be available on')
    args = parser.parse_args()
    server = rpyc.utils.server.ThreadedServer(Worker, port=args.port, listener_timeout=10)
    server.start()