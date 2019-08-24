import rpyc
import time
import glob
import argparse

class Client():
    def __init__(self, func, scale, src, out, ext, numproc):
        self.name = "client"
        print(self.name)
        self.phase = ["GREYSCALE", "SCALE", "TRAIN", "PREDICT"]
        
        self.func = func
        self.scale = scale
        self.src = src
        self.out = out
        self.ext = ext
        self.numproc = numproc
        
        self.workerList = {}
        self.connected = []
        self.bgSrvThreads = {}
        self.notConnected = []
        self.workersToRemove = []
        self.taskList = []
        self.address = "localhost"
        self.port = []
        for i in range(10):
            self.port.append(12345+i)  
        self.connect()
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

    """
    Initial connection to port list for active servers
    """
    def connect(self):
        for p in self.port:
            try:
                c = rpyc.connect(self.address, port=p)
                #self.bgSrvThreads[c] = rpyc.BgServingThread(c)
                self.workerList[c] = p
                self.addValue(p, self.connected)
                self.local_log("worker-"+str(p)+" READY FOR WORK")
            except ConnectionError:
                self.addValue(p, self.notConnected)
                #print("Could not connect "+str(p)) 
        print(self.connected)

    """
    Reattempts connection to non-connected port numbers
    Done periodically in case servers are re/started
    """
    def checkConnect(self):
        for p in self.notConnected:
            try:
                c = rpyc.connect(self.address, port=p)
                self.removeValue(p, self.notConnected)
                self.addValue(p, self.connected)
                self.workerList[c] = p
                print("worker-"+p+" READY FOR WORK")
            except:
                #print("Could not connect "+str(p)) 
                pass
    
    def switch(self, func):
        if func == "greyscale":
            self.greyscale(self.src, self.out, self.ext)
        elif func == "scale":
            self.scaleImgs(self.scale, self.src, self.out, self.ext, self.numproc)
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
        self.taskList = glob.glob(src+"/*."+ext)
        print(self.taskList)
    
    def clearWorkerBuff(self):
        for w, p in self.workerList.items():
                print("CLEARING WORKER BUFFER"+str(p))
                try: 
                    # Try to deliver a task to a worker
                    rpyc.async_(w.root.clearBuffer)
                except:
                    self.addValue(p, self.notConnected)
                    self.removeValue(p, self.connected)
                    self.workersToRemove.append(w)
                    #print("notConnected port: "+str(p))
        print("LOADED ALL TASKS")
        # Bumped outside loop since dicts cannot change len at runtime
        for w in self.workersToRemove:
            self.workerList.pop(w)
        self.workersToRemove = [] # Empty list to prevent violations


    def loadTasks(self):
        print("LOADING TASKS")
        while len(self.taskList) != 0:
            for w, p in self.workerList.items():
                print("LOADING TASK "+str(self.taskList[0]))
                try: 
                    # Try to deliver a task to a worker
                    rpyc.async_(w.root.loadTask)(self.taskList.pop(0))
                    print("here")
                except:
                    self.addValue(p, self.notConnected)
                    self.removeValue(p, self.connected)
                    self.workersToRemove.append(w)
                    print("notConnected port: "+str(p))
        print("LOADED ALL TASKS")
        # Bumped outside loop since dicts cannot change len at runtime
        for w in self.workersToRemove:
            self.workerList.pop(w)
        self.workersToRemove = [] # Empty list to prevent violations

        #self.checkConnect()
        #self.scale()
        
    def greyscale(self, src, out, ext):
        print("PREPROCESSING: GREYSCALE")
        self.local_log("PREPROCESSING: GREYSCALE")
        self.getTasks(src, "nc")
        self.loadTasks()
        for w, p in self.workerList.items():
            try: 
                # Try to deliver a task to a worker
                rpyc.async_(w.root.greyscale)()
            except:
                self.addValue(p, self.notConnected)
                self.removeValue(p, self.connected)
                self.workersToRemove.append(w)
                #print("notConnected port: "+str(p))

        # Bumped outside loop since dicts cannot change len at runtime
        for w in self.workersToRemove:
            self.workerList.pop(w)
        self.workersToRemove = [] # Empty list to prevent violations
        self.wait()
    
    def scaleImgs(self, scale, src, out, ext, numproc):
        print("PREPROCESSING: SCALE")
        #self.local_log("PREPROCESSING: SCALE")
        self.clearWorkerBuff()
        self.getTasks(src, "png")
        self.loadTasks()
        for w, p in self.workerList.items():
            try: 
                # Try to deliver a task to a worker
                rpyc.async_(w.root.imgScale)(scale, out, ext, numproc)
            except:
                self.addValue(p, self.notConnected)
                self.removeValue(p, self.connected)
                self.workersToRemove.append(w)
                print("notConnected port: "+str(p))

        # Bumped outside loop since dicts cannot change len at runtime
        for w in self.workersToRemove:
            self.workerList.pop(w)
        self.workersToRemove = [] # Empty list to prevent violations
        self.wait()

    def train(self):
        pass

    def predict(self):
        pass
        
    def wait(self):
        while 1:
            time.sleep(0.1)
            self.checkConnect()
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


if __name__ == "__main__":
    print("STARTING CLIENT")
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
    parser.add_argument('--numproc', type=int, default=4,
                        help='Number of sub processes each worker will have')
    args = parser.parse_args()
    mr = Client(args.func, args.scale, args.src, args.out, args.ext, args.numproc)
