import ray
import ray_worker
import time
import glob
import ray_config as config_args

class Client():
    def __init__(self, config):
        self.name = "client"
        print(self.name)
        self.phase = ["GREYSCALE", "SCALE", "TRAIN", "TEST"]
        
        self.workerList = []
        self.taskList = []
        self.switch(config)

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
    def connect(self, numWorkers):
        #self.workerList = [ray_worker.Worker.remote() for _ in range(numWorkers)]
        self.workerList = [ray_worker.Worker(i) for i in range(numWorkers)]

    def switch(self, config):
        if config.mode == "grey":
            print("GREYSCALING")
            self.greyscale(config)
        elif config.mode == "scale":
            self.scaleImgs(config)
        elif config.mode == "train":
            self.train(config)
        elif config.mode == "predict":
            self.predict(config)
        else:
            print("invalid")
            return 0

    """
    Get input file paths
    """
    def getTasks(self, src, ext):
        print("GETTING TASKS")
        self.taskList = glob.glob(src+"/*."+ext)
        print(self.taskList)
    
    def clearWorkerBuff(self):
        for w in self.workerList:
                print("CLEARING WORKER BUFFER")
                try: 
                    w.clearBuffer()
                except:
                    print("Could not clear buffer")
        print("CLEARED WORKERS BUFFER")
        
    def loadTasks(self):
        print("LOADING TASKS")
        while len(self.taskList) != 0:
            currTask = str(self.taskList[0])
            for w in self.workerList:
                print("LOADING TASK "+currTask)
                try: 
                    w.loadTask(self.taskList.pop(0))
                except:
                    print("could not load "+currTask)
        print("LOADED ALL TASKS")
        
    def greyscale(self, config):
        print("PREPROCESSING: GREYSCALE")
        self.connect(config.numWorker)
        self.clearWorkerBuff()
        self.getTasks(config.gsrc, "nc")
        self.loadTasks()
        futures = [w.greyscale.remote(w, config) for w in self.workerList]
        for fy in futures:
            print(ray.get(fy))
       
    

    def scaleImgs(self, config):
        print("PREPROCESSING: SCALE")
        self.connect(config.numWorker)
        self.clearWorkerBuff()
        self.getTasks(config.ssrc, "png")
        self.loadTasks()
        futures = [w.imgScale.remote(w, config) for w in self.workerList]
        for f in futures:
            print(ray.get(f))


        
    def train(self, config):
        print("TRAINING MODEL")
        self.connect(1)
        futures = [w.train.remote(w, config) for w in self.workerList]
        for fy in futures:
            ray.get(fy)

    def predict(self, config):
        print("TRAINING MODEL")
        self.connect(1)
        futures = [w.predict.remote(w, config) for w in self.workerList]
        for fy in futures:
            ray.get(fy)
        
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


if __name__ == "__main__":
    config, unparsed = config_args.get()
    if len(unparsed) == 0:
        print("STARTING CLIENT")
        ray.init()
        mr = Client(config)
    else:
        " Unparsed arguments "
        config_args.print_usage()