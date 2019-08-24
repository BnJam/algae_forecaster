import local_worker
import time
import glob
import local_config as config_args

class Client():
    def __init__(self, config):
        self.name = "client"
        print(self.name)
        self.phase = ["GREYSCALE", "SCALE", "TRAIN", "PREDICT"]
        
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
        for n in range(numWorkers):
            self.workerList.append(local_worker.Worker(str(n)))


    def switch(self, config):
        if config.mode == "grey":
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
        self.taskList = glob.glob(src+"/*."+ext)
        print(self.taskList)
    
    def clearWorkerBuff(self):
        for w in self.workerList:
                print("CLEARING WORKER BUFFER")
                try: 
                    # Try to deliver a task to a worker
                    w.clearBuffer()
                except:
                    print("Could not clear buffer")
        print("CLEARED WORKERS BUFFER")
        
    def loadTasks(self):
        print("LOADING TASKS")
        while len(self.taskList) != 0:
            currTask = str(self.taskList[0])
            for w in self.workerList:
                if len(self.taskList) == 0:
                    break
                print("LOADING TASK "+currTask)
                try: 
                    # Try to deliver a task to a worker
                    w.loadTask(self.taskList.pop(0))
                except:
                    print("could not load "+currTask)
        print("LOADED ALL TASKS")
        
    def greyscale(self, config):
        print("PREPROCESSING: GREYSCALE")
        self.connect(config.numWorker)
        self.getTasks(config.gsrc, "nc")
        self.loadTasks()
        for w in self.workerList:
            try: 
                # Try to deliver a task to a worker
                w.greyscale(config)
            except:
                print("Could not greyscale")
        
    def scaleImgs(self, config):
        print("PREPROCESSING: SCALE")
        self.connect(config.numWorker)
        self.clearWorkerBuff()
        self.getTasks(config.ssrc, "png")
        self.loadTasks()
        for w in self.workerList:
            try: 
                # Try to deliver a task to a worker
                w.imgScale(config)
            except:
                print("Could not get worker to scale images")


    def train(self, config):
        pass
        
    def predict(self, config):
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
        file = open("log/master_log/"+self.name+".txt", "a+")
        for i in args:
            file.write(i + "\n")
        file.close()


if __name__ == "__main__":
    print("STARTING CLIENT")

    config, unparsed = config_args.get()
    # Verify all arguments are parsed before continuing.
    if len(unparsed) == 0:
        mr = Client(config)
    else:
        " Unparsed arguments "
        config_args.print_usage()

    
