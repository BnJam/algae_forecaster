import ray
import ray_worker
import time
import glob
import ray_config as config_args

class Task():
    def __init__(self, idNum, path, attempts, operation):
        self.idNum = idNum # id of task
        self.path = path # path to file 
        self.attempts = attempts #number of tries for a failed task
        self.operation = operation # operation for the worker to perform
    def __del__(self):
        print("Erasing task "+str(self.idNum))

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

    """
    Operation switch based on argument configs
    """
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
    Get input file paths and save them into a lsit
    """
    def getTasks(self, src, ext):
        print("GETTING TASKS")
        self.taskList = glob.glob(src+"/*."+ext)
        print(self.taskList)
    
    """
    Invoke the clear buffer operation on the workers
    """
    def clearWorkerBuff(self):
        for w in self.workerList:
                print("CLEARING WORKER BUFFER")
                try: 
                    w.clearBuffer()
                except:
                    print("Could not clear buffer")
        print("CLEARED WORKERS BUFFER")
        
    """
    serially load tasks one at a time into 
    workers task buffers

    TODO make this a response operation to task request from workers
    """
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
        
    """
    Invoke workers remote greyscaling handler
    """
    def greyscale(self, config):
        print("PREPROCESSING: GREYSCALE")
        self.connect(config.numWorker)
        self.clearWorkerBuff()
        self.getTasks(config.gsrc, "nc")
        self.loadTasks()
        futures = [w.greyscale.remote(w, config) for w in self.workerList]
        # wait for workers to be complete
        for fy in futures:
            print(ray.get(fy))
       
    
    """
    Invoke workers remote image scaling handler
    """
    def scaleImgs(self, config):
        print("PREPROCESSING: SCALE")
        self.connect(config.numWorker)
        self.clearWorkerBuff()
        self.getTasks(config.ssrc, "png")
        self.loadTasks()
        futures = [w.imgScale.remote(w, config) for w in self.workerList]
        # wait for workers to be complete
        for f in futures:
            print(ray.get(f))

    """
    Invoke workers remote train_model handler
    """    
    def train(self, config):
        print("TRAINING MODEL")
        self.connect(1)
        futures = [w.train_model.remote(w, config) for w in self.workerList]
        # wait for workers to be complete
        for fy in futures:
            ray.get(fy)

    """
    Invoke workers remote predict handler
    """
    def predict(self, config):
        print("TRAINING MODEL")
        self.connect(1)
        futures = [w.predict.remote(w, config) for w in self.workerList]
        # wait for workers to be complete
        for fy in futures:
            ray.get(fy)
        
    """
    Kill all worker objects
    """
    def killWorkers(self):
        for w in self.workerList:
            try: 
                w.kill()
            except Exception as e:
                print("Error killing worker: "+str(e))

    """
    Logging operation
    """    
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