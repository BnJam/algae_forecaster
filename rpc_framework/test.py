import glob

self.taskList = glob.glob("../OLCI/png/*.png")
#for file in glob.glob("png/*.png"):
#    self.taskList.append(file)
print(self.taskList)