import ray
import time
import sys
from PIL import Image
import numpy as np
import argparse
import netCDF4 as nc
import psutil

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import os

from tensorboardX import SummaryWriter
from model import AutoEncoder
from tqdm import tqdm, trange

# Uncomment line below to use this class as a Ray Actor
#@ray.remote(num_cpus=4)
class Worker():

    def __init__(self, name):
        self.name = "worker-"+str(name)
        print(self.name)
        self.active = True
        self.taskBuff = []

    def clearBuffer(self):
        self.taskBuff = []

    
    def loadTask(self, task):
        print(self.name+" LOADING TASK "+task)
        self.taskBuff.append(task)
        #print(self.taskBuff)
        return True
    
    @ray.remote
    def greyscale(self, config):
        print("WORKER GREYSCALE")
        futures = [self.process.remote(self, config.gsrc, f, config.gout) for f in self.taskBuff]
        print("LOADED GREYSCALE PROCESSES")
        for fy in futures:
            print(ray.get(fy))
        

        print(self.name+" done greyscaling")
        
        return self.name+" done greyscaling"

    @ray.remote
    def imgScale(self, config):
        print("IMGSCALE")       
        futures = [self.poolscale.remote(self, f, config.scale, config.sout, config.sext) for f in self.taskBuff]
        print("LOADED IMGSCALE PROCESSES")
        for fy in futures:
            print(ray.get(fy))

    @ray.remote
    def train(self, config):
        print("WORKER "+self.name+" TRAINING MODEL")
        ray.get(self.train.remote(self, config))

    @ray.remote
    def predict(self, config):
        print("WORKER "+self.name+" TESTING MODEL")
        ray.get(self.test.remote(self, config))
    

    def killServer(self):
        server.close()
        self.__del__()
   
    def local_log(self, *items):
        file = open("../log/worker_log/"+self.name+".txt", "a+")
        for i in items:
            file.write(i + "\n")
        file.close()

    @ray.remote
    def poolscale(self, filename, scale, out, ext):
        print("POOLSCALE")
        def padding(shape, scale):
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
        pad = padding(img.shape, scale)
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
    
    @ray.remote
    def process(self, src, filename, out):
        print("GREYSCALE PROCESS...")
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
        " a helper function for mapping chisqr values to interval [0, 255]. "
        to_grey = linear_sol(x_min=-1.0, x_max=1.602, y_min=0, y_max=255)
        " Open the source file "
        #f = nc.Dataset(os.path.join(src, filename), 'r', format="NETCDF4")
        try:
            f = nc.Dataset(filename)
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
            
            img.convert('L')
            try:
                " Save the output to file "
                #fname = os.path.splitext(os.path.basename(filename))
                #pname = os.path.join(out, fname[0] + '.png')
                fname = filename.split('/')[2]
                fname = fname.split('.')[0]
                pname = out+"/"+fname+".png"
                print(pname)
                img.save(pname)
            except Exception as e:
                print("Could not save file. "+str(e))
        except Exception as e:
            print("Could not open nc file: "+str(e))
    
    @ray.remote(num_gpus=1)
    def train(self, config):
        """Training routine"""
        # Initialize datasets for both training and validation
        train_data = torchvision.datasets.ImageFolder(
            root=os.path.join(config.data_dir, "train"),
            transform=torchvision.transforms.ToTensor()
        )
        valid_data = torchvision.datasets.ImageFolder(
            root=os.path.join(config.data_dir, "valid"),
            transform=torchvision.transforms.ToTensor()
        )
    
        # Create data loader for training and validation.
        tr_data_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=config.batch_size,
            num_workers=config.numWorker,
            shuffle=True
        )
        va_data_loader = torch.utils.data.DataLoader(
            dataset=valid_data,
            batch_size=config.batch_size,
            num_workers=config.numWorker,
            shuffle=False
        )
    
        # Create model instance.
        #model = Model()
        model = AutoEncoder()

        # Move model to gpu if cuda is available
        if torch.cuda.is_available():
            model = model.cuda()
        # Make sure that the model is set for training
        model.train()
    
        # Create loss objects
        data_loss = nn.MSELoss()
    
        # Create optimizier
        optimizer = optim.Adam(model.parameters(), lr=config.learn_rate)
        # No need to move the optimizer (as of PyTorch 1.0), it lies in the same
        # space as the model
    
        # Create summary writer
        tr_writer = SummaryWriter(
            log_dir=os.path.join(config.log_dir, "train"))
        va_writer = SummaryWriter(
            log_dir=os.path.join(config.log_dir, "valid"))
    
        # Create log directory and save directory if it does not exist
        if not os.path.exists(config.log_dir):
            os.makedirs(config.log_dir)
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
    
        # Initialize training
        iter_idx = -1  # make counter start at zero
        best_va_acc = 0  # to check if best validation accuracy
        # Prepare checkpoint file and model file to save and load from
        checkpoint_file = os.path.join(config.save_dir, "checkpoint.pth")
        bestmodel_file = os.path.join(config.save_dir, "best_model.pth")
    
        # Check for existing training results. If it existst, and the configuration
        # is set to resume `config.resume==True`, resume from previous training. If
        # not, delete existing checkpoint.
        if os.path.exists(checkpoint_file):
            if config.resume:
                # Use `torch.load` to load the checkpoint file and the load the
                # things that are required to continue training. For the model and
                # the optimizer, use `load_state_dict`. It's actually a good idea
                # to code the saving part first and then code this part.
                print("Checkpoint found! Resuming")
                # Read checkpoint file.
    
                # Fix gpu -> cpu bug
                compute_device = 'cuda' if torch.cuda.is_available() else 'cpu'
                load_res = torch.load(checkpoint_file, map_location=compute_device)
    
                # Resume iterations
                iter_idx = load_res["iter_idx"]
                # Resume best va result
                best_va_acc = load_res["best_va_acc"]
                # Resume model
                model.load_state_dict(load_res["model"])
    
                # Resume optimizer
                optimizer.load_state_dict(load_res["optimizer"])
                # Note that we do not resume the epoch, since we will never be able
                # to properly recover the shuffling, unless we remember the random
                # seed, for example. For simplicity, we will simply ignore this,
                # and run `config.num_epoch` epochs regardless of resuming.
            else:
                os.remove(checkpoint_file)
    
        # Training loop
        for epoch in range(config.num_epoch):
            # For each iteration
            prefix = "Training Epoch {:3d}: ".format(epoch)
    
            for data in tqdm(tr_data_loader, desc=prefix):
                # Counter
                iter_idx += 1
    
                # Split the data
                # x is img, y is label
                x, y = data 
                #print(x)
                # Send data to GPU if we have one
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()
    
                # Apply the model to obtain scores (forward pass)
                logits = model.forward(x)
                # Compute the loss
                loss = data_loss(logits, x.float())
                print("LOSS: "+str(loss))
                # Compute gradients
                loss.backward()
                # Update parameters
                optimizer.step()
                # Zero the parameter gradients in the optimizer
                optimizer.zero_grad()
    
                # Monitor results every report interval
                if iter_idx % config.rep_intv == 0:
                    # Compute accuracy (No gradients required). We'll wrapp this
                    # part so that we prevent torch from computing gradients.
                    with torch.no_grad():
                        pred = torch.argmax(logits, dim=1)
                        acc = torch.mean(torch.eq(pred.view(x.size()), x).float()) * 100.0
                    # Write loss and accuracy to tensorboard, using keywords `loss`
                    # and `accuracy`.
                    tr_writer.add_scalar("loss", loss, global_step=iter_idx)
                    tr_writer.add_scalar("accuracy", acc, global_step=iter_idx)
                    
                    print("LOSS: "+str(loss))
                    print("ACC: "+str(acc))
                    
                    # Save
                    torch.save({
                        "iter_idx": iter_idx,
                        "best_va_acc": best_va_acc,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "loss": loss,
                        "epoch": epoch,
                        "acc":acc
                    }, checkpoint_file)
    
                # Validate results every validation interval
                if iter_idx % config.val_intv == 0:
                    # List to contain all losses and accuracies for all the
                    # training batches
                    va_loss = []
                    va_acc = []
                    # Set model for evaluation
                    model = model.eval()
                    for data in va_data_loader:
    
                        # Split the data
                        x, y = data
    
                        # Send data to GPU if we have one
                        if torch.cuda.is_available():
                            x = x.cuda()
                            y = y.cuda()
    
                        # Apply forward pass to compute the losses
                        # and accuracies for each of the validation batches
                        with torch.no_grad():
                            # Compute logits
                            logits = model.forward(x)
                            # Compute loss and store as numpy
                            loss = data_loss(logits, x.float())
                            va_loss += [loss.cpu().numpy()]
                            # Compute accuracy and store as numpy
                            pred = torch.argmax(logits, dim=1)
                            acc = torch.mean(torch.eq(pred.view(x.size()), x).float()) * 100.0
                            va_acc += [acc.cpu().numpy()]
                    # Set model back for training
                    model = model.train()
                    # Take average
                    va_loss = np.mean(va_loss)
                    va_acc = np.mean(va_acc)
    
                    # Write to tensorboard using `va_writer`
                    va_writer.add_scalar("loss", va_loss, global_step=iter_idx)
                    va_writer.add_scalar("accuracy", va_acc, global_step=iter_idx)
                    # Check if best accuracy
                    if va_acc > best_va_acc:
                        best_va_acc = va_acc
                        # Save best model using torch.save. Similar to previous
                        # save but at location defined by `bestmodel_file`
                        torch.save({
                            "iter_idx": iter_idx,
                            "best_va_acc": best_va_acc,
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "loss":loss,
                            "acc":acc
                        }, bestmodel_file)
    
    @ray.remote
    def test(self, config):
        """Testing routine"""
        # Initialize Dataset for testing.
        test_data = torchvision.datasets.ImageFolder(
            root=os.path.join(config.data_dir, "test"),
            transform=torchvision.transforms.ToTensor()
        )

        # Create data loader for the test dataset with two number of workers and no
        # shuffling.
        te_data_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=config.batch_size,
            num_workers=config.numWorker,
            shuffle=False
        )

        # Create model
        model = AutoEncoder()

        # Move to GPU if you have one.
        if torch.cuda.is_available():
            model = model.cuda()

        # Create loss objects
        data_loss = nn.MSELoss()

        # Fix gpu -> cpu bug
        compute_device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load our best model and set model for testing
        load_res = torch.load(os.path.join(config.save_dir, "best_model.pth"),
                                map_location=compute_device)

        model.load_state_dict(load_res["model"])

        model.eval()

        # Implement The Test loop
        prefix = "Testing: "
        te_loss = []
        te_acc = []
        for data in tqdm(te_data_loader, desc=prefix):
            # Split the data
            x, y = data

            # Send data to GPU if we have one
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # Don't invoke gradient computation
            with torch.no_grad():
                # Compute logits
                logits = model.forward(x)
                # Compute loss and store as numpy
                loss = data_loss(logits, x.float())
                te_loss += [loss.cpu().numpy()]
                # Compute accuracy and store as numpy
                pred = torch.argmax(logits, dim=1)
                acc = torch.mean(torch.eq(pred.vewi(x.size()), x).float()) * 100.0
                te_acc += [acc.cpu().numpy()]

        # Report Test loss and accuracy
        print("Test Loss = {}".format(np.mean(te_loss)))
        print("Test Accuracy = {}%".format(np.mean(te_acc)))
        
    