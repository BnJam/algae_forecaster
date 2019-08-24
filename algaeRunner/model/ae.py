import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn

import matplotlib.pyplot as plt
import numpy as np

# Helper functions
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()
    
def sampleshow(batch, limit=5):
    images = batch[:limit]
    imshow(torchvision.utils.make_grid(images))

# Hyper Parameters
batch_size = 100
learning_rate = 1e-3
epochs = 4
epochs = 10 # loops over the dataset
iterations = 100

"""
trainset = torchvision.datasets.ImageFolder(root='./data/',
                                      transform=torchvision.transforms.ToTensor(), 
                                      )
"""
trainset = 

trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=batch_size, 
                                          shuffle=True, 
                                          num_workers=4)
#dataiter = iter(trainloader)

# Model definition
class AutoEncoder(nn.Module):
    def __init__(self, size):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.LSTM(size, 256),
            nn.ReLU(),
            nn.LSTM(256, 128),
            nn.ReLU(),
            nn.LSTM(128, 64),
            nn.ReLU(),
            nn.LSTM(64, 32),
        )
        self.decoder = nn.Sequential(
            nn.LSTM(32, 64),
            nn.ReLU(),
            nn.LSTM(64, 128),
            nn.ReLU(),
            nn.LSTM(128, 256),
            nn.ReLU(),
            nn.LSTM(256, size),
            nn.Sigmoid(),       # compress to a range (0, 1)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Training
model = AutoEncoder(6493*7823)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

shape = torch.Size([batch_size, 1, 28, 28])

model.train()
for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get image, discard label
        inputs, _ = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs.reshape(batch_size, 256))
        outputs = outputs.reshape(shape)

        loss = criterion(outputs, inputs)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % iterations == iterations-1:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / batch_size))
            running_loss = 0.0

print('Finished Training')

# Save trained model
torch.save(model.state_dict(), "ae")

# Evaluate model
saved = AutoEncoder(6493*7823)
saved.load_state_dict(torch.load("ae"))
saved.eval()

sample_size = 5 # how many sample images to display

with torch.no_grad():
    original, _ = dataiter.next()
    sampleshow(original, sample_size)
    generated = saved(original.reshape(batch_size, -1)).reshape(shape)
    sampleshow(generated, sample_size)
    loss = [criterion(y, x) for (x, y) in zip(original, generated)]
    print(loss[:5])