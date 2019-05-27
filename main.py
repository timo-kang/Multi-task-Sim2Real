import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import random
from model import network
from loader import SawyerSimDataset

# Hyperparameters
batch_size = 64
num_epochs = 10
input_size = 1024
time_step = 4
learning_rate = 0.0001

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

# https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder
train_dataset = SawyerSimDataset(csv_file='/home/yena/test_label.csv',
                                 root_dir='/home/yena/saved_imgs',
                                 transform=transforms.ToTensor())

# test_dataset = torchvision.datasets.DatsetFolder(root='./data',
#                                           train=False,
#                                           transform=transforms.ToTensor())

print('Number of train images: {}'.format(len(train_dataset)))
# print('Number of test images: {}'.format(len(test_dataset)))

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)


total_step = len(train_loader)

# Get single image data
image_tensor, image_label = train_dataset.__getitem__(random.randint(0, len(train_dataset)))
print('Size of single image tensor: {}'.format(image_tensor.size()))

# Torch tensor to numpy array
image_array = image_tensor.squeeze().numpy()
print('Size of single image array: {}\n'.format(image_array.shape))

# Plot image
plt.title('Image of {}'.format(image_label))
plt.imshow(image_array, cmap='gray')



## training step
model = network.to(device)
criterion = nn.MSELoss() # combines nn.LogSoftmax() and nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = images.view(-1, input_size, time_step).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))




