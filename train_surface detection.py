import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
import torchvision
from PIL import Image
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
data_dir = 'Data/'
training_dir = 'Data/train/'
eval_dir = 'Data/valid/'
test_dir = 'Data/train/'
from glob import glob
images = glob(os.path.join(data_dir, '*/*.png'))
tot_images = len(images)
im_cnt = []
class_names = []
print('{:18s}'.format('Class'), end='')
print('Count')
print('-' * 24)

#Cropping all the images to only focus on the road part and then replacing the original pics
dir_path = "Data/"
substring = "cropped"
def resize_im(path):
    if os.path.isfile(path) and substring not in path:
        im = Image.open(path)
        height, width = im.size
        f, e = os.path.splitext(path)
        im1 = im.crop((0, height/6, width, height/2))
        im1.save(f + 'cropped.png', quality=100)
        os.remove(path)

def resize_all(mydir):
    for subdir , _ , fileList in os.walk(mydir):
        for f in fileList:
                full_path = os.path.join(subdir,f)
                resize_im(full_path)

resize_all(dir_path)
    

for folder in os.listdir(os.path.join(data_dir)):
    folder_num = len(os.listdir(os.path.join(data_dir, folder)))
    im_cnt.append(folder_num)
    class_names.append(folder)
    print('{:20s}'.format(folder), end=' ')
    print(folder_num)
    if (folder_num < tot_images):
        tot_images = folder_num
        folder_num = folder
num_classes = len(class_names)
print('Total number of classes: {}'.format(num_classes))

# Define transforms for the training and validation sets
data_transforms ={
    "train_transforms": transforms.Compose([transforms.RandomResizedCrop(400),  
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], 
                                                                [0.229, 0.224, 0.225])]),
   "valid_transforms": transforms.Compose([transforms.Resize(400),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])]), 
    "test_transforms": transforms.Compose([transforms.Resize(400),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])
}


# Load the datasets with ImageFolder
train_data = datasets.ImageFolder(training_dir, transform=data_transforms["train_transforms"])#loading dataset
valid_data = datasets.ImageFolder(eval_dir, transform=data_transforms["valid_transforms"])
test_data = datasets.ImageFolder(test_dir, transform=data_transforms["test_transforms"])


# Obtain training indices that will be used for validation and test

num_train = len(train_data)
num_eval = len(valid_data)
num_test = len (test_data)

indices_train = list(range(num_train))
indices_eval = list(range(num_eval))
indices_test = list(range(num_test))

train_count = int(num_train)
valid_count = int(num_eval)
test_count = int(num_test)

train_idx = indices_train[:train_count]
valid_idx = indices_eval[:valid_count]
test_idx = indices_test[:test_count]


print(len(train_idx), len(valid_idx), len(test_idx))
print("Training", train_count, np.sum(len(train_idx)/num_train))
print("Validation", valid_count, np.sum(len(valid_idx)/num_train))
print("Test", test_count, np.sum(len(test_idx)/num_train))

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)
test_sampler = SubsetRandomSampler(test_idx)
trainloader = torch.utils.data.DataLoader(train_data, batch_size = 64, shuffle = True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size = 64, sampler = valid_sampler)
testloader = torch.utils.data.DataLoader(test_data, batch_size = 64, sampler = test_sampler)
classes=['Asphalt_dry', 'Asphalt_wet', 'Gravel_dry', 'Gravel_wet', 'Cobble_dry', 'Cobble_wet']
def imshow(img):
    img = img / 2 + 0.5 #unnormalize
    plt.imshow(np.transpose(img, (1,2,0))) #convert tensor image type to numpy image type for visualization

# Specify model architecture
# Load the pretrained model from pytorch's library and stored it in model_transfer
model_transfer = models.googlenet(pretrained=True)

# Check if GPU is available
use_cuda = torch.cuda.is_available()
if use_cuda:
    model_transfer = model_transfer.cuda()


#Lets read the fully connected layer
print(model_transfer.fc.in_features)
print(model_transfer.fc.out_features)
for param in model_transfer.parameters():
    param.requires_grad=True
# Define n_inputs takes the same number of inputs from pre-trained model
n_inputs = model_transfer.fc.in_features #refer to the fully connected layer only

# Add last linear layer (n_inputs -> 6 classes). In this case the ouput is 6 classes
# New layer automatically has requires_grad = True
last_layer = nn.Linear(n_inputs, len(classes))

model_transfer.fc = last_layer

# If GPU is available, move the model to GPU
if use_cuda:
    model_transfer = model_transfer.cuda()
  
# Check to see the last layer produces the expected number of outputs
print(model_transfer.fc.out_features)

# Specify loss function and optimizer
criterion_transfer = nn.CrossEntropyLoss()
optimizer_transfer = optim.SGD(model_transfer.parameters(), lr=0.001, momentum=0.9)

def train(loader, model, optimizer, criterion, use_cuda):
    model.train()
    train_loss = 0.0
    size=len(loader.dataset)
    for batch_idx, (data,target) in enumerate(loader):
        # 1st step: Move to GPU
        if use_cuda:
            data,target = data.cuda(), target.cuda()
  
        # Then, clear (zero out) the gradient of all optimized variables
        optimizer.zero_grad()
        # Forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # Perform the Cross Entropy Loss. Calculate the batch loss.
        loss = criterion(output, target)
        # Backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # Perform optimization step (parameter update)
        optimizer.step()
        # Record the average training loss
        #train_loss = train_loss + ((1/ (batch_idx + 1 ))*(loss.data-train_loss))
        train_loss += loss.item() * len(data)
        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
    train_loss /= size
    return model, train_loss
 
def validate(loader, model, criterion, use_cuda):
    size=len(loader.dataset)
    valid_loss = 0.0
    model.eval()
    with torch.no_grad():
        for batch_idx, (data,target) in enumerate(loader):
            # Move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
                
            # Update the average validation loss
            # Forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # Calculate the batch loss
            loss = criterion(output, target)
            # Update the average validation loss
            valid_loss += loss.item() * len(data)
        
    valid_loss = valid_loss/size
    return valid_loss
    
# Train the model
def training(n_epochs, loaders, model, optimizer, criterion, use_cuda, save_path):
    '''returns trained model'''
    # Initialize tracker for minimum validation loss
    valid_loss_min = np.inf
  
    for epoch in range(1, n_epochs+1):
        # In the training loop, I track down the loss
        # Initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
    
        print(f'Started Epoch: {epoch}')
        # Model training
        model, train_loss = train(loaders['train'], model, optimizer, criterion, use_cuda)
      
        # Model validation
        valid_loss = validate(loaders['valid'], model, criterion, use_cuda)
      
        # print training/validation stats
        print('Finished Epoch: {} \tAvg. Training Loss: {:.5f} \tAvg. Validation Loss: {:.5f}'.format(
            epoch,
            train_loss,
            valid_loss))
    
        # Save the model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.5f} --> {:.5f}). Saving model ...'.format(
                  valid_loss_min,
                  valid_loss))
            torch.save(model.state_dict(), 'model_transfer.pt')
            valid_loss_min = valid_loss
  
    # Return trained model
    return model

# Define loaders transfer
loaders_transfer = {'train': trainloader,
                    'valid': validloader,
                    'test': testloader}

# Train the model
model_transfer = training(50, loaders_transfer, model_transfer, optimizer_transfer, criterion_transfer, use_cuda, 'model_transfer.pt')
model_transfer.load_state_dict(torch.load('model_transfer.pt'))
def test(loaders, model, criterion, use_cuda):

    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.

    model.eval() #set model into evaluation/testing mode. It turns of drop off layer
    #Iterating over test data
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loaders['test']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # update average test loss 
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss.data - test_loss))
            # convert output probabilities to predicted class
            pred = output.data.max(1, keepdim=True)[1]
            # compare predictions to 
            correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
            total += data.size(0)
            
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: {:.3f} ({}/{})'.format(
        100. * correct / total, correct, total))

# call test function   
test(loaders_transfer, model_transfer, criterion_transfer, use_cuda)

# This code causes erroring so I commented i, probably you need to free the memory before doing this:
#Obtain one batch of test images
#dataiter = iter(testloader)
#images, labels = dataiter.next()
#images.numpy

#Move model inputs to cuda, if GPU available
#if use_cuda:
#    images = images.cuda()
    
#Get sample outputs
#output= model_transfer(images)

#Convert output probabilities to predicted class
#_,preds_tensor = torch.max(output,1)
#preds = np.squeeze(preds_tensor.numpy()) if not use_cuda else np.squeeze(preds_tensor.cpu().numpy())
'''
#Plot the images in the batch, along with predicted and true labels
fig = plt.figure(figsize=(30,4))
for idx in np.arange(20):
    ax = fig.add_subplot(30, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images.cpu()[idx], (1,2,0)))
    ax.set_title("{} ({})".format(classes[preds[idx]],classes[labels[idx]]),
                color=("green" if preds[idx]==labels[idx].item() else "red"))
'''
