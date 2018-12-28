import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import torch
from torch import nn
from torch import tensor
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from  collections import OrderedDict

# PROGRAMMER: Ritwik Sinha  
# DATE CREATED: 28-12-2018(dd-mm-yyyy)                                 
# REVISED DATE: 28-12-2018(dd-mm-yyyy)
# PURPOSE: This function is used to create initialise the model,criterion,optimiser,device processor 
#            and also update the model classifier,
# PARAMERTERS:structure=neural network architechture
#             hidden_units=no of node for first hidden layer
#             dropout_value=Dropout value for the NN 
#             lr=learning rate for the NN
#             power=device processor type
def nn_setup(structure,hidden_units,dropout_value,lr,power):
#     #create a vgg model
    model=None
    if(structure=='vgg16'):
        model=models.vgg16(pretrained=True)
         # Freeze parameters to avoid  backprop 
        for param in model.parameters():
            param.requires_grad = False
        #Define a new classifier        
        classifier = nn.Sequential(OrderedDict([
                          ('dropout',nn.Dropout(dropout_value)),
                          ('inputLayer', nn.Linear(25088, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('HiddenLayer', nn.Linear(hidden_units, 200)),
                          ('relu2', nn.ReLU()),
                          ('OutputLayer', nn.Linear(200, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        #Update the existing model classifier
        model.classifier=classifier
    elif(structure=='alexnet'):
        model = models.alexnet(pretrained=True)
         # Freeze parameters to avoid  backprop 
        for param in model.parameters():
            param.requires_grad = False
        #Define a new classifier        
        classifier = nn.Sequential(OrderedDict([
                          ('dropout',nn.Dropout(dropout_value)),
                          ('inputLayer', nn.Linear(1024, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('HiddenLayer', nn.Linear(hidden_units, 200)),
                          ('relu2', nn.ReLU()),
                          ('OutputLayer', nn.Linear(200, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        #Update the existing model classifier
        model.classifier=classifier
    elif(structure=='densenet'):
        model = models.densenet161()
         # Freeze parameters to avoid  backprop 
        for param in model.parameters():
            param.requires_grad = False
        #Define a new classifier        
        classifier = nn.Sequential(OrderedDict([
                          ('dropout',nn.Dropout(dropout_value)),
                          ('inputLayer', nn.Linear(9216, hidden_units)),
                          ('relu1', nn.ReLU()),
                          ('HiddenLayer', nn.Linear(hidden_units, 200)),
                          ('relu2', nn.ReLU()),
                          ('OutputLayer', nn.Linear(200, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        #Update the existing model classifier
        model.classifier=classifier
    else:
        print("Invalid Model.Please restart system ")
        model=None,
        criterion=None
        optimizer=None
        device=None
        return model,criterion,optimizer,device
    #Define the criterion parameter
    criterion = nn.NLLLoss()
    #Define the optimizer parmeter
    optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    #Check the device whether cpu or gpu is being utilised 
    if(power=='gpu'):
        device = 'cuda'
    else:
        device = 'cpu'
    # Return the model,criterion and optmizer
    return model,criterion,optimizer,device


# PROGRAMMER: Ritwik Sinha  
# DATE CREATED: 28-12-2018(dd-mm-yyyy)                                 
# REVISED DATE: 28-12-2018(dd-mm-yyyy)
# PURPOSE: This function is used to create initialise the model,criterion,optimiser,device processor 
#            and also update the model classifier,
# PARAMERTERS:data_dir=directory
def load_data(data_dir):
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    training_transforms =           transforms.Compose([transforms.RandomRotation(30),transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    validation_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    #1. Training Dataset
    training_dataset=datasets.ImageFolder(train_dir, transform=training_transforms)
    #2 Validation dataset
    validation_dataset=datasets.ImageFolder(valid_dir, transform=validation_transforms)
    #3 Test dataset
    test_dataset=datasets.ImageFolder(test_dir, transform=test_transforms)
    #1. Training dataloader
    trainingloader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
    #2. Validation dataloader
    validationloader=torch.utils.data.DataLoader(validation_dataset, batch_size=32, shuffle=True)   
    #3. Test dataloader
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=20,shuffle=True)
    dataLoader={'loaders':{
                'trainingloader':trainingloader,
                'validationloader':validationloader,
                'testloader':testloader
                },
                'dataset':
                {
                    'training_dataset':training_dataset,
                    'validation_dataset':validation_dataset,
                    'test_dataset':test_dataset
                }
               }
    
    return dataLoader


# PROGRAMMER: Ritwik Sinha  
# DATE CREATED: 28-12-2018(dd-mm-yyyy)                                 
# REVISED DATE: 28-12-2018(dd-mm-yyyy)
# PURPOSE: This function is used train the NN
# PARAMERTERS:data_dir=model=Model Architecture
#                      criterion=criterion 
#                      optimizer=optimizer
#                      device=device processor type
#                      trainingloader=training loader
def train_network(model,criterion,optimizer,device,trainingloader,validationloader,epoch):
    print_every=5
    steps=0
    loss_show=[]
    model.to(device)
    for e in range(epoch):
        running_loss=0
        for ii,(inputs,labels) in enumerate(trainingloader):
            steps+=1
            inputs,labels=inputs.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs=model.forward(inputs)
            loss=criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss+=loss.item()
            if steps%print_every==0:
               model.eval()
               vlost=0
               accuracy=0
               for ii,(vinputs,vlabels) in enumerate(validationloader):
                    optimizer.zero_grad()
                    vinputs,vlabels=vinputs.to(device+':0'),vlabels.to(device+':0')
                    model.to(device+':0')
                    with torch.no_grad():
                        output=model.forward(vinputs)
                        vloss=criterion(output,vlabels)
                        ps=torch.exp(output).data
                        equality=(vlabels.data == ps.max(1)[1])
                        accuracy+=equality.type_as(torch.FloatTensor()).mean()
               vloss=vloss/len(validationloader)
               accuracy=accuracy/len(validationloader)
               print("Epoch: {}/{}... ".format(e+1, epoch),"Loss: {:.4f}".format(running_loss/print_every),"Validation Lost: {:.4f}".format(running_loss/print_every),"Accuracy  Loss: {:.4f}".format(running_loss/print_every))
            running_loss=0
    print('Model Training Complete Final Result')
    print(' Epoch: {} '.format(epoch))
    print(' Steps: {}  '.format(steps))

# PROGRAMMER: Ritwik Sinha      
# DATE CREATED: 28-12-2018(dd-mm-yyyy)                                 
# REVISED DATE: 28-12-2018(dd-mm-yyyy)
# PURPOSE: This function is save model configuration 
# PARAMERTERS:path=filepath
#             structure=Neural Network Architecture
#             hidden_layer1=hidden layer
#             dropout=dropout
#             lr=learning rate
#             epochs=epoch value
def save_checkpoint(path,model,structure,hidden_layer1,dropout,lr,epochs,training_dataset,power):
    model.class_to_idx = training_dataset.class_to_idx
    model.cpu
    torch.save({'structure' :structure,
                'hidden_layer1':hidden_layer1,
                'dropout':dropout,
                'lr':lr,
                'epochs':epochs,
               'power':power,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx
               },
                path)
    print('Model Configuration Saved....')

# PROGRAMMER: Ritwik Sinha      
# DATE CREATED: 28-12-2018(dd-mm-yyyy)                                 
# REVISED DATE: 28-12-2018(dd-mm-yyyy)
# PURPOSE: This function is load model configuration 
# PARAMERTERS:path=filepath
    
def load_checkpoint(path):
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']
    power=checkpoint['power']
    model,_,_,_ = nn_setup(structure,hidden_layer1,dropout,lr,power)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    print('Model Configurion Loaded....')
    return model


# PROGRAMMER: Ritwik Sinha      
# DATE CREATED: 28-12-2018(dd-mm-yyyy)                                 
# REVISED DATE: 28-12-2018(dd-mm-yyyy)
# PURPOSE: This function is used to process the image and convert it into a tensor 
# PARAMERTERS:imagepath=image path

def process_image(image_path):
    img = Image.open(image_path)
    make_img_good = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tensor_image = make_img_good(img)
    return tensor_image


# PROGRAMMER: Ritwik Sinha  
# DATE CREATED: 28-12-2018(dd-mm-yyyy)                                 
# REVISED DATE: 28-12-2018(dd-mm-yyyy)
# PURPOSE: This function is save model configuration 
# PARAMERTERS:image_path=path of the image for prediction
#             model= NN Model being used
#             topk= top K most likely classes
#             power=device processor type (gpu or cpu)
def predict(image_path, model, topk, power):

    if torch.cuda.is_available() and power=='gpu':
        model.to('cuda:0')

    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    if power == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)

    probability = F.softmax(output.data,dim=1)

    return probability.topk(topk)