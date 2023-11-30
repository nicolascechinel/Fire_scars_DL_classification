import torchvision
import torchvision.transforms as transforms
import torch.optim as optim 
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch 
from cv2 import resize
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


from tensorboardX import SummaryWriter

import os 
import sys 
import time 
import argparse
import datetime
import copy 

from spectrumnet import SpectrumNet
from GeoTiffDataset import DatasetFolder

def get_pretrained_network(file, num_bands, num_output_classes, version=1.1):
    net = SpectrumNet(num_bands=num_bands, version=version)
    net.load_state_dict(torch.load(file, map_location='cuda:0'))

    # change the last conv2d layer
    net.classifier._modules["1"] = nn.Conv2d(512, num_output_classes, kernel_size=1)
    # change the internal num_classes variable rather than redefining the forward pass
    net.num_classes = num_output_classes

    tm = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    file_name = 'SpectrumNet-'+str(num_bands)+tm 

    return net, file_name

# Return network and filename 
def getNetwork(num_bands, version=1.1):
    net = SpectrumNet(num_bands=num_bands, version=version)
    tm = time.strftime("%Y-%m-%d_%H:%M:%S", time.gmtime())
    file_name = 'SpectrumNet-'+str(num_bands)+tm 
    return net, file_name


# Training 
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time() 
    input_num = 0

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0 

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-" * 10)

        # Each epoch has a training and validation phase 
        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                model.train() # Set model to training mode 
            else:
                model.eval()  # Set model to evalution mode 

            running_loss = 0.0 
            running_corrects = 0.0 

            # iterate over data 

            for inputs, labels, _ in data_loaders[phase]:
                #print(inputs.size())
                inputs = inputs.to(device)
                inputs = inputs.type(torch.cuda.FloatTensor)
                labels = labels.to(device)

                # zero the parameter gradients 
                optimizer.zero_grad()

                # forward pass, track history only in train mode 
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backprop only if training phase 
                    if phase == 'train':
                        loss.backward()
                        optimizer.step() 
                        writer.add_scalar('data/train_loss', loss.item(), input_num)
                        input_num += 1

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == "val":
                writer.add_scalar("data/acc", epoch_acc, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model 
            if phase == 'val' and epoch_acc > best_acc: 
                best_acc = epoch_acc 
                best_model_wts = copy.deepcopy(model.state_dict())
        

        print()
        
    
    time_elapsed = time.time() - since 
    print("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    print("Best validation Acc: {:.4f}".format(best_acc))
    # load best model weights 
    model.load_state_dict(best_model_wts)
    return model 

def test_model(model):
    running_corrects = 0

    model.eval()
    #create a classificated image

    for inputs, labels, _ in data_loaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
        
        running_corrects += torch.sum(preds == labels)
        test_acc = running_corrects.double() / dataset_sizes['test']
        writer.add_scalar("data/test_acc", test_acc, 0)
    
    writer.confusion_matrix.add(createConfusionMatrix(data_loaders['test']))
   
    print("Test set acc: {:.4f}".format(test_acc))

    with open("test_acc_master.txt", 'a') as f:
        f.write("{},".format(test_acc))


def createConfusionMatrix(loader):
    y_pred = [] # save predction
    y_true = [] # save ground truth

    # iterate over data
    for inputs, labels in loader:
        output = net(inputs)  # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.numpy()
        y_pred.extend(output)  # save prediction

        labels = labels.data.numpy()
        y_true.extend(labels)  # save ground truth

    # constant for classes
    classes = ('cicatriz', 'cultivo', 'others')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(12, 7))    
    return sn.heatmap(df_cm, annot=True).get_figure()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SpectrumNet Training')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--num_bands', '-n', nargs='*', default=10, type=int, help='number of input spectral bands')
    args = parser.parse_args()

    # select desired spectral bands
    #means = [0.0145, 0.0198, 0.0246, 0.0952, 0.1538, 0.1721, 0.1681, 0.1754, 0.1829, 0.1707]
    #stds = [0.0137, 0.017, 0.02, 0.06465, 0.096, 0.1075, 0.1062, 0.11044, 0.1146, 0.1084]
    means = [0.082069306480097,0.074374801854573,0.068535816134024,0.17827156146805,0.1217472755784,0.17687370267105,0.21212231313656,0.2460061763849,0.21545063001746,0.11406500629675]
    stds = [0.037165501271468,0.062455810052342,0.079689261727973,0.067736464701906,0.052863302745539,0.029269906858373,0.0473160933149,0.024672873967778,0.01250081264775,0.0079772340140334]

    cur_means = []
    cur_stds = []
    for i in range(10):
        # rasterio indexing starts at 1 and that is what the flag corresponds to 
        cur_means.append(means[i])
        cur_stds.append(stds[i])
        # cur_means.append(means[int(b)-1])
        # cur_stds.append(stds[int(b)-1])
    print(args.num_bands)
    print(cur_means)
    print(cur_stds)

    writer = SummaryWriter()

    data_transforms = {
        'train': transforms.Compose([
                                     transforms.ToTensor(),
                                     transforms.Normalize(cur_means, cur_stds),
                                    ]),
        'val': transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(cur_means, cur_stds),
                                  ]),
        'test': transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(cur_means, cur_stds),
                                  ])
    }

    #data_dir = '/Users/senecal/Repos/hyperspectral/data/Tomato2'
    data_dir = '/home/nico/dev/cicatrizes_certo/dataset_dividido'
    #data_dir = '/home/nico/dev/cicatrizes_certo/dataset_dividido2class'


    image_datasets = {x: DatasetFolder(os.path.join(data_dir, x), ['.tif'], num_bands=args.num_bands, transform=data_transforms[x]) for x in ['train', 'val', 'test']}
    data_loaders = {x: DataLoader(image_datasets[x], batch_size=2, shuffle=True, num_workers=8) for x in ['train', 'val', 'test']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train','val', 'test']}
    class_names = image_datasets['train'].classes
    print(class_names)
    print(dataset_sizes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get network and savepath 
    if(args.resume):
        print("Resuming from checkpoint...")
        fname = '/home/nico/dev/cicatrizes_certo/spectrumNet/intel_spectrumnet-master/intel_spectrumnet-master/data/SpectrumNet_DWS.pt'
        net, filename = get_pretrained_network(file=fname,
                                               num_bands=10,
                                               num_output_classes=4,
                                               version=1.1)
    else:
        net, filename = getNetwork(10)
    net.to(device)

    # set up loss criterion and optimizer 
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, nesterov=True)
    #optimizer = optim.RMSprop(net.parameters(), lr=0.001)

    # lr scheduler
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.75)

    # start training 
    model_ft = train_model(net, criterion, optimizer, exp_lr_scheduler, num_epochs=25)

    # run best model on test set
    test_model(model_ft)
    model_ft.eval()
    x = torch.randn(1,10, 64, 64, requires_grad=True).to(device)
    torch.onnx.export(model_ft, x,"torchToOnrrnx_mean4_class.onnx", verbose=True, input_names = ['input'], output_names = ['output'])
   
    print("Saving best model...")
    if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
    save_point = os.path.join("checkpoint", filename)
    torch.save(model_ft, save_point + '.pt')
    print("Model saved!")
   

