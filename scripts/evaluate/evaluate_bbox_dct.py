import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from resnet_model import *  # Imports the ResNet Model
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import cm

from PIL import Image
import numpy as np
import torch_dct as dct
from torch.utils.data import DataLoader, TensorDataset

"""
Adversarial Attack Options: fgsm, bim, mim, pgd
"""

num_classes=10
class DCT(object):


    def __call__(self, X):
        X_dct = dct.dct(X)
        return X_dct
dct_op=DCT()
model = resnet(num_classes=num_classes,depth=20)
if True:
    model = nn.DataParallel(model).cuda()
def normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0])/std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1])/std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2])/std[2]

    return t
def un_normalize(t):
    t[:, 0, :, :] = (t[:, 0, :, :] * std[0]) + mean[0]
    t[:, 1, :, :] = (t[:, 1, :, :] * std[1]) + mean[1]
    t[:, 2, :, :] = (t[:, 2, :, :] * std[2]) + mean[2]

    return t
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
#Loading Trained Model
softmax_filename= 'Models_Softmax/CIFAR10_Softmax20DCT.pth.tar'    
filename= 'Models_PCL/CIFAR10_PCLDCT.pth.tar' 
robust_model= 'robust_model.pth.tar'

checkpoint = torch.load(softmax_filename)#select model you want to test
model.load_state_dict(checkpoint['state_dict'])
model.eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

adversarial_data = np.load("attack.npy") #generated via numpy
epsilon=8/255
adversarial_tensor = torch.Tensor(adversarial_data)/255 # transform to torch tensor



transform_test = transforms.Compose([transforms.ToTensor()])#make appropropriate changes here acc to method used
adv_dataset = TensorDataset(adversarial_tensor.permute(0,3,1,2))
testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                         download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=128, pin_memory=True,
                                          shuffle=False, num_workers=4)
adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=128, pin_memory=True,
                                          shuffle=False, num_workers=4)

clean_acc = 0
adv_acc=0
for i, (adv_img,real_data) in enumerate(zip(adv_loader,test_loader)):
    adv_img,img, label = adv_img[0].to(device),real_data[0].to(device), real_data[1].to(device)
    
    clean_acc += torch.sum(model(dct_op(normalize(img.clone().detach())))[3].argmax(dim=-1) == label).item()
    adv_acc += torch.sum(model(dct_op(normalize(adv_img.clone().detach())))[3].argmax(dim=-1) == label).item()

    print('Batch: {0}'.format(i))
print('Clean accuracy:{0:.3%}\t BBox Adv Accuracy:{1:.3%}\t'.format(clean_acc / len(testset), adv_acc / len(testset)))
