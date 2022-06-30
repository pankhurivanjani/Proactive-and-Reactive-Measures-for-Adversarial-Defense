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
# import torch_dct as dct
from torch.utils.data import DataLoader, TensorDataset
import imgaug.augmenters as iaa

"""
Adversarial Attack Options: fgsm, bim, mim, pgd
"""

num_classes=10

# class DCT(object):
#     def __call__(self, X):
#         X_dct = dct.dct(X)
#         return X_dct
# dct_op=DCT()

# class Augspace(object):
#     def __call__(self, X):
#       seq = iaa.Sequential([
#                             iaa.flip.Fliplr(p=0.5),
#                             iaa.flip.Flipud(p=0.5),
#                             iaa.GaussianBlur(sigma=(0.0, 3.0))])
#       X = seq(images=X)
#       return X

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
softmax_filename= 'Models_Softmax_resnet20/CIFAR10_Softmax_Resnet20_IMGAUG_Affine.pth.tar'    
filename= 'Models_PCL_resnet20/CIFAR10_PCL_Resnet20_IMGAUG_Affine.pth.tar' 
robust_model= 'robust_model.pth.tar'

checkpoint = torch.load(softmax_filename)#select model you want to test
model.load_state_dict(checkpoint['state_dict'])
model.eval()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

adversarial_data = np.load("blackbox_attacks/attack.npy") #generated via numpy
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
seq = iaa.pillike.Affine(rotate=(-20, 20), fillcolor=(0, 256))
for i, (adv_img,real_data) in enumerate(zip(adv_loader,test_loader)):
    adv_img,img, label = adv_img[0].to(device),real_data[0].to(device), real_data[1].to(device)
    
#     ## Specific to GaussianBlurr case as the normalization is done before the transformation in the data loader and it doesn't permutes the data:
#     # clean acc
#     temp_clean = img.detach().cpu().clone().numpy()
#     temp_clean = seq(images=temp_clean)
#     temp_clean = model(torch.from_numpy(np.array(temp_clean)).cuda()) # for GB
#     temp_clean = torch.sum(temp_clean[3].argmax(dim=-1) == label).item()
#     clean_acc+=temp_clean
    
#     #adv acc
#     temp_adv = adv_img.detach().cpu().clone().numpy()
#     temp_adv = seq(images=temp_adv)
#     temp_adv = model(torch.from_numpy(np.array(temp_adv)).cuda()) # GB
#     temp_adv = torch.sum(temp_adv[3].argmax(dim=-1) == label).item()
#     adv_acc+=temp_adv
        
    # clean acc

    temp_clean = img.detach().cpu().clone().permute(0,2,3,1).numpy()
    # temp_clean = normalize(temp_clean)
    temp_clean = np.array(temp_clean*255).astype('uint8')
    temp_clean = seq(images=temp_clean)
    temp_clean = model(normalize(torch.from_numpy(np.array(temp_clean/255).astype('float32'))).permute(0,3,1,2).cuda())
    # temp_clean = model(normalize(torch.from_numpy(np.array(temp_clean)).cuda()))
    temp_clean = torch.sum(temp_clean[3].argmax(dim=-1) == label).item()
    clean_acc+=temp_clean
    # clean_acc += torch.sum(model(seq(images=normalize(img.clone().detach())))[3].argmax(dim=-1) == label).item()
    
    #adv acc
    temp_adv = adv_img.detach().cpu().clone().permute(0,2,3,1).numpy()
    # temp_adv = normalize(temp_adv)
    temp_adv = np.array(temp_adv*255).astype('uint8') 
    temp_adv = seq(images=temp_adv)
    temp_adv = model(normalize(torch.from_numpy(np.array(temp_adv/255).astype('float32'))).permute(0,3,1,2).cuda())
    temp_adv = torch.sum(temp_adv[3].argmax(dim=-1) == label).item()
    adv_acc+=temp_adv
    # adv_acc += torch.sum(model(seq(images=normalize(adv_img.clone().detach())))[3].argmax(dim=-1) == label).item()
    
    print('Batch: {0}'.format(i))
print('Clean accuracy:{0:.3%}\t BBox Adv Accuracy:{1:.3%}\t'.format(clean_acc / len(testset), adv_acc / len(testset)))
