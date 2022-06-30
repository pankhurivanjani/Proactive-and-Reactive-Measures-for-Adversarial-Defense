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

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),DCT()])

#transform_test = transforms.Compose([transforms.ToTensor()])#make appropropriate changes here acc to method used
adv_dataset = TensorDataset(adversarial_tensor.permute(0,3,1,2))
testset = torchvision.datasets.CIFAR10(root='./data/cifar10', train=False,
                                         download=True, transform=transform_test)

test_loader = torch.utils.data.DataLoader(testset, batch_size=1, pin_memory=True,
                                          shuffle=False, num_workers=4)

adv_loader = torch.utils.data.DataLoader(adv_dataset, batch_size=1, pin_memory=True,
                                          shuffle=False, num_workers=4)

clean_acc = 0
adv_acc=0


features_extracted_clean = []
features_extracted_adv = []
outputs_labels = []

count = 0

output_labels=[]

for i, (img, label) in enumerate(test_loader):
    output_labels.append(label) 
    
for i, (img, label) in enumerate(test_loader):
    img, label = img.to(device), label.to(device)
    
    #output_labels.append(label)
    #extract features from last layer of model in variable features_extracted 
    feats1024 = model((img))[2]

    #extract labels from last layer of model in variable outputs_labels
    features_extracted_clean.append(feats1024.reshape(len(img),-1).detach().cpu().numpy())
    features_extracted_clean.append(feats1024.reshape(len(img),-1).detach().cpu().numpy())


    #Attack 
    if count%100==0:     
        print('Batch: {0}'.format(i))
    count = count +1

print('feats_shap',feats1024.shape)
print(features_extracted_clean[0].shape)

feature_extracted=np.array(features_extracted_clean).reshape(10000,-1)
output_labels=np.array(output_labels).reshape(10000,1)
embedding_data = TSNE().fit_transform(feature_extracted)


output_labels=np.array(output_labels).reshape(10000,1)
output_labels=list(output_labels.reshape(10000,))

cifar_labels = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def plot_embedding(data, label, title):
   

    fig, ax = plt.subplots()
   
    scatter=ax.scatter(data[:, 0], data[:, 1],c=label)
    #plt.savefig(img_title, dpi=300)
  
    legend1 = ax.legend(*scatter.legend_elements(),
                    loc="lower left", title="Classes")
    ax.add_artist(legend1)
    plt.show()
plot_embedding(embedding_data, output_labels, 'softmax_dct')
#print length of embedding_data
print(type(embedding_data))

