#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
from IPython.display import Audio, display

import torch
import torch.nn as nn
from torch.nn import init
import torchaudio
import math, random
from torchaudio import transforms
from torch.utils.data import DataLoader, Dataset, random_split
import torch.nn.functional as F

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from torchsummary import summary
from matplotlib import cm
import matplotlib as mpl


# In[2]:


dataset='C:/Users/Engem/Downloads/data/GroundType1/'

ID=[]
label=[]
ClassID=[]
 
# iterate over files in that directory

for folder in os.listdir(dataset):                               #go into the directory
    for filename in os.listdir(dataset+ str(folder)):          #go in every class 
        f = os.path.join(dataset+ str(folder), filename) #scan through every file in that class
        if os.path.isfile(f):
            ID.append(f.split('\\')[-1])
            label.append(str(folder))
            
for i in range(len(label)):
    if(label[i]=='Concrete'):
        ClassID.append(0) 
    elif(label[i]=='Flatmountain'):
        ClassID.append(1)
    else:
        ClassID.append(2)


# In[3]:


dset=pd.DataFrame()
dset['ID']=ID
dset['label']=label
dset['ClassID']=ClassID


# In[4]:


dset.head(599)


# In[5]:


dset.to_csv('C:/Users/Engem/Downloads/data/data.csv')


# In[6]:


df = pd.read_csv('C:/Users/Engem/Downloads/data/data.csv')


# In[7]:


df['ID'].loc[150]


# In[8]:


Grass = 'C:/Users/Engem/Downloads/data/GroundType1/Grass/'
path=os.path.join(Grass, df['ID'].loc[595])
print(path)


# In[9]:


def plot_audio(filename):
    waveform, sample_rate = torchaudio.load(filename)

    print("Shape of waveform: {}".format(waveform.size()))
    print("Sample rate of waveform: {}".format(sample_rate))

    fig1, ax1 = plt.subplots(figsize=(12, 5))    
  
    ax1.plot(waveform[0,:].numpy())

    ax1.set_xlabel('time') 
    ax1.set_ylabel("amplitude")

    return waveform, sample_rate


aud= plot_audio(path)
print(aud)


# In[10]:


def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig, sr = aud
    top_db = 80
    
    # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
    spec1 = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    # Convert to decibels
    spec = transforms.AmplitudeToDB(top_db=top_db)(spec1)
    
    
    print("Shape of spec: {}".format(spec1.size()))
    print("Shape of spec: {}".format(spec.size()))
     
    fig, ax = plt.subplots(nrows=2, ncols=1, sharey=True,  figsize=(10, 5))
   
    
    s= ax[0].imshow(spec1[0,:,:].numpy())
    s1= ax[1].imshow(spec[0,:,:].numpy())
    ax[0].grid(False)
    ax[1].grid(False)
    fig.colorbar(s, ax=ax[0], format='%+2.0f dB') 
    fig.colorbar(s1, ax=ax[1], format='%+2.0f dB')    
    ax[0].set_title('Mel power spectrogram') 
    ax[0].set_xlabel('time') 
    ax[0].set_ylabel("Freq")
    
    ax[1].set_title('log Mel-frequency spectrogram')
    ax[1].set_xlabel('time') 
    ax[1].set_ylabel("Freq")


    return (spec)


spectro_gram(aud)


# In[11]:


plt.figure(figsize=(8,6),dpi=80)
sn.set_theme(style="darkgrid")
sn.countplot(x ='label',data=dset)
plt.title('counts: \n' +'Concrete:'+str(dset.label.value_counts()[0])+'\n Flatmountain:'+str(dset.label.value_counts()[1])+'\n Grass:'+str(dset.label.value_counts()[2]))
plt.show()


# In[12]:


print(dset.label.unique())


# In[13]:


print("Number of training examples=", dset.shape[0], "  Number of classes=", len(dset.label.unique()))


# In[14]:


# Read file

df['ID'] = df['label'].astype(str) + '/' + df['ID'].astype(str)
# Take relevant columns
df = df[['ID', 'ClassID']]
df.head(350)


# In[15]:


class AudioUtil():
    def open(audio_file): # Load an audio file. Return the signal as a tensor and the sample rate
        sig, sr = torchaudio.load(audio_file)
        return (sig, sr)
    
    
    def resample(aud, newsr):
        sig, sr = aud
  
        if (sr == newsr):# Nothing to do
            #print('newsr',newsr)
            return aud

        num_channels = sig.shape[0]# Resample first channel
        #print('num_channels',num_channels)
        resig = torchaudio.transforms.Resample(sr, newsr)(sig[:1,:])
        #print("Shape of transformed waveform: {}".format(resig.size()))

        #plt.figure()
        #plt.plot(resig[0,:].numpy())
        #print('resig',resig)
        #print('resig',resig.shape)

        return (resig, newsr)
    
    

    def pad_trunc(aud, max_ms):
        sig, sr = aud
        #print('sig',sig)
        #print('sr',sr)
        #print(sig.shape)
        num_rows, sig_len = sig.shape
        #print('num_rows',num_rows)
        #print('sig_len',sig_len)
        max_len = sr//1000 * max_ms
        #print('max_len',max_len)

        if (sig_len > max_len): # Truncate the signal to the given length
           sig = sig[:,:max_len]
           #print('sig_pad_trunc',sig.shape)
           #plt.figure()
           #plt.plot(sig[0,:].numpy())
         

        elif (sig_len < max_len): # Length of padding to add at the beginning and end of the signal
              pad_begin_len = random.randint(0, max_len - sig_len)
              #print('pad_begin_len',pad_begin_len)
              pad_end_len = max_len - sig_len - pad_begin_len
              #print('pad_end_len',pad_end_len)
              # Pad with 0s
              pad_begin = torch.zeros((num_rows, pad_begin_len))
              #print('pad_begin',pad_begin)
              pad_end = torch.zeros((num_rows, pad_end_len))
              sig = torch.cat((pad_begin, sig, pad_end),1) 
              #print('sig_pad_trunc', sig.shape)
              #plt.figure()
              #sig1=sig[0,:].numpy()
              #plt.plot(sig1)      
                                     
        return (sig, sr)          
                                      

    def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
        sig, sr = aud
        top_db = 80
    
        # spec has shape [channel, n_mels, time], where channel is mono, stereo etc
        spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

        # Convert to decibels
        spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
        
        #print("Shape of spec: {}".format(spec.size()))
        return (spec)
    
        #def play_audio(waveform, sample_rate):
        #display(Audio(waveform[0], rate=sample_rate))
        
        


# In[16]:


#dataset=[]
# ----------------------------
# Sound Dataset
# ----------------------------
class SoundDS(Dataset):
   def __init__(self, df, data_path):
       self.df = df
       self.data_path = str(data_path)
       self.duration = 4000
       self.sr = 48000
       dataset=[]  
       # ----------------------------
       # Number of items in dataset
       # ----------------------------
   def __len__(self):
       return len(self.df)    
   
      # ----------------------------
      # Get i'th item in dataset
      # ----------------------------
   def __getitem__(self, idx):
      # Absolute file path of the audio file - concatenate the audio directory with the relative path
      audio_file = self.data_path + self.df.loc[idx, 'ID']
      #print(audio_file)

      class_id = self.df.loc[idx, 'ClassID']
       
      ID = self.df.loc[idx, 'ID']

      aud = AudioUtil.open(audio_file)

      reaud = AudioUtil.resample(aud, self.sr)

      dur_aud= AudioUtil.pad_trunc(reaud, self.duration)
      #print(dur_aud)
   
      sgram_features = AudioUtil.spectro_gram(dur_aud)
       
      #listen = AudioUtil.play_audio(dur_aud,self.sr)


      return  sgram_features, class_id


# In[17]:


myds = SoundDS(df, dataset)
print(myds[20])


# In[18]:


# data loader

random_seed = 64
torch.manual_seed(random_seed)
batch_size = 32

# Random split of 80:20 between training and validation
num_items = len(myds)
num_train = round(num_items * 0.8)
num_val = num_items - num_train
train_ds, val_ds = random_split(myds, [num_train, num_val])
len(train_ds), len(val_ds)

# Create training and validation data loaders
train_loader = torch.utils.data.DataLoader(train_ds, batch_size= batch_size, shuffle=True)
print(len(train_loader))
test_loader = torch.utils.data.DataLoader(val_ds, batch_size= batch_size, shuffle=False)
print(len(test_loader))
#for i, batch in enumerate(train_loader):
   #print(i, batch)


# In[19]:


examples = next(iter(train_loader))
data, targets= examples
print(data.shape, targets.shape)
#print(' '.join(f'{classes[targets[j]]:5s}' for j in range(batch_size)))



#examples = iter(test_loader)
#example_data,example_targets= examples.next()
#print(example_data.shape, example_targets.shape)
#print(' '.join(f'{classes[example_targets[j]]:5s}' for j in range(batch_size)))


# In[20]:


# Audio Classification Model, Build the model architecture
# ----------------------------

class AudioClassifier(nn.Module):

    def __init__(self):
        super().__init__()
        # first Convolution Block
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(4, 4), stride=(2, 2), padding=(2, 2))
        self.bn1 = nn.BatchNorm2d(16)
        self.drop1=nn.Dropout(0.25)
        init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.conv1.bias.data.zero_()

        # Second Convolution Block
        self.conv2 = nn.Conv2d(16, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2=nn.Dropout(0.4)
        init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.conv2.bias.data.zero_()
        
        #linear ouput
        self.ap = nn.AdaptiveMaxPool2d(output_size=1)# Input: (N, C, Hin, Win).Output: (N, C, S0, S1), where S=output_size.
        self.fc1 = nn.Linear(in_features=64, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=3)
 
    # ----------------------------
    # Forward pass computations
    # ----------------------------
    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))# Run the convolutional blocks
        x= self.drop1(x)
        x = self.bn2(F.relu(self.conv2(x)))
        x= self.drop2(x)
        x = self.ap(x)# Adaptive pool
        x = torch.flatten(x, 1) # flatten for input to linear layer
        x = F.relu(self.fc1(x))     # Linear layer
        x = self.fc2(x)
        return x            # Final output



# In[21]:


#TRAINING THE NETWORK

num_epochs= 50


def train(model, train_loader, optimizer,scheduler):
        model.train()
        loss_total = 0
        correct_prediction = 0
        total_prediction = 0
        
        for i, data in enumerate(train_loader):
            #LOADING THE DATA IN A BATCH
            data, target = data[0], data[1]
            #target= target.float()pred = np.round(output)
            
            # clear the gradients
            optimizer.zero_grad()
            
            # Normalize the inputs
            inputs_m, inputs_s = data.mean(), data.std()
            data = (data - inputs_m) / inputs_s

        
            #FORWARD PASS
            output = model(data)
            loss = criterion(output, target) 
            
            #BACKWARD AND OPTIMIZE
            loss.backward()
            optimizer.step()
            scheduler.step()
        
            # calculating the total_loss for checking
            loss_total += loss.item()
        
        
            # Get the predicted class with the highest score
            _, prediction = torch.max(output,1)
            # Count of predictions that matched the target label
            correct_prediction += (prediction == target).sum().item()
            total_prediction += prediction.shape[0]
        
        
        
        # Print stats at the end of the epoch
        train_loss = loss_total /len(train_loader)
        acc = 100* correct_prediction/total_prediction
        print(f'Epoch: {epoch}, Loss: {train_loss:.2f}, Accuracy on training set is: {acc:.2f}')
        train_accu.append(acc)
        train_losses.append(train_loss)
        
        print('Finished Training') 
        
  


# In[22]:


#TESTING THE MODE

def test(model, test_loader):
        y_true = []
        y_pred = []
        loss_total = 0
        correct_prediction = 0
        total_prediction = 0
        model.eval()
        
        with torch.no_grad():
             for i, data in enumerate(test_loader):
            
                #LOADING THE DATA IN A BATCH
                data, target = data[0], data[1]
                #target= target.float()
                
                # Normalize the inputs
                inputs_m, inputs_s = data.mean(), data.std()
                data = (data - inputs_m) / inputs_s
                       
                #FORWARD PASS
                output = model(data)
                loss = criterion(output, target)
            
                # calculating the total_loss for checking
                loss_total += loss.item() 
        
                # Get the predicted class with the highest score
                _, prediction = torch.max(output,1)
                y_pred.extend(prediction.tolist())
                y_true.extend(target.tolist()) 
                conf_matrix = confusion_matrix(y_true, y_pred)
                conf_matrix_df = pd.DataFrame(conf_matrix , index = ['Concrete','Flatmountain','Grass'], columns = ['Concrete','Flatmountain','Grass'])
                # Count of predictions that matched the target label
                correct_prediction += (prediction == target).sum().item()
                total_prediction += prediction.shape[0]
                
     
        # Print stats at the end of the epoch
        test_loss = loss_total /len(test_loader)
        acc = 100* correct_prediction/total_prediction
        print(f'Epoch: {epoch}, Loss: {test_loss:.2f}, Accuracy on testing set is: {acc:.2f}')
        test_accu.append(acc)
        test_losses.append(test_loss)
        
        
        plt.figure(figsize=(5,4))
        sn.heatmap(conf_matrix_df, annot=True , cmap="YlGnBu")
        plt.title('Confusion Matrix')
        plt.ylabel('Actal Values')
        plt.xlabel('Predicted Values')
        plt.show()
        



# In[23]:


# Define the loss function and optimizer

model = AudioClassifier()
criterion = nn.CrossEntropyLoss()#this case is equivalent to the combination of LogSoftmax and NLLLoss.
optimizer = torch.optim.Adam(model.parameters(),lr=0.001, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001,
                                                steps_per_epoch=int(len(train_loader)),
                                                epochs=num_epochs,
                                                anneal_strategy='linear')


# In[24]:


summary(model, (1, 64, 376))


# In[25]:


test_accu = []
train_accu= []
train_losses =[]
test_losses =[]
for epoch in range(num_epochs):
    train(model,train_loader,optimizer,scheduler)
    test(model,test_loader)


# In[26]:


plt.plot(train_accu)
plt.plot(test_accu,)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(['Train','test'])
plt.title('Train vs test Accuracy')

plt.show()


# In[27]:


plt.plot(train_losses)
plt.plot(test_losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['Train','test'])
plt.title('Train vs test loss')

plt.show()


# In[28]:


test_accuarcy = 0.0
for samples, labels in test_loader:
    with torch.no_grad():
        samples, labels = samples, labels
        inputs_m, inputs_s = samples.mean(), samples.std()
        samples = (samples - inputs_m) / inputs_s
        output = model(samples)
        # calculate accuracy
        pred = torch.argmax(output, dim=1)
        correct = pred.eq(labels)
        test_accuarcy += torch.mean(correct.float())
print('Accuracy of the network on {} test images: {}%'.format(len(val_ds), round(test_accuarcy.item()*100.0/len(test_loader), 2)))


# In[29]:


# prepare to count predictions for each class
classes = ('Concrete', 'Flatmountain', 'Grass')

correct_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}


# again no gradients needed
with torch.no_grad():
    for data in test_loader:
        samples, labels = samples, labels
        inputs_m, inputs_s = samples.mean(), samples.std()
        samples = (samples - inputs_m) / inputs_s
        outputs = model(samples)
        _, predictions = torch.max(outputs, 1)
        #collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[classes[label]] += 1
            total_pred[classes[label]] += 1

            

for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')


# In[30]:


#visualize the ouput layer for the test data 

def test2(model, test_loader):
        model.eval()
        test_targets = []
        #test_embeddings = torch.zeros((32), dtype=torch.float32)
        test_embeddings = []
        
        
        with torch.no_grad():
            for x,y in test_loader:
    
                logits =y
                inputs_m, inputs_s = x.mean(), x.std()
                x = (x - inputs_m) / inputs_s
    
                embeddings= model(x)
    
                test_targets.extend(y.detach().cpu().tolist())
                test_embeddings.extend(embeddings.detach().cpu().tolist())
                #test_embeddings = torch.cat((test_embeddings, embeddings.detach().cpu()), 0)


        test_embeddings = np.array(test_embeddings)
        test_targets = np.array(test_targets)
        
        tsne = TSNE(n_components=2, verbose=1, learning_rate= 300, init = 'random')
        tsne_proj = tsne.fit_transform(test_embeddings)#.reshape(1,-1))

        
        cmap = cm.get_cmap('tab10')
        fig, ax = plt.subplots(figsize=(4,4))
        ax.set_title('tSNE Visualization')
        ax.set_xlabel('tSNE_1') 
        ax.set_ylabel('tSNE_2')
  
        ax.grid(False)
        #ax.axis('off')
        mpl.style.use('default')
        num_categories = 3
        for lab in range(num_categories):
            indices = test_targets==lab
            ax.scatter(x=tsne_proj[indices,0],y=tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label = lab ,alpha=0.5)
        ax.legend(fontsize='small', markerscale=2)
        plt.show()
        




for epoch in range(num_epochs):
    test2(model,test_loader)


# In[31]:


# Visualize feature maps
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.conv2.register_forward_hook(get_activation('conv2'))
data= train_loader.dataset[0][0]
data.unsqueeze_(0)
inputs_m, inputs_s = data.mean(), data.std()
data = (data - inputs_m) / inputs_s
        
output = model(data)

act = activation['conv2'].squeeze()
    
    

fig, axarr = plt.subplots(nrows=8, ncols=8, figsize=(18, 6))

k = 0
for idx in range(act.size(0)//8):
    for idy in range(act.size(0)//8):
        axarr[idx, idy].imshow(act[k])
        k += 1


# In[ ]:




