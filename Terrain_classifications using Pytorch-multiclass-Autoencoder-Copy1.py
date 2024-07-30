#!/usr/bin/env python

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
import torch.utils.data as data_utils
import torch.nn.functional as F

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
#from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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


dset=pd.DataFrame()
dset['ID']=ID
dset['label']=label
dset['ClassID']=ClassID


dset.head(599)

dset.to_csv('C:/Users/Engem/Downloads/data/data.csv')

df = pd.read_csv('C:/Users/Engem/Downloads/data/data.csv')

df['ID'].loc[150]

df['ID']

Grass = 'C:/Users/Engem/Downloads/data/GroundType1/Grass/'
path=os.path.join(Grass, df['ID'].loc[595])
print(path)

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

plt.figure(figsize=(8,6),dpi=80)
sn.set_theme(style="darkgrid")
sn.countplot(x ='label',data=dset)
plt.title('counts: \n' +'Concrete:'+str(dset.label.value_counts()[0])+'\n Flatmountain:'+str(dset.label.value_counts()[1])+'\n Grass:'+str(dset.label.value_counts()[2]))
plt.show()

print(dset.label.unique())

print("Number of training examples=", dset.shape[0], "  Number of classes=", len(dset.label.unique()))

# Read file

df['ID'] = df['label'].astype(str) + '/' + df['ID'].astype(str)
# Take relevant columns
df = df[['ID', 'ClassID']]
df.head(350)


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
      #inputs_m, inputs_s = sgram_features.mean(), sgram_features.std()
      #sgram_features = (sgram_features - inputs_m) / inputs_s


      return  sgram_features,  class_id


myds = SoundDS(df, dataset)

print(myds[20])

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


examples = next(iter(train_loader))
data, targets= examples
print(data.shape, targets.shape)
#print(' '.join(f'{classes[targets[j]]:5s}' for j in range(batch_size)))



examples = next(iter(test_loader))
example_data,example_targets= examples
print(example_data.shape, example_targets.shape)
#print(' '.join(f'{classes[example_targets[j]]:5s}' for j in range(batch_size)))


#Utility functions to un-normalize and display an image
def imshow(img):
    img = img / 2 + 0.5  
    plt.imshow(np.transpose(img, (1, 2, 0))) 

 
#Define the image classes
classes = ['concrete', 'flat', 'grass']


#Obtain one batch of training images
dataiter = next(iter(train_loader))
images, labels = dataiter
images = images.numpy() # convert images to numpy for display

#Plot the images
fig = plt.figure(figsize=(8, 8))
# display 20 images
for idx in np.arange(9):
    ax = fig.add_subplot(3, 3, idx+1, xticks=[], yticks=[])
    imshow(images[idx])
    ax.set_title(classes[labels[idx]])



data2 = data[0].reshape(len(data[0]),-1)

print(data[0].shape)
print(data2)
print(data2.shape)


conv1= nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1)
conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=2, padding=(1, 1))
conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=2, padding=0)
pool= nn.BatchNorm2d(16)
x2=nn.BatchNorm2d(16)
x=conv1(example_data)
x1=conv2(x)
x2=pool(x1)
x3=conv3(x2)
print(x.shape)
print(x1.shape)
print(x2.shape)
print(x3.shape)


class Encoder(nn.Module):
    
    def __init__(self, encoded_space_dim=128,fc2_input_dim=128):
        super().__init__()
        
        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 3, stride=2, padding=0),
            nn.ReLU(True)
        )
        
        ### Flatten layer
        self.flatten = nn.Flatten(start_dim=1)
        ### Linear section
        self.encoder_lin = nn.Sequential(nn.Linear(46 * 7 * 32, 32),
            nn.ReLU(True),
            nn.Linear(32, 128)
        )
        
    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x

class Decoder(nn.Module):
    
    def __init__(self, encoded_space_dim, fc2_input_dim):
        super().__init__()
        self.decoder_lin = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(True),
            nn.Linear(32, 7 * 46 * 32),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, 
        unflattened_size=(32, 7, 46))

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 4, stride=2, 
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 5, stride=2, padding=1, output_padding=1)
        )
        
    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x


#Loss function
loss_fn =  nn.BCELoss()


### Define an optimizer (both for the encoder and the decoder!)
lr= 0.001

### Set the random seed for reproducible results
torch.manual_seed(0)

### Initialize the two networks
#d = 4

#model = Autoencoder(encoded_space_dim=encoded_space_dim)
encoder = Encoder(encoded_space_dim=128,fc2_input_dim=128)
#encoder= encoder.numpy()
decoder = Decoder(encoded_space_dim=118,fc2_input_dim=128)
params_to_optimize = [{'params': encoder.parameters()},{'params': decoder.parameters()}]

print(encoder)

#Optimizer
optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

# Check if the GPU is available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f'Selected device: {device}')

# Move both the encoder and the decoder to the selected device
encoder.to(device)
decoder.to(device)


### Training function
def train_epoch(encoder, decoder, device, data2, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    train_loss = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in train_loader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
     
        # Encode data
        encoded_data = encoder(image_batch)
        # Decode data
        decoded_data = decoder(encoded_data)
        # Evaluate loss
        loss = loss_fn(decoded_data, image_batch)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Print batch loss
        print('\t partial train loss (single batch): %f' % (loss.data))
        train_loss.append(loss.detach().cpu().numpy())

    return np.mean(train_loss)


### Testing function
def test_epoch(encoder, decoder, device, test_loader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()
    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in test_loader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
            
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label) 
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
        
    return val_loss


data= train_loader.dataset[89][0]
print(data)



data, targets = train_loader.dataset[89]
print(targets)



def plot_ae_outputs(encoder,decoder,n=4):
    classes = ['concrete', 'flat', 'grass']
    print("Original Images")
    fig = plt.figure(figsize=(16, 4))
    
    
    for i in range(n):
      ax = plt.subplot(2,n,i+1)
      img =  test_loader.dataset[i][0].unsqueeze(0).to(device)
      data, targets = train_loader.dataset[i]
      encoder.eval()
      decoder.eval()
      with torch.no_grad():
          rec_img  = decoder(encoder(img))
        
      ax.set_title(classes[targets])
      plt.imshow(img.cpu().squeeze().numpy())
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
    
  
      ax = plt.subplot(2, n, i+1+n)
      plt.imshow(rec_img.cpu().squeeze().numpy())  
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)  
        
      #if i == n//6:
         #ax.set_title('Reconstructed images')
      ax.set_title(classes[targets])
    plt.show()  

num_epochs = 40
diz_loss = {'train_loss':[],'val_loss':[]}
for epoch in range(num_epochs):
   train_loss =train_epoch(encoder,decoder,device,
   train_loader,loss_fn,optim)
   val_loss = test_epoch(encoder,decoder,device,test_loader,loss_fn)
   print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch + 1, num_epochs,train_loss,val_loss))
   diz_loss['train_loss'].append(train_loss)
   diz_loss['val_loss'].append(val_loss)
   plot_ae_outputs(encoder,decoder,n=4)


test_epoch(encoder,decoder,device,test_loader,loss_fn).item()


from tqdm import tqdm

encoded_samples = []


for sample in tqdm (val_ds):
    
    img = sample[0].unsqueeze(0).to(device)
    label = sample[1]
    # Encode image
    encoder.eval()
    with torch.no_grad():
        encoded_img  = encoder(img)
    # Append to list
    encoded_img = encoded_img.flatten().cpu().numpy()
    encoded_sample = {f"Enc. Variable {i}": enc for i, enc in enumerate(encoded_img)}
    encoded_sample['label'] = label
    encoded_samples.append(encoded_sample)
encoded_samples = pd.DataFrame(encoded_samples)

encoded_samples

#list(encoded_samples.values[0])


import plotly.express as px

px.scatter(encoded_samples, x='Enc. Variable 0', y='Enc. Variable 1', 
           color=encoded_samples.label.astype(str), opacity=0.7)



from sklearn.manifold import TSNE

tsne = TSNE(n_components=2)
tsne_results = tsne.fit_transform(encoded_samples.drop(['label'],axis=1))
fig = px.scatter(tsne_results, x=0, y=1,
                 color=encoded_samples.label.astype(str),
                 labels={'0': 'tsne-2d-one', '1': 'tsne-2d-two'})
fig.show()





