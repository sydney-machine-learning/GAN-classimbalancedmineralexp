import pandas as pd
import numpy as np
import torch
from tqdm.auto import tqdm
from torchvision.datasets.utils import download_url
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
from SMOTified_GAN_model import SMOTified_GANs_Discriminator, SMOTified_GANs_Generator
from SMOTified_GANs_trainers import train_discriminator, train_generator

#print("This is train.py file")

def two_classes_Abalone(Abalone_df):
  class_category = np.repeat("empty000", Abalone_df.shape[0])  
  for i in range(0, Abalone_df['Class_number_of_rings'].size):
    if(Abalone_df["Class_number_of_rings"][i] <= 7):
        class_category[i] = int(1)
    elif(Abalone_df["Class_number_of_rings"][i] > 7):
        class_category[i] = int(0)

  Abalone_df = Abalone_df.drop(['Class_number_of_rings'], axis=1)
  Abalone_df['class_category'] = class_category
  return Abalone_df


def four_classes_Abalone(Abalone_df):
  class_category = np.repeat("empty000", Abalone_df.shape[0])  
  for i in range(0, Abalone_df['Class_number_of_rings'].size):
    if(Abalone_df["Class_number_of_rings"][i] <= 7):
        class_category[i] = int(0)
    elif(Abalone_df["Class_number_of_rings"][i] > 7 and Abalone_df["Class_number_of_rings"][i] <= 10):
        class_category[i] = int(1)
    elif(Abalone_df["Class_number_of_rings"][i] > 10 and Abalone_df["Class_number_of_rings"][i] <= 15):
        class_category[i] = int(2)
    else:
        class_category[i] = int(3)

  Abalone_df = Abalone_df.drop(['Class_number_of_rings'], axis=1)
  Abalone_df['class_category'] = class_category
  return Abalone_df




def get_features(Abalone_df, Sex_onehotencoded, test_size):

    features = Abalone_df.iloc[:,np.r_[0:7]]
    X_train, X_test, X_gender, X_gender_test = train_test_split(features, Sex_onehotencoded, random_state=10, test_size=test_size)
    X_train = np.concatenate((X_train.values, X_gender), axis=1)
    X_test = np.concatenate((X_test.values, X_gender_test), axis=1)
    return X_train, X_test




def get_labels(Abalone_df, test_size):
    labels = Abalone_df.iloc[:,7]
    y_train, y_test = train_test_split(labels, random_state=10, test_size=test_size)
    train_list = [int(i) for i in y_train.ravel()] 
    y_train=np.array(train_list)
    test_list = [int(i) for i in y_test.ravel()]    #Flattening the matrix
    y_test=np.array(test_list)

    return y_train, y_test


######### GETTING THE GPU ###########
def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def GANs_train_data(X_train, y_train):   #Defining the real data for GANs
  X_real = []
  y_train = y_train.ravel()
  for i in range(len(y_train)):
    if int(y_train[i])==1:
      X_real.append(X_train[i])
  X_real = np.array(X_real)
  y_real = np.ones((X_real.shape[0],))
  return X_real, y_real


def fit(epochs, lr, discriminator, generator, X_oversampled, train_dl, device, start_idx=1):
    torch.cuda.empty_cache()
    
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []
    
    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        for real_data, _ in tqdm(train_dl):
            # Train discriminator
            train_disc = train_discriminator(real_data, X_oversampled, opt_d, generator, discriminator, device)
            loss_d, real_score, fake_score = train_disc()
            # Train generator
            train_gen = train_generator(X_oversampled, opt_g, generator, discriminator, device)
            loss_g = train_gen()
            
        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)
        
        # Log losses & scores (last batch)
        print("Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
            epoch+1, epochs, loss_g, loss_d, real_score, fake_score))
    
        # Save generated images
        #save_samples(epoch+start_idx, fixed_latent, show=False)
    
    return losses_g, losses_d, real_scores, fake_scores


def main():

    dataset_url =  'https://raw.githubusercontent.com/sydney-machine-learning/GANclassimbalanced/main/DATASETS/abalone_csv.csv'
    download_url(dataset_url, '.')
    Abalone_df  = pd.read_csv('D:/Projects/Internship UNSW/abalone_csv.csv')

    #print(Abalone_df.Class_number_of_rings.size)

    Abalone_df = four_classes_Abalone(Abalone_df)

    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(sparse=False) 
    Sex_labelencoded= label_encoder.fit_transform(Abalone_df['Sex']) 
    Sex_labelencoded = Sex_labelencoded.reshape(len(Sex_labelencoded), 1)
    Sex_onehotencoded = onehot_encoder.fit_transform(Sex_labelencoded)

    Abalone_df = Abalone_df.drop(['Sex'], axis=1)
    
    X_train, X_test = get_features(Abalone_df, Sex_onehotencoded, 0.2)
    y_train, y_test = get_labels(Abalone_df, 0.2)

    '''
    print("Before OverSampling, counts of label '0': {}".format(sum(y_train==0)))
    print("Before OverSampling, counts of label '1': {} \n".format(sum(y_train==1))) 
    print("Before OverSampling, counts of label '2': {} \n".format(sum(y_train==2)))
    print("Before OverSampling, counts of label '3': {} \n".format(sum(y_train==3)))     '''

    X_train_SMOTE,y_train_SMOTE = SMOTE().fit_resample(X_train,y_train)

    print('After OverSampling, the shape of train_X: {}'.format(X_train_SMOTE.shape))
    print('After OverSampling, the shape of train_y: {} \n'.format(y_train_SMOTE.shape))

    print("After OverSampling, counts of label '0': {}".format(sum(y_train_SMOTE==0)))
    print("After OverSampling, counts of label '1': {}".format(sum(y_train_SMOTE==1))) 
    print("After OverSampling, counts of label '2': {}".format(sum(y_train_SMOTE==2))) 
    print("After OverSampling, counts of label '3': {}".format(sum(y_train_SMOTE==3))) 

    device = get_default_device()
    #print(device)
    
    ##### Initialising the generator and discriminator objects ######
    gen1 = SMOTified_GANs_Generator(X_train.shape[1], X_train.shape[1], 128)
    disc1 = SMOTified_GANs_Discriminator(X_train.shape[1], 128)

    ##### Loading the model in GPU #####
    generator = to_device(gen1.generator, device)      
    discriminator = to_device(disc1.discriminator, device)

    ##### Oversampled data from SMOTE that is now to be passed in SMOTified GANs #####
    X_oversampled = X_train_SMOTE[(X_train.shape[0]):]
    X_oversampled = torch.from_numpy(X_oversampled)
    X_oversampled = to_device(X_oversampled.float(), device)

    #print(X_oversampled.shape)

    X_real, y_real = GANs_train_data(X_train, y_train)

    lr = 0.0002
    epochs = 150
    batch_size = 128

    ##### Wrapping all the tensors in a Tensor Dataset. #####
    tensor_x = torch.Tensor(X_real) 
    tensor_y = torch.Tensor(y_real)
    my_dataset = TensorDataset(tensor_x,tensor_y)

    ##### Loading our Tensor Dataset into a Dataloader. #####
    train_dl = DataLoader(my_dataset, batch_size=batch_size, shuffle=True) 
    train_dl = DeviceDataLoader(train_dl, device)


    history = fit(epochs, lr, discriminator, generator, X_oversampled, train_dl, device)


main()