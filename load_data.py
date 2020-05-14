from sklearn.cluster import DBSCAN
from facenet_pytorch import MTCNN, InceptionResnetV1
from glob import glob
from getpass import getpass
from instaloader import ConnectionException, Instaloader, Profile
import instaloader
from os import mkdir, getcwd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import os
import scipy.misc
def loginUser(session):
    L = session
    USER = input("Username: ")
    password = getpass()
    
    try:
        L.login(USER, password)
    except InvalidArgumentException: 
        raise SystemExit("ERROR: Invalid username")
    except BadCredentialsException:
        raise SystemExit("ERROR: Invalid password")
    except ConnectionException:
        raise SystemExit("ERROR: Conneciton failed")
    except TwoFactorAuthRequiredException:
        print("TODO: Two factor Auth required, not implemented yet")
        #Instaloader.two_factor_login()

def getProfilePosts(USERNAME):
    #mkdir("training/" + USERNAME)
    #save_path =  'training/' + USERNAME 
    profile = Profile.from_username(L.context, USERNAME)
    save_target = USERNAME
    for post in profile.get_posts():
        #, target=save_target
        if(not L.download_post(post)):
            print("ERROR: Problem downloading post - Line 30")

def collate_fn(x):
    return x[0]
def readFolder(dataset,folder):
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
    return loader
#MAIN
workers = 0 if os.name == 'nt' else 4
mtcnn = MTCNN(prewhiten=True, select_largest=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

save_path = 'training/' 
L = instaloader.Instaloader(dirname_pattern={profile})
loginUser(L)
cont = True
while(cont):
    user = input('Enter username to download:')
    getProfilePosts(user)
    if (input("Continue? (y/n)") != 'y'): cont = False

#compute embeddings
#FIND distance matrix: 
##dists =  [[(e1 - average).norm().item() for e1 in embeddings]for e2 in embeddings]
##db = DBSCAN(eps=0.8).fit(dists)
##print("Labels: ")
##print(db.labels_)
##print("Core sample indices")
##print(db.core_sample_indices_)

