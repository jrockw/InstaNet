from sklearn.cluster import DBSCAN
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import os
import scipy.misc
workers = 0 if os.name == 'nt' else 4
mtcnn = MTCNN(keep_all=True,prewhiten=True)
resnet = InceptionResnetV1(pretrained='vggface2').eval()
def meanArr(lis):
    lis = lis.detach()
    sumElt = np.zeros(len(lis[0]))
    for point in lis:
        sumElt = np.add(sumElt, point)
    sumElt = sumElt / len(lis)
    return sumElt
def readImage():
    def openImage(path):
        try:
            img = Image.open(path).convert('RGB')
            return img
        except IOError:
            print("nice try, come again!")
            return False
    img = False
    while (img is False ):
        path = input("Enter image path: ")
        img = openImage(path)
    return img
def collate_fn(x):
    return x[0]
def readFolder(dataset,folder):
    dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
    return loader
def addUserToList(db, embNump, userProfiles, userNames, username):
    db = db.labels_
    print(db)
    print(userProfiles.shape)
    userProfiles = userProfiles.tolist()
    userNames = userNames.tolist()
    for i, val in enumerate(db):
        if (val == 0):
            userProfiles.append(embNump[i])
            userNames.append(username)
    return userProfiles, userNames
folder = input("Enter user folder path: ")
if folder == '':
    folder = '/Users/User/Documents/Personal/Projects/FACIAL/jimParsed'

dataset = datasets.ImageFolder(folder)
loader = readFolder(dataset,folder) 
aligned = []
names = []
i = 0
username = input("Enter username:")
def addUser(username):
    for x, y in loader: 
    #     save_name = username +'_'+str(i) + '.png'
        x_aligned, prob = mtcnn(x, return_prob=True)
    # save all the faces
        if x_aligned is not None:
    #        print('Face detected with probability: {:8f}'.format(prob))
            for z in x_aligned:
                aligned.append(z)
                names.append(dataset.idx_to_class[y])
            i += 1
    aligned = torch.stack(aligned)
    
    embeddings = resnet(aligned)
    embNump = np.zeros((len(embeddings), len(embeddings[0])))
    for i, val in enumerate(embeddings):
        embNump[i] = val.detach().numpy()
        print(embNump.size)
    print("Number of Faces: ")
    print(len(embNump))
    print("Clustering now")
   #cluster faces 
    db = DBSCAN(eps=0.8).fit(embNump)
    #load files
    userProfiles = np.load('faceDictionary.npy')
    userNames = np.load('userNames.npy')
   #add users to list 
    userProfiles, userNames = addUserToList(db, embNump, userProfiles, userNames, username)
    np.save('faceDictionary.npy', userProfiles)
    np.save('userNames.npy', userNames)
if input("Get features? y/n") == 'y':
    for x, y in loader: 
#        save_name = username +'_'+str(i) + '.png'
        x_aligned, prob = mtcnn(x, return_prob=True)
    # save all the faces
        if x_aligned is not None:
    #        print('Face detected with probability: {:8f}'.format(prob))
            for z in x_aligned:
                aligned.append(z)
                names.append(dataset.idx_to_class[y])
            i += 1
    aligned = torch.stack(aligned)
    
    embeddings = resnet(aligned)
    embNump = np.zeros((len(embeddings), len(embeddings[0])))
    for i, val in enumerate(embeddings):
        embNump[i] = val.detach().numpy()
        print(embNump.size)
    if input("Save features to file? y/n") == 'y':
        filename = username + '.npy'
        np.save(filename, embNump)
elif input("Load features from data? y/n") == 'y':
    inFile = input("Filename: ")
    embNump = np.load(inFile)

print("Number of Faces: ")
print(len(embNump))
print("Clustering now")
#compute embeddings
#FIND distance matrix: 
#dists =  [[(e1 - e2).norm().item() for e1 in embeddings]for e2 in embeddings]
db = DBSCAN(eps=0.8).fit(embNump)
print("Labels: ")
print(db.labels_)
userProfiles = np.empty(0)
userNames = np.empty(0)
if input("load userProfiles? y/n") == 'y':
    userProfiles = np.load('faceDictionary.npy')
    userNames = np.load('userNames.npy')
userProfiles, userNames = addUserToList(db, embNump, userProfiles, userNames, username)
print(len(userProfiles))
#call function
print("Saving new user profiles...")
np.save('faceDictionary.npy', userProfiles)
np.save('userNames.npy', userNames)
print("Completed adding: " + username)
#dists = [[(e1 - e2).norm().item() for e2 in embeddings]for e1 in embeddings]
#average = meanArr(embeddings)
#print("average: ")
#print(average)
#def jimID():
#    img = readImage()
#    img_resize = mtcnn(img)
#    img_embeddings = resnet(img_resize)
#    print("Similarity")
#    similarity = (img_embeddings- average).norm().item()
#    print(similarity)
#    if (similarity > 0.95):
#        print("probably not " + username)
#    else:
#        print("It's " +username+"!")
#cont = True
#while (cont):
#    jimID()
#    cont = input("continue? (y/n)")
#    if (cont != "y"): cont = False
#
#dists = [(e1 - average).norm().item() for e1 in embeddings]
minimum = 10
index = -1
#for indice, val in enumerate(dists):
#    print(val)
#    if (val < minimum):
#        minimum = dists[indice]
#        index = indice
#
#print('Result: ')
#print(names[index])
#print('Distance')
#print(minimum)
