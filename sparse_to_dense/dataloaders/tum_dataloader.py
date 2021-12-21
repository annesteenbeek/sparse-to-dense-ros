import argparse
import sys
import os
import numpy as np
from PIL import Image
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader

iheight, iwidth = 480, 640 # raw image size


# taken from associate.py by tum
def read_file_list(filename):
    """
    Reads a trajectory from a text file. 
    
    File format:
    The file format is "stamp d1 d2 d3 ...", where stamp denotes the time stamp (to be matched)
    and "d1 d2 d3.." is arbitary data (e.g., a 3D position and 3D orientation) associated to this timestamp. 
    
    Input:
    filename -- File name
    
    Output:
    dict -- dictionary of (stamp,data) tuples
    
    """
    file = open(filename)
    data = file.read()
    lines = data.replace(","," ").replace("\t"," ").split("\n") 
    list = [[v.strip() for v in line.split(" ") if v.strip()!=""] for line in lines if len(line)>0 and line[0]!="#"]
    list = [(float(l[0]),l[1:]) for l in list if len(l)>1]
    return dict(list)

def associate(first_list, second_list,offset,max_difference):
    """
    Associate two dictionaries of (stamp,data). As the time stamps never match exactly, we aim 
    to find the closest match for every input tuple.
    
    Input:
    first_list -- first dictionary of (stamp,data) tuples
    second_list -- second dictionary of (stamp,data) tuples
    offset -- time offset between both dictionaries (e.g., to model the delay between the sensors)
    max_difference -- search radius for candidate generation
    Output:
    matches -- list of matched tuples ((stamp1,data1),(stamp2,data2))
    
    """
    first_keys = first_list.keys()
    second_keys = second_list.keys()
    potential_matches = [(abs(a - (b + offset)), a, b) 
                         for a in first_keys 
                         for b in second_keys 
                         if abs(a - (b + offset)) < max_difference]
    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in first_keys and b in second_keys:
            first_keys.remove(a)
            second_keys.remove(b)
            matches.append((a, b))
    
    matches.sort()
    return matches

def make_dataset(dir):
    images = []
    offset = 0. # time to add
    max_difference = 0.02

    first_list = read_file_list(os.path.join(dir,'rgb.txt')) # rgb
    second_list = read_file_list(os.path.join(dir,'depth.txt')) # depth

    matches = associate(first_list,second_list,offset,max_difference)    
    for a, b in matches:
        rgb_path = os.path.join(dir," ".join(first_list[a]))
        depth_path = os.path.join(dir," ".join(second_list[b]))
        images.append((rgb_path, depth_path))

    return images

def tum_loader(paths):
    rgb_path, depth_path = paths
    rgb = Image.open(rgb_path)
    rgb = np.array(rgb)
    depth = Image.open(depth_path)
    depth = np.array(depth)

    return rgb, depth


class TUMDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb', augArgs=None):
        # super(TUMDataset, self).__init__(root, type, sparsifier, modality, augArgs)
        imgs = make_dataset(root)
        assert len(imgs)>0, "Found 0 images in subfolders of: " + root + "\n"
        print("Found {} images in {} folder.".format(len(imgs), type))
        self.root = root
        self.imgs = imgs
        self.augArgs = augArgs
        if type == 'train':
            self.transform = self.train_transform
        elif type == 'val':
            self.transform = self.val_transform
        else:
            raise (RuntimeError("Invalid dataset type: " + type + "\n"
                                "Supported dataset types are: train, val"))
        self.loader = tum_loader
        self.sparsifier = sparsifier

        assert (modality in self.modality_names), "Invalid modality type: " + modality + "\n" + \
                                "Supported dataset types are: " + ''.join(self.modality_names)
        self.modality = modality

        self.output_size = (228, 304)

    def val_transform(self, rgb, depth):
        s = self.getFocalScale()

        # print("--------------")
        # print("rgb shape: " + str(rgb.shape))
        # print("depth shape: " + str(depth.shape))
 
        if(self.augArgs.varScale): #Variable global scale simulation
            scale = self.getDepthGroup()
            depth_np = depth*scale
        else:
            depth_np = depth

        if(self.augArgs.varFocus):
            transform = transforms.Compose([
                transforms.Resize(240.0 / iheight),
                transforms.Resize(s), #Resize both images without correcting the depth values
                transforms.CenterCrop(self.output_size),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(240.0 / iheight),
                transforms.CenterCrop(self.output_size),
            ])

        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)
        depth_np = np.asfarray(depth_np, dtype='float') / 5000 # depth scale ratio tum
        # print(depth_np[20:22, 20:22])

        return rgb_np, depth_np

    def __getraw__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """
        paths = self.imgs[index]
        rgb, depth = self.loader(paths)
        return rgb, depth


