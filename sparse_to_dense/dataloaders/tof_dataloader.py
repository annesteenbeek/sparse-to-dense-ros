import numpy as np
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader
import h5py

iheight, iwidth = 480, 640 # raw image size

def h5_tof_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    tof = h5f['tof']
    return rgb, depth, tof

to_tensor = transforms.ToTensor()

class TOFDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb', augArgs=None, loader=h5_tof_loader):
        super(TOFDataset, self).__init__(root, type, sparsifier, modality, augArgs, loader)
        self.output_size = (228, 304)

    def __getraw__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (rgb, depth) the raw data.
        """
        path, target = self.imgs[index]
        rgb, depth, tof = self.loader(path)
        return rgb, depth, tof

    def __getitem__(self, index):
        rgb, depth, tof = self.__getraw__(index)
        augArgs = self.augArgs
        if self.transform is not None:
            rgb_np, depth_np = self.transform(rgb, depth)
        else:
            raise(RuntimeError("transform not defined"))

        # color normalization
        # rgb_tensor = normalize_rgb(rgb_tensor)
        # rgb_np = normalize_np(rgb_np)

        if self.modality == 'rgb':
            input_np = rgb_np
        elif self.modality == 'rgbd':
            input_np = self.create_rgbd(rgb_np, depth_np, tof)

        input_tensor = to_tensor(input_np)
        while input_tensor.dim() < 3:
            input_tensor = input_tensor.unsqueeze(0)
        depth_tensor = to_tensor(depth_np)
        depth_tensor = depth_tensor.unsqueeze(0)

        return input_tensor, depth_tensor

    def create_rgbd(self, rgb, depth, tof):
        sparse_depth = self.create_sparse_depth(rgb, depth, tof)
        rgbd = np.append(rgb, np.expand_dims(sparse_depth, axis=2), axis=2)
        return rgbd

    def train_transform(self, rgb, depth):
        s = self.getFocalScale()

        if(self.augArgs.varFocus): #Variable focal length simulation
            depth_np = depth
        else:
            depth_np = depth / s #Correct for focal length

        if(self.augArgs.varScale): #Variable global scale simulation
            scale = self.getDepthGroup()
            depth_np = depth_np*scale

        angle = np.random.uniform(-5.0, 5.0) # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5 # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Resize(250.0 / iheight), # this is for computational efficiency, since rotation can be slow
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        s = self.getFocalScale()

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

        return rgb_np, depth_np

    def create_sparse_depth(self, rgb, depth, tof):
        #Sparsifier is hard coded for this dataset, the statsam mask is used
        mask_keep = self.sparsifier.dense_to_sparse(rgb, depth)
        sparse_depth = np.zeros(depth.shape)
        sparse_depth[mask_keep] = tof
        return sparse_depth
