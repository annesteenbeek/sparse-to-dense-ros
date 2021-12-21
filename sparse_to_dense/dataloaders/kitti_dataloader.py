import numpy as np
import dataloaders.transforms as transforms
from dataloaders.dataloader import MyDataloader

class KITTIDataset(MyDataloader):
    def __init__(self, root, type, sparsifier=None, modality='rgb', augArgs=None):
        super(KITTIDataset, self).__init__(root, type, sparsifier, modality, augArgs)
        self.output_size = (228, 912)

    def train_transform(self, rgb, depth):
        #s = np.random.uniform(1.0, 1.5)  # random scaling
        #depth_np = depth / s
        s = self.getFocalScale()

        if(self.augArgs.varFocus): #Variable focal length simulation
            depth_np = depth
        else:
            depth_np = depth / s #Correct for focal length

        if(self.augArgs.varScale): #Variable global scale simulation
            scale = self.getDepthGroup()
            depth_np = depth_np*scale        

        angle = np.random.uniform(-5.0, 5.0)  # random rotation degrees
        do_flip = np.random.uniform(0.0, 1.0) < 0.5  # random horizontal flip

        # perform 1st step of data augmentation
        transform = transforms.Compose([
            transforms.Crop(130, 10, 240, 1200),
            transforms.Rotate(angle),
            transforms.Resize(s),
            transforms.CenterCrop(self.output_size),
            transforms.HorizontalFlip(do_flip)
        ])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np) # random color jittering
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        # Scipy affine_transform produced RuntimeError when the depth map was
        # given as a 'numpy.ndarray'
        depth_np = np.asfarray(depth_np, dtype='float32')
        depth_np = transform(depth_np)

        return rgb_np, depth_np

    def val_transform(self, rgb, depth):
        s = self.getFocalScale()

        depth = np.asfarray(depth, dtype='float32') #This used to be the last step, not sure if it goes here?
        if(self.augArgs.varScale): #Variable global scale simulation
            scale = self.getDepthGroup()
            depth_np = depth*scale
        else:
            depth_np = depth

        if(self.augArgs.varFocus):
            transform = transforms.Compose([
                transforms.Crop(130, 10, 240, 1200),
                transforms.Resize(s), #Resize both images without correcting the depth values
                transforms.CenterCrop(self.output_size),
            ])
        else:
            transform = transforms.Compose([
                transforms.Crop(130, 10, 240, 1200),
                transforms.CenterCrop(self.output_size),
            ])

        rgb_np = transform(rgb)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)
        return rgb_np, depth_np

