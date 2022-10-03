from skimage import io, transform
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from os import listdir
from os.path import isfile, join
import os
import re

class DrivingDataset(Dataset):
    
    def __init__(self, root_dir, categorical = False, classes=-1, transform=None, min_turns_per_batch=0, batch_size=256):
        """
        root_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = [f for f in listdir(self.root_dir) if f.endswith('jpg')]
        self.categorical = categorical
        self.classes = classes
        # Variables below allow us to get a minimum number of turns per batch
        self.min_turns_per_batch = min_turns_per_batch
        self.samples_served = 0
        self.turns_served_this_batch = 0
        self.batch_size = batch_size
        
    def __len__(self):
        return len(self.filenames)
        
    def __getitem__(self, idx):
        # Check if we need a minimum number of turns in each batch
        need_turn = (
            self.min_turns_per_batch > 0
            and (self.samples_served + self.min_turns_per_batch) >= self.batch_size
            and (self.turns_served_this_batch < self.min_turns_per_batch)
        )

        found_data = False
        while not found_data:
            basename = self.filenames[idx]
            img_name = os.path.join(self.root_dir, basename)
            image = io.imread(img_name)

            m = re.search('expert_([0-9]+)_([0-9]+)_([-+]?\d*\.\d+|\d+).jpg', basename)
            steering_command = np.array(float(m.group(3)), dtype=np.float32)

            # Based on distribution, this seemed like a reasonable cut-off to separate turns from straight driving
            is_turn = np.abs(steering_command) > 0.1
            self.turns_served_this_batch += is_turn

            if self.categorical:
                steering_command = int(((steering_command + 1.0)/2.0) * (self.classes - 1))

            found_data = not need_turn or is_turn
            # If did not find data we wanted, get a random sample
            idx = torch.randint(0, len(self.filenames), (1,)).item()
            
        if self.transform:
            image = self.transform(image)

        self.samples_served += 1
        # If we have served enough samples, reset the counter
        if self.samples_served % self.batch_size == 0:

            self.samples_served = 0
            self.turns_served_this_batch = 0
        
        return {'image': image, 'cmd': steering_command}
        
