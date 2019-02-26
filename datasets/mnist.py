from __future__ import print_function
from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data

import numpy as np
from random import shuffle

class MNISTInstance(datasets.MNIST):
    """MNIST Instance Dataset.
    """

    def __init__(self, subset_ratio=1.0, classes_ratio=1.0, batch_size=32, *args, **kwargs):
        super(MNISTInstance, self).__init__(*args, **kwargs)

        self.subset_ratio = subset_ratio
        self.classes_ratio = classes_ratio
        self.batch_size = batch_size

        self.train_indices = None 
        self.train_index = 0
        
        self.test_indices = None
        self.test_index = 0

        np.random.seed(232323)
        self.classes = np.random.choice(10, int(10*self.classes_ratio), replace=False)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        if self.train:                
            if self.train_index == 0:
                train_inds = [i for i in range(len(self.train_labels)) if self.train_labels[i] in self.classes]

                np.random.seed(232323)
                self.train_indices = np.random.choice(train_inds, int(self.subset_ratio*len(train_inds)), replace=False)
                shuffle(self.train_indices)
            
            index = self.train_indices[self.train_index]
            img, target = self.train_data[index], self.train_labels[index]

            self.train_index = self.train_index + 1

            if self.train_index % self.batch_size == 0 and self.train_index + self.batch_size >= len(self.train_indices):
                self.train_index = 0
        else:
            if self.test_index == 0:
                test_inds = [i for i in range(len(self.test_labels)) if self.test_labels[i] in self.classes]
                
                self.test_indices = np.random.choice(test_inds, int(self.subset_ratio*len(test_inds)), replace=False)
                
            index = self.test_indices[self.test_index]
            img, target = self.test_data[index], self.test_labels[index]
            self.test_index = self.test_index + 1

            if self.test_index % self.batch_size == 0 and self.test_index + self.batch_size >= len(self.test_indices):
                self.test_index = 0

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
