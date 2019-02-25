from __future__ import print_function
from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data

class MNISTInstance(datasets.MNIST):
    """MNIST Instance Dataset.
    """

    def __init__(self, subset_ratio=1.0, *args, **kwargs):
        super(MySubClassBetter, self).__init__(*args, **kwargs)

        self.subset_ratio = subset_ratio

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        if self.train:
            index = index % int(self.subset_ratio*len(self.train_data))
            img, target = self.train_data[index], self.train_labels[index]
        else:
            index = index % int(self.subset_ratio*len(self.test_data))
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
