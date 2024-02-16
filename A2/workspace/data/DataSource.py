import os
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image
import numpy as np


class DataSource(data.Dataset):
    def __init__(self, root, resize=256, crop_size=224, train=True):
        self.root = os.path.expanduser(root)
        self.resize = resize
        self.crop_size = crop_size
        self.train = train

        self.image_poses = []
        self.images_path = []

        self._get_data()

        # TODO: Define preprocessing
        if self.train:
            self.preprocess = T.Compose([
                T.ToTensor(),
                T.RandomCrop(self.crop_size),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        else:
            self.preprocess = T.Compose([
                T.ToTensor(),
                T.CenterCrop(self.crop_size),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ])
        # Load mean image
        self.mean_image_path = os.path.join(self.root, 'mean_image.npy')
        if os.path.exists(self.mean_image_path):
            self.mean_image = np.load(self.mean_image_path)
            print("Mean image loaded!")
        else:
            self.mean_image = self.generate_mean_image()

    def _get_data(self):

        if self.train:
            txt_file = self.root + 'dataset_train.txt'
        else:
            txt_file = self.root + 'dataset_test.txt'

        with open(txt_file, 'r') as f:
            next(f)  # skip the 3 header lines
            next(f)
            next(f)
            for line in f:
                fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)
                self.image_poses.append((p0, p1, p2, p3, p4, p5, p6))
                self.images_path.append(self.root + fname)

    def resize_shorter_side(self,img, shorter_side=256):
        width, height = img.size

        # Determine the scale factor, keeping the aspect ratio
        if height < width:
            scale_factor = shorter_side / height
            new_height = shorter_side
            new_width = int(width * scale_factor)
        else:
            scale_factor = shorter_side / width
            new_width = shorter_side
            new_height = int(height * scale_factor)

        # Resize the image using the calculated dimensions
        resized_image = img.resize((new_width, new_height))

        return resized_image

    def generate_mean_image(self):
        print("Computing mean image:")
        # TODO: Compute mean image
        # Initialize mean_image
        num_image = len(self.images_path)
        sum_image = None
        # Iterate over all training images
        for img_path in self.images_path:
            with Image.open(img_path) as img:
                # Resize, Compute mean, etc...
                img_resized = self.resize_shorter_side(img, self.resize)
                img_array = np.array(img_resized, dtype=np.float32)
                if sum_image is None:
                    sum_image = np.zeros_like(img_array)
            sum_image += img_array
        # Resize, Compute mean, etc...
        mean_image = sum_image / num_image
        # Store mean image
        np.save(os.path.join(self.root, 'mean_image.npy'), mean_image)
        print("Mean image computed!")

        return mean_image

    def __getitem__(self, index):
        """
        return the data of one image
        """
        img_path = self.images_path[index]
        img_pose = self.image_poses[index]

        data = Image.open(img_path)

        # TODO: Perform preprocessing
        data = self.resize_shorter_side(data, self.resize)
        data = np.array(data, dtype=np.float64) - self.mean_image
        data = self.preprocess(data)

        return data, img_pose

    def __len__(self):
        return len(self.images_path)

