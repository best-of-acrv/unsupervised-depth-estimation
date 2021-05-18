import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.misc import imread, imresize
from collections import Counter
from scipy.interpolate import LinearNDInterpolator as LinearNDInterpolator
import extract_depth

class KITTILoader(Dataset):

    def __init__(self, root, image_files, left_subdir="image_02", right_subdir="image_03", img_fmt="png", depth=False, inv_depth=False):

        self.root = root
        self.left_subdir = left_subdir
        self.right_subdir = right_subdir
        self.img_fmt = img_fmt
        self.depth = depth
        self.inv_depth = inv_depth

        #Read in image files
        f = open(image_files, "r")
        self.files = f.read().splitlines()
        f.close()

        #Split directories/image IDs
        for i in range(len(self.files)):
            self.files[i] = self.files[i].split(" ")

    def process_image(self, read_file):
        image = imread(read_file)

        orig_im_size = image.shape[:2]

        # Resize to be 160 by 608 pixels.
        image = imresize(image, (160, 608))
        image_BGR = image[..., ::-1]  # ::-1 inverts the order of the last dimension (channels). img = img[:, :, : :-1] is equivalent to img = img[:, :, [2,1,0]].
        image_BGR = np.transpose(image_BGR, (1, 0, 2))
        image_BGR.astype(float)
        # mean_data = np.mean(image_BGR)
        mean_data = np.tile(np.reshape([104, 117, 123], (1, 1, 3), order='F'),
                               (image_BGR.shape[0], image_BGR.shape[1], 1))
        image_BGR = image_BGR - mean_data
        image_BGR = np.transpose(image_BGR, (2, 1, 0))  # (W, H, C) -> (C, H, W)

        return torch.from_numpy(image_BGR).float(), orig_im_size
        
    def __getitem__(self, index):
        img_left, orig_im_size= self.process_image("{}/{}/{}/data/{}.{}".format(self.root, self.files[index][0], self.left_subdir, self.files[index][1], self.img_fmt))
        img_right, _ = self.process_image("{}/{}/{}/data/{}.{}".format(self.root, self.files[index][0], self.right_subdir, self.files[index][1], self.img_fmt))
        
        if self.depth:
            calib_dir = "{}/{}".format(self.root, self.files[index][0].split("/")[0])
            velo_file_name = "{}/{}/velodyne_points/data/{}.bin".format(self.root, self.files[index][0], self.files[index][1])

            #Interp should be False for computing errors, can be True for visualisation
            depth_map = extract_depth.get_depth(calib_dir, velo_file_name, orig_im_size, cam=2, interp=False, vel_depth=False, inv_depth=self.inv_depth)
            depth_map = torch.from_numpy(depth_map).unsqueeze(0).type(torch.FloatTensor)
        else:
            depth_map = torch.Tensor([-1])
        
        return img_left, img_right, self.files[index], depth_map

    def __len__(self):
        return len(self.files)