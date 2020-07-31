#!/usr/bin/python3

import os
import torch
import matplotlib.pyplot as plt
from scipy.misc import imread, imresize
import numpy
from pytorch_net import *
import tarfile
from run_pytorch_model import RunPytorchModel
import argparse
import kitti_loader
from torch.utils.data import DataLoader

def init_weights_xavier(m):
    if (type(m) == torch.nn.Linear) or (type(m) == torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.0)

class TrainEvalPytorchModel(RunPytorchModel):
    def __init__(self, kitti_root, save_root, save_name, gpu_id):
        super(TrainEvalPytorchModel, self).__init__()

        self.save_root = save_root
        self.save_name = save_name
        self.kitti_root = kitti_root

        if gpu_id is None:
            self.device = torch.device('cpu')
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
            self.device = torch.device('cuda:0')

        self.smoothness_factor = 0.05
        self.train_batch_size = 24
        self.eval_batch_size = 1 #Other eval batch sizes may cause error

    def train(self, train_image_files, val_image_files=None):
        
        self.pytorch_model = PytorchNet("../../pycaffe_version/network/deploy_resnet50by2_pool.prototxt").to(self.device)
        self.pytorch_model.apply(init_weights_xavier)

        #Load data loaders from training (and validation)
        self.train_loader = self.get_data_loader(train_image_files, batch_size=self.train_batch_size)
        if val_image_files is not None:
            self.val_loader = self.get_data_loader(val_image_files, batch_size=self.train_batch_size)
        else: 
            self.val_loader = None

        #Train
        self.train_pytorch_model()

    def train_pytorch_model(self, num_epochs=100):
        if self.pytorch_model:
            #For reconstruction loss
            self.l1_loss = torch.nn.L1Loss() 

            #For smoothness loss
            self.get_edge_convs() 

            self.optimiser = torch.optim.Adam(self.pytorch_model.parameters(), lr=1e-3)
            self.pytorch_model.train()
            for epoch in range(num_epochs):
                self.train_loop()
                if self.val_loader is not None:
                    self.validate()
        else:
            raise ValueError("Are you sure the PyTorch model is loaded? Model (self.pytorch_model) not found.")

    def train_loop(self):
        loss_running = 0.0
        for index, data in enumerate(self.train_loader):
            images_left = data[0].to(self.device) * 0.004 #Caffe version has a fixed scale layer with param=0.004
            images_right = data[1].to(self.device) * 0.004 #Caffe version has a fixed scale layer with param=0.004
            self.optimiser.zero_grad()

            #Forward pass through network
            outputs = self.pytorch_model.forward(images_left)['h_flow']

            #Perform inverse warp
            warped = self.warp(images_right, outputs)

            #Loss is combination of reconstruction error and smoothness regulariser 
            loss = self.l1_loss(warped, images_left) + self.smoothness_factor*self.smoothness_loss(outputs, images_left)

            #Update network weights
            loss.backward()
            self.optimiser.step()
            loss_running += loss.item()
        
        #Save model and print loss
        torch.save(self.pytorch_model, self.save_root + "/" + self.save_name)
        print("[{0} / {1}] {2:.6f}".format(index, len(self.train_loader), loss_running/len(self.train_loader) ))

    def get_edge_convs(self):
        # Edge filters
        self.edge_conv_x_3 = torch.nn.Conv2d(3,1,3, bias=False).to(self.device)
        self.edge_conv_y_3 = torch.nn.Conv2d(3,1,3, bias=False).to(self.device)
        self.edge_conv_x_1 = torch.nn.Conv2d(1,1,3, bias=False).to(self.device)
        self.edge_conv_y_1 = torch.nn.Conv2d(1,1,3, bias=False).to(self.device)

        #Set layer weights to be edge filters
        with torch.no_grad():
            for layer in [self.edge_conv_x_3, self.edge_conv_x_1]:
                for ch in range(layer.weight.size(1)):
                    layer.weight[0,ch] = torch.Tensor([[0,0,0],[-0.5,0,0.5],[0,0,0]]).to(self.device)

            for layer in [self.edge_conv_y_3, self.edge_conv_y_1]:
                for ch in range(layer.weight.size(1)):
                    layer.weight[0,ch] = torch.Tensor([[0,-0.5,0],[0,0,0],[0,0.5,0]]).to(self.device)


    def smoothness_loss(self, disparity, image):
        edge_x_im = torch.exp( (self.edge_conv_x_3(image).abs() * -0.33) ) #Caffe version has these ops as exp and scale layers
        edge_y_im = torch.exp( (self.edge_conv_y_3(image).abs() * -0.33) )
        edge_x_d = self.edge_conv_x_1(disparity)
        edge_y_d = self.edge_conv_y_1(disparity)

        return ((edge_x_im*edge_x_d)).abs().mean() + ((edge_y_im*edge_y_d)).abs().mean()

    def warp(self, image, disparity):

        #Create tensors of row and column indices
        r_ind = torch.arange(0,image.size(2)).view(-1,1).repeat(1,image.size(3)).to(self.device)
        c_ind = torch.arange(0,image.size(3)).repeat(image.size(2),1).to(self.device)

        #For bilinear interp. between two sets of pixel offsets (since disparities are floats that fall inbetween pixels)
        x0 = torch.floor(disparity).type(torch.LongTensor).to(self.device)
        x1 = x0+1

        #Empty list to store warped batch images
        warped_ims = []

        #How to do this without loops???
        for b in range(image.size(0)): #Loop over images in batch

            #Empty list to store warped channels for current image
            warped_ims_ch = []

            for ch in range(image.size(1)): #Loop over RGB channels

                #Column indicies for left side of bilinear interpolation 
                c_ind_d_0 = c_ind+x0[b,0]

                #Mask of invalid indices
                c_ind_d_0_invalid = (c_ind_d_0<0) | (c_ind_d_0>=image.size(3))

                #Make indices within bounds
                c_ind_d_0[c_ind_d_0>=image.size(3)] = image.size(3)-1
                c_ind_d_0[c_ind_d_0<0] = 0

                #Same as above but for right side of interpolation
                c_ind_d_1 = c_ind+x1[b,0]
                c_ind_d_1_invalid = (c_ind_d_1<0) | (c_ind_d_1>=image.size(3))
                c_ind_d_1[c_ind_d_1>=image.size(3)] = image.size(3)-1
                c_ind_d_1[c_ind_d_1<0] = 0

                #Inverse warp
                warped_ims_ch.append(((x1[b,0]-disparity[b,0])*image[b, ch, r_ind, c_ind_d_0] + (disparity[b,0]-x0[b,0])*image[b, ch, r_ind, c_ind_d_1]).unsqueeze(0).unsqueeze(0))
                
                #Set invalid areas to 0
                warped_ims_ch[-1][0, 0, c_ind_d_0_invalid] = 0.0
                warped_ims_ch[-1][0, 0, c_ind_d_1_invalid] = 0.0

            #List to tensor
            warped_ims.append(torch.cat(warped_ims_ch,1))

        return torch.cat(warped_ims,0)
    
    def eval(self, image_files, save_preds):
        data_loader = self.get_data_loader(image_files, depth=True, batch_size=self.eval_batch_size, shuffle=False)
        self.pytorch_model = torch.load(self.save_root + "/" + self.save_name, map_location=torch.device('cpu'))

        #Change metrics to True/False to compute error values (or not), save to True/False to save preidcted depth maps
        self.eval_pytorch_model(data_loader, metrics=True, save=save_preds)

    def eval_pytorch_model(self, loader, metrics=True, save=False):
        if self.pytorch_model:
            self.pytorch_model.eval()
            self.pytorch_model = self.pytorch_model.to(self.device)

            if metrics:
                self.reset_running_metrics()

            with torch.no_grad():
                for index, data in enumerate(loader):
                    images = data[0]
                    image_ids = data[2]
                    outputs = self.pytorch_model.forward(images.to(self.device))['h_flow'].cpu()

                    if metrics:
                        self.compute_running_metrics(outputs, data[3])

                    if save:
                        for i in range(outputs.size(0)):
                            self.save_image(outputs[i:i+1], image_ids[0][i], image_ids[1][i])

                if metrics:
                    self.print_running_metrics()
        else:
            raise ValueError("Are you sure the PyTorch model is loaded? Model (self.pytorch_model) not found.")

    def reset_running_metrics(self):
        self.squared_diff_running = 0.0
        self.squared_diff_log_running = 0.0
        self.abs_relative_diff_running = 0.0
        self.squared_relative_diff_running = 0.0
        self.num_pixels = 0
        self.RMS = 0.0
        self.RMS_log = 0.0
        self.abs_relative_diff = 0.0
        self.squared_relative_diff = 0.0

    def print_running_metrics(self):
        print("RMS       = {:.4f}".format(self.RMS))
        print("Log RMS   = {:.4f}".format(self.RMS_log))
        print("Abs. rel. = {:.4f}".format(self.abs_relative_diff))
        print("Sq. rel.  = {:.4f}".format(self.squared_relative_diff))

    def compute_running_metrics(self, pred, gt):
        disp_scale = float(gt.size(3))/float(pred.size(3))
        if pred.size()[2:] != gt.size()[2:]:
            pred = torch.nn.functional.interpolate(pred, (gt.size()[2:]), mode='bilinear', align_corners=False)

        valid_mask = gt>0

        bounds = [151, 367, 44, 1180]
        valid_mask[:,:,0:bounds[0],:] = 0
        valid_mask[:,:,bounds[1]:,:] = 0
        valid_mask[:,:,:,0:bounds[2]] = 0
        valid_mask[:,:,:,bounds[3]:] = 0

        #Convert predicted disparity to depth
        baseline = 0.5327254279298227
        focal_length = 721.5377
        pred_depth = (baseline * focal_length) / (disp_scale*pred)
        pred_depth[pred_depth>50] = 50
        pred_depth[pred_depth<1] = 1

        diff = (pred_depth-gt)*valid_mask
        diff_log = (pred_depth.log()-gt.log())*valid_mask

        self.squared_diff_running += (diff**2).sum().item()
        self.squared_diff_log_running += (diff_log[valid_mask]**2).sum().item()

        self.abs_relative_diff_running += (diff.abs()[valid_mask]/gt[valid_mask]).sum().item()
        self.squared_relative_diff_running += ((diff[valid_mask]**2)/gt[valid_mask]).sum().item()

        self.num_pixels += valid_mask.sum()

        self.RMS = np.sqrt(self.squared_diff_running / float(self.num_pixels))
        self.RMS_log = np.sqrt(self.squared_diff_log_running / float(self.num_pixels))

        self.abs_relative_diff = self.abs_relative_diff_running / float(self.num_pixels)
        self.squared_relative_diff = self.squared_relative_diff_running / float(self.num_pixels)


    def save_image(self, image, scene_id, image_id):
        image = image.cpu().numpy()
        image_transposed = numpy.transpose(image, (2, 3, 0, 1))
        image_transposed = numpy.squeeze(image_transposed)

        save_dir = "{}/{}".format(self.save_root, scene_id)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.imsave("{}/{}.png".format(save_dir, image_id), image_transposed, cmap='plasma')

    def get_data_loader(self, image_files, depth=False, batch_size=1, shuffle=True):
        dataset = kitti_loader.KITTILoader(self.kitti_root, image_files, depth=depth)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="'train' or 'eval' mode", type=str)
    parser.add_argument("--image_files", help="Text file with list of images", type=str)
    parser.add_argument("--kitti_root", help="Root directory of KITTI dataset", type=str)
    parser.add_argument("--experiment_root", help="Directory to save/load models and predictions", type=str)
    parser.add_argument("--save_name", help="Name of model file to save/load in experiment_root", type=str, default="model.pt")
    parser.add_argument("--gpu", help="GPU index", type=int, dest='gpu_id', default=0)
    parser.add_argument("--cpu", help="CPU mode", action='store_true')
    parser.add_argument("--save_preds", help="Save predicted depth maps (for eval mode)", action='store_true')

    args = parser.parse_args()

    if args.cpu:
        gpu_id = None
    else:
        gpu_id = args.gpu_id

    model = TrainEvalPytorchModel(args.kitti_root, args.experiment_root, args.save_name, gpu_id)

    if args.mode == "train":
        model.train(args.image_files)
    elif args.mode == "eval":
        model.eval(args.image_files, args.save_preds)
    else:
        raise Exception("Invalid mode, must be 'train' or 'val'")
