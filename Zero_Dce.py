import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from tqdm import tqdm
import random
from random import choice
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from PIL import Image, ImageOps
from mpl_toolkits.axes_grid1 import ImageGrid
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import sys
import torch.utils.data as data
import numpy as np
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import time
from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

def populate_train_list(lowlight_images_path):    
    # Danh sách tất cả các file trong thư mục
    all_files = os.listdir(lowlight_images_path)

    # Sử dụng glob để lấy tất cả file hình ảnh có đuôi .png, .jpg, hoặc .jpeg
    image_list_lowlight = glob.glob(os.path.join(lowlight_images_path, "*.png")) + \
                          glob.glob(os.path.join(lowlight_images_path, "*.jpg")) + \
                          glob.glob(os.path.join(lowlight_images_path, "*.jpeg"))

    
    valid_image_list = [img for img in image_list_lowlight if os.path.exists(img)]
    random.shuffle(valid_image_list)
    
    return valid_image_list

class lowlight_loader(data.Dataset):
    def __init__(self, lowlight_images_path):
        self.train_list = populate_train_list(lowlight_images_path) 
        self.size = 512  # Kích thước ảnh sẽ resize
        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]
        
        # Kiểm tra sự tồn tại của file trước khi mở
        if not os.path.exists(data_lowlight_path):
            print(f"Warning: File {data_lowlight_path} not found. Skipping this file.")
            return None  # Hoặc trả về một giá trị mặc định, chẳng hạn như một tensor rỗng

        try:
            # Đọc ảnh bằng OpenCV
            data_lowlight = cv2.imread(data_lowlight_path)
            if data_lowlight is None:
                raise ValueError(f"Error opening file {data_lowlight_path}: Unable to read image.")
        except Exception as e:
            print(f"Error opening file {data_lowlight_path}: {e}")
            return None  # Hoặc bỏ qua file có lỗi

        # Resize ảnh bằng OpenCV
        data_lowlight = cv2.resize(data_lowlight, (self.size, self.size), interpolation=cv2.INTER_LANCZOS4)
        
        # Chuẩn hóa dữ liệu (0-1)
        data_lowlight = data_lowlight.astype(np.float32) / 255.0
        
        # Chuyển ảnh sang tensor và chuyển đổi thứ tự kênh (HWC -> CHW)
        data_lowlight = torch.from_numpy(data_lowlight).float()
        
        if data_lowlight is None:
            print(f"Warning: Empty image at index {index}")
        
        # Chuyển từ [H, W, C] sang [C, H, W]
        return data_lowlight.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)
class CSDN_Tem(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CSDN_Tem, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out

class enhance_net_nopool(nn.Module):
    def __init__(self,scale_factor):
        super(enhance_net_nopool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=self.scale_factor)
        number_f = 32

#   zerodce DWC + p-shared
        self.e_conv1 = CSDN_Tem(3,number_f) 
        self.e_conv2 = CSDN_Tem(number_f,number_f) 
        self.e_conv3 = CSDN_Tem(number_f,number_f) 
        self.e_conv4 = CSDN_Tem(number_f,number_f) 
        self.e_conv5 = CSDN_Tem(number_f*2,number_f) 
        self.e_conv6 = CSDN_Tem(number_f*2,number_f) 
        self.e_conv7 = CSDN_Tem(number_f*2,3) 

    def enhance(self, x,x_r):

        x = x + x_r*(torch.pow(x,2)-x)
        x = x + x_r*(torch.pow(x,2)-x)
        x = x + x_r*(torch.pow(x,2)-x)
        enhance_image_1 = x + x_r*(torch.pow(x,2)-x)		
        x = enhance_image_1 + x_r*(torch.pow(enhance_image_1,2)-enhance_image_1)		
        x = x + x_r*(torch.pow(x,2)-x)	
        x = x + x_r*(torch.pow(x,2)-x)
        enhance_image = x + x_r*(torch.pow(x,2)-x)	

        return enhance_image
    def forward(self, x):
        if self.scale_factor==1:
            x_down = x
        else:
            x_down = F.interpolate(x,scale_factor=1/self.scale_factor, mode='bilinear')

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
        x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
        if self.scale_factor==1:
            x_r = x_r
        else:
            x_r = self.upsample(x_r)
        enhance_image = self.enhance(x,x_r)
        return enhance_image,x_r

from tqdm import tqdm
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Function to load a model checkpoint
def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path,map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.eval()  # Set to evaluation mode
    return model

# Define the inference function
def resize_image_and_labels(image, label_path, target_size=(640, 640)):
    """
    Resize image and adjust YOLO labels accordingly.
    Args:
        image (PIL.Image): Original image.
        label_path (str): Path to the label file.
        target_size (tuple): Target size (width, height).
    Returns:
        PIL.Image: Resized image.
        list: Updated labels.
    """
    original_width, original_height = image.size
    target_size=(640, 640)
    resized_image = image.resize(target_size, Image.Resampling.LANCZOS)  # Updated to use LANCZOS
    
    updated_labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as label_file:
            lines = label_file.readlines()
            for line in lines:
                if line.strip().startswith('bbGt'):  # Skip lines starting with 'bbGt'
                    continue
                
                parts = line.strip().split()
                if len(parts) < 5:  # Ensure the line has enough parts for YOLO format
                    continue
                
                try:
                    class_id = parts[0]
                    x_center = float(parts[1]) * original_width / target_size[0]
                    y_center = float(parts[2]) * original_height / target_size[1]
                    width = float(parts[3]) * original_width / target_size[0]
                    height = float(parts[4]) * original_height / target_size[1]
                    updated_labels.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
                except ValueError:
                    print(f"Skipping invalid label line: {line.strip()}")
                    continue
    return resized_image, updated_labels

# Define the inference function with additional formatting for saving
def infer(original_image, model, label_path, target_size=(640, 640)):
    """
    Enhance the image using the given model with resizing and adjust labels.
    Args:
        original_image (PIL.Image): The original image.
        model (torch.nn.Module): The enhancement model.
        label_path (str): Path to the label file.
        target_size (tuple): Target size for resizing (width, height).
    Returns:
        np.ndarray: Enhanced image in numpy format.
        list: Updated labels.
    """
    # Resize the image and labels
    target_size=(640, 640)
    resized_image= original_image.resize(target_size, Image.Resampling.LANCZOS)  # Updated to use LANCZOS

    # Preprocess the image: convert to tensor and normalize
    preprocess = transforms.ToTensor()
    img_tensor = preprocess(resized_image).unsqueeze(0) # Add batch dimension and move to GPU

    # Run the model
    with torch.no_grad():
        enhanced_image, _ = model(img_tensor)  # Only use the enhanced image output
        enhanced_image = enhanced_image.squeeze(0)  # Remove batch dimension

    # Convert the enhanced image back to numpy format
    enhanced_image_np = enhanced_image.permute(1, 2, 0).cpu().numpy()  # Convert from CHW to HWC
    enhanced_image_np = np.clip(enhanced_image_np * 255, 0, 255).astype(np.uint8)  # Ensure uint8 format for saving
    return enhanced_image_np

# Function to plot results
def plot_results(images, titles, figure_size=(12, 12)):
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        plt.imshow(images[i])
        plt.axis("off")
    plt.show()

