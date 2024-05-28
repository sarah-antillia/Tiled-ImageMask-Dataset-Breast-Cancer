# Copyright 2024 antillia.com Toshiyuki Arai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# 2024/05/29 sarah@antillia.com

# PreAugmentedImageMaskDatasetGenerator.py

import os
import sys
import shutil
import cv2
from PIL import Image
import glob
import numpy as np
import math
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

from skimage.io import imread

import traceback

  # deform method of this class has been taken from the following code.
  # https://github.com/MareArts/Elastic_Effect/blob/master/Elastic.py
  #
  # https://cognitivemedium.com/assets/rmnist/Simard.pdf
  #
  # See also
  # https://www.kaggle.com/code/jiqiujia/elastic-transform-for-data-augmentation/notebook
  # 
class PreAugmentedImageMaskDatasetGenerator:

  def __init__(self, augmentation=True):
    self.seed = 137
    self.W = 512
    self.H = 512

    self.augmentation = augmentation
    if self.augmentation:
      self.hflip    = True
      self.vflip    = True
      self.rotation = True
      self.ANGLES   = [90, 180, 270,]


      # defomration parameters
      self.deformation = True
      self.alpha    = 1300
      self.sigmoid  = 8
      # distortion parameters

      # distoration parameters
      self.distortion =True
      self.gaussina_filer_rsigma = 40
      self.gaussina_filer_sigma  = 0.5

      self.distortions           = [ 0.02, 0.03]
      self.rsigma = "sigma"  + str(self.gaussina_filer_rsigma)
      self.sigma  = "rsigma" + str(self.gaussina_filer_sigma)

      #shrink images/masks
      self.resize = True
      self.SHRINKS = [0.8]

    
  def create_mask_files(self, mask_file, output_dir, index):
    print("--- create_mask_files {}".format(mask_file))
    mask = cv2.imread(mask_file)
    mask = self.resize_to_512x512(mask, is_mask=True)
    basename = str(index) + ".jpg"
    
    filepath = os.path.join(output_dir, basename)
    cv2.imwrite(filepath, mask)
    print("--- Save {}".format(filepath))
    if self.augmentation:
      self.augment(mask, basename, output_dir, border=(0, 0, 0), mask=True)
    return 1
  
  def create_image_files(self, image_file, output_dir, index):
    print("--- create_image_files {}".format(image_file))
    image = cv2.imread(image_file)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if image.ndim == 2:
       image  = np.expand_dims(image, axis=-1)
    image = self.resize_to_512x512(image, is_mask=False)
    basename = str(index) + ".jpg"

    filepath = os.path.join(output_images_dir, basename)
    cv2.imwrite(filepath, image)
    print("--- Save {}".format(filepath))
 
    if self.augmentation:
      self.augment(image, basename, output_dir, border=(0, 0, 0), mask=False)
  

  def generate(self, train_images_dir, train_masks_dir, 
                        output_images_dir, output_masks_dir):

    image_files = glob.glob(train_images_dir + "/*.jpg")
    mask_files  = glob.glob(train_masks_dir  + "/*.jpg")
    image_files = sorted(image_files)
    mask_files  = sorted(mask_files)
    num_image_files = len(image_files)
    num_mask_files  = len(mask_files)
    print("--- num_image_files {}".format(num_image_files))
    print("--- num_mask_files  {}".format(num_mask_files))
    if num_image_files != num_mask_files:
       raise Exception("The number of images and mask files unmatched.")
    index = 1001
    for i, _ in enumerate(mask_files):
      mask_file  = mask_files [i]
      image_file = image_files[i]
      self.create_mask_files(mask_file,   output_masks_dir,  index+i)
      self.create_image_files(image_file, output_images_dir, index+i)

  def resize_to_512x512(self, image, is_mask=False):

    color = image[2][2]  
    h, w = image.shape[:2]
    RESIZE = h
    if w > h:
      RESIZE = w
    # 1. Create a black background
    if is_mask:
      background = np.zeros((RESIZE, RESIZE, 3),  np.uint8)
    else:
      background = np.ones((RESIZE, RESIZE, 3),  np.uint8)
      background = background * color
    x = int((RESIZE - w)/2)
    y = int((RESIZE - h)/2)
    # 2. Paste the image to the background 
    background[y:y+h, x:x+w] = image
    # 3. Resize the background to (512x512)
    resized = cv2.resize(background, (self.W, self.H))

    return resized

  def augment(self, image, basename, output_dir, border=(0, 0, 0), mask=False):
    if self.hflip:
      flipped = self.horizontal_flip(image)
      output_filepath = os.path.join(output_dir, "hflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

    if self.vflip:
      flipped = self.vertical_flip(image)
      output_filepath = os.path.join(output_dir, "vflipped_" + basename)
      cv2.imwrite(output_filepath, flipped)
      print("--- Saved {}".format(output_filepath))

    if self.rotation:
      self.rotate(image, basename, output_dir, border)

    if self.deformation:
      self.deform(image, basename, output_dir)

    if self.distortion:
      self.distort(image, basename, output_dir)

    if self.resize:
      self.shrink(image, basename, output_dir)

  def horizontal_flip(self, image): 
    print("shape image {}".format(image.shape))
    if len(image.shape)==3:
      return  image[:, ::-1, :]
    else:
      return  image[:, ::-1, ]

  def vertical_flip(self, image):
    if len(image.shape) == 3:
      return image[::-1, :, :]
    else:
      return image[::-1, :, ]

  def rotate(self, image, basename, output_dir, border):
    for angle in self.ANGLES:      
      center = (self.W/2, self.H/2)
      rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

      rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(self.W, self.H), borderValue=border)
      output_filepath = os.path.join(output_dir, "rotated_" + str(angle) + "_" + basename)
      cv2.imwrite(output_filepath, rotated_image)
      print("--- Saved {}".format(output_filepath))
      
  # This method has been taken from the following code.
  # https://github.com/MareArts/Elastic_Effect/blob/master/Elastic.py
  #
  # https://cognitivemedium.com/assets/rmnist/Simard.pdf
  #
  # See also
  # https://www.kaggle.com/code/jiqiujia/elastic-transform-for-data-augmentation/notebook
  # 
  def deform(self, image, basename, output_dir):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    random_state = np.random.RandomState(self.seed)

    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigmoid, mode="constant", cval=0) * self.alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), self.sigmoid, mode="constant", cval=0) * self.alpha
    #dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    deformed_image = map_coordinates(image, indices, order=1, mode='nearest')  
    deformed_image = deformed_image.reshape(image.shape)

    image_filename = "deformed" + "_alpha_" + str(self.alpha) + "_sigmoid_" +str(self.sigmoid) + "_" + basename
    print("--- filename {}".format(image_filename))
    image_filepath  = os.path.join(output_dir, image_filename)
    cv2.imwrite(image_filepath, deformed_image)
    
 
  def distort(self, image, basename, output_dir):
    shape = (image.shape[1], image.shape[0])
    (w, h) = shape
    xsize = w
    if h>w:
      xsize = h
    # Resize original img to a square image
    resized = cv2.resize(image, (xsize, xsize))
 
    shape   = (xsize, xsize)
 
    t = np.random.normal(size = shape)
    for size in self.distortions:
      filename = "distorted_" + str(size) + "_" + self.sigma + "_" + self.rsigma + "_" + basename
      output_file = os.path.join(output_dir, filename)    
      dx = gaussian_filter(t, self.gaussina_filer_rsigma, order =(0,1))
      dy = gaussian_filter(t, self.gaussina_filer_rsigma, order =(1,0))
      sizex = int(xsize*size)
      sizey = int(xsize*size)
      dx *= sizex/dx.max()
      dy *= sizey/dy.max()

      image = gaussian_filter(image, self.gaussina_filer_sigma)

      yy, xx = np.indices(shape)
      xmap = (xx-dx).astype(np.float32)
      ymap = (yy-dy).astype(np.float32)

      distorted = cv2.remap(resized, xmap, ymap, cv2.INTER_LINEAR)
      distorted = cv2.resize(distorted, (w, h))
      cv2.imwrite(output_file, distorted)
      print("=== Saved distorted image file{}".format(output_file))

  def shrink(self, image, basename, output_dir):

    h, w = image.shape[:2]
  
    for shrink in self.SHRINKS:
      rw = int (w * shrink)
      rh = int (h * shrink)
      resized_image = cv2.resize(image, dsize= (rw, rh),  interpolation=cv2.INTER_NEAREST)
      
      squared_image = self.paste(resized_image, mask=False)
    
      ratio   = str(shrink).replace(".", "_")
      image_filename = "shrinked_" + ratio + "_" + basename
      image_filepath  = os.path.join(output_dir, image_filename)
      cv2.imwrite(image_filepath, squared_image)
    

  def paste(self, image, mask=False):
    l = len(image.shape)
   
    h, w,  = image.shape[:2]

    if l==3:
      background = np.zeros((self.H, self.W, 3), dtype=np.uint8)
      (b, g, r) = image[h-10][w-10] 
      #print("r:{} g:{} b:c{}".format(b,g,r))
      background += [b, g, r][::-1]
    else:
      v =  image[h-10][w-10] 
      image  = np.expand_dims(image, axis=-1) 
      background = np.zeros((self.H, self.W, 1), dtype=np.uint8)
      background[background !=v] = v
    x = (self.W - w)//2
    y = (self.H - h)//2
    background[y:y+h, x:x+w] = image
    return background
  


if __name__ == "__main__":
  try:
    input_images_dir  = "./BCSS-Mostly-Tumor-master/images"
    input_masks_dir   = "./BCSS-Mostly-Tumor-master/masks"
    output_images_dir = "./PreAugmented-Breast-Cancer-master/images/"
    output_masks_dir  = "./PreAugmented-Breast-Cancer-master/masks/"

    if os.path.exists(output_images_dir):
      shutil.rmtree(output_images_dir)
    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)

    if os.path.exists(output_masks_dir):
      shutil.rmtree(output_masks_dir)
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)

    # Create jpg image and mask files from tif files.
    generator = PreAugmentedImageMaskDatasetGenerator(augmentation=True)
    generator.generate(input_images_dir, input_masks_dir, 
                        output_images_dir, output_masks_dir)
  except:
    traceback.print_exc()


