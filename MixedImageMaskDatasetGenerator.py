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

# 2024/05/26 to-arai
# 2024/05/29 Modified to created ruduced images and mask of 512x512 and
# augment them.
# 2024/05/29 sarah@antillia.com
#  Modfied self.exclude_empty=False to True to avoid an empty mask to be saved.

# MixedImageMaskDatasetGenerator.py

"""
From 
./BCSS-Mostly-Tumor-master
 ├─images
 └─masks

this script creates
./Mixed-BCSS-Mostly-Tumor-master
 ├─images
 └─masks


1 Create tiledly split patches (images and masks) of 512x512 from the large image and mask files
  in .//BCSS-Mostly-Tumor-master, and save them to ./Mixed-BCSS-Mostly-Tumor-master

2 Create non-tiled reduced images and masks of 512x512 from the large image and mask files
  in .//BCSS-Mostly-Tumor-master, and save them to ./Mixed-BCSS-Mostly-Tumor-master


"""

import os
import shutil
import glob
import cv2
import numpy as np

from PIL import Image, ImageFilter
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates

import traceback

class MixedImageMaskDatasetGenerator:

  def __init__(self, split_size=512, exclude_empty=True, include_resized=True, augmentation=True):
    self.seed         = 137

    self.split_size   = split_size
    self.resize       = split_size
    self.cut_in_half  = False
    # 2024/05/29 Modfied self.exclude_empty=False to True
    self.exclude_empty = exclude_empty
    self.W             = split_size
    self.H             = split_size
    # 2024/05/29 Include resized images and mask.
    self.include_resized = include_resized
    self.num_skipped   = 0
    # Blur flag
    self.blur         = True
    # GausssinaBlur parameters
    # Blur parameter for the resized 
    self.blur_ksize1  = 3
    # Blur parameter for the splitted
    self.blur_ksize2  = 3

    #Augmentation parameters for the resized.
    self.augmentation = augmentation
    # 2024/05/29
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
      #self.resize = True
      self.SHRINKS = [0.8]

 
  def generate(self, root_dir, output_dir):
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    input_images_dir = os.path.join(root_dir, "images")
    input_masks_dir  = os.path.join(root_dir, "masks")

    output_images_dir = os.path.join(output_dir, "images")
    output_masks_dir  = os.path.join(output_dir, "masks")
    if not os.path.exists(output_images_dir):
      os.makedirs(output_images_dir)
    if not os.path.exists(output_masks_dir):
      os.makedirs(output_masks_dir)

    if self.include_resized:
      self.resize_one(input_masks_dir,  output_masks_dir, output_images_dir, mask=True)
      self.resize_one(input_images_dir, output_masks_dir, output_images_dir, mask=False)

    # 2024/05/29 Exchanged the calling split_one for image and mask
    # to mask and image, to check an empty mask.
    # 1. Split masks 1st
    self.split_one(input_masks_dir,  output_masks_dir, output_images_dir, mask=True)

    # 2. Split images 2nd
    self.split_one(input_images_dir, output_masks_dir, output_images_dir, mask=False)


  def resize_one(self, input_images_dir, output_masks_dir, output_images_dir,  mask=False):
    image_files  = glob.glob(input_images_dir + "/*.jpg")
    image_files += glob.glob(input_images_dir + "/*.png")
    image_files  = sorted(image_files)
    print("--- image_files {}".format(image_files))
    index = 1000

    for image_file in image_files:
      index += 1
      image   = cv2.imread(image_file)
      #resized = cv2.resize(image, (self.resize, self.resize))
      resized = self.resize_to_square(image, mask=mask)

      filename = "r_" + str(index) + ".jpg"
      output_mask_filepath  = os.path.join(output_masks_dir,  filename) 
      output_image_filepath = os.path.join(output_images_dir, filename) 
      if mask:
        if self.blur:
          resized = cv2.GaussianBlur(resized, ksize=(self.blur_ksize1, self.blur_ksize1), sigmaX=0)
        cv2.imwrite(output_mask_filepath, resized)
        print("--- Saved {}".format(output_mask_filepath))
        if self.augmentation:
          self.augment(resized, filename, output_masks_dir, mask=True )

      else:
        if os.path.exists(output_mask_filepath):
          cv2.imwrite(output_image_filepath, resized)
          print("--- Saved {}".format(output_image_filepath))
          if self.augmentation:
            self.augment(resized, filename, output_images_dir, mask=False)
        else:
          pass
  
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
      
  def resize_to_square(self, image, mask=False):     
    h, w = image.shape[:2]
    RESIZE = h
    if w > h:
      RESIZE = w
    # 1. Create a black background
    background = np.zeros((RESIZE, RESIZE, 3),  np.uint8) 
    if mask ==False:
      background = np.ones((RESIZE, RESIZE, 3),  np.uint8) 
      color = image[20][20] 
      #print("r:{} g:{} b:c{}".format(b,g,r))
      background = background * color 
      
    x = int((RESIZE - w)/2)
    y = int((RESIZE - h)/2)
    # 2. Paste the image to the background 
    background[y:y+h, x:x+w] = image
    # 3. Resize the background to (self.resize, self.resize)
    resized = cv2.resize(background, (self.resize, self.resize))

    return resized

  def split_one(self, input_images_dir, output_masks_dir, output_images_dir,  mask=False):
    image_files  = glob.glob(input_images_dir + "/*.jpg")
    image_files += glob.glob(input_images_dir + "/*.png")
    image_files  = sorted(image_files)

    # Take half of the image_files to reduce the number of splitted files. 
    if self.cut_in_half:
      hlen = int(len(image_files)/2)
      image_files = image_files[:hlen]
    #print("--- split_one image_files {}".format(image_files))
    index = 1000
    split_size = self.split_size

    for image_file in image_files:
      index += 1
      print("---- split_one {}".format(image_file))
      image = Image.open(image_file)
      
      w, h  = image.size

      vert_split_num  = h // split_size
      if h % split_size != 0:
        vert_split_num += 1

      horiz_split_num = w // split_size

      for j in range(vert_split_num):
        for i in range(horiz_split_num):
          left  = split_size * i
          upper = split_size * j
          right = left  + split_size
          lower = upper + split_size
  
          cropbox = (left,  upper, right, lower )
          
          # Crop a region specified by the cropbox from the whole image to create a tiled image segmentation.      
          cropped = image.crop(cropbox)

          #line = "image file {}x{} : x:{} y:{} width: {} height:{}\n".format(j, i, left, upper, cw, ch)
          #print(line)            
          cropped_image_filename = str(index) + "_" + str(j) + "x" + str(i) + ".jpg"
          output_mask_filepath  = os.path.join(output_masks_dir,  cropped_image_filename) 
          output_image_filepath = os.path.join(output_images_dir, cropped_image_filename) 

          if mask:
            # 2024/05/29 Modified to exclude an empty black mask
            if self.exclude_empty:
              if self.is_not_empty(cropped):
                if self.blur:
                  cropped = cropped.filter(ImageFilter.GaussianBlur(radius = self.blur_ksize2)) 
                cropped.save(output_mask_filepath)
                print("--- Saved {}".format(output_mask_filepath))
              else:
                print("--- Don't save an empty mask")
                self.num_skipped += 1
                continue
            
            # self.exclude_empty == False
            else:
                if self.blur:
                  cropped = cropped.filter(ImageFilter.GaussianBlur(radius = self.blur_ksize2)) 
                cropped.save(output_mask_filepath)
                print("--- Saved {}".format(output_mask_filepath))
                  
          else:
            if os.path.exists(output_mask_filepath):
              cropped.save(output_image_filepath)
              print("--- Saved {}".format(output_image_filepath))
            else:
              print("--- Don't save an image, because the corresponding mask is empty.")


  def is_not_empty(self, img):
    rc = False
    img = self.pil2cv(img)
    if img.any() > 0:    
       rc = True
    return rc
  
  def pil2cv(self, image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2: 
        pass
    elif new_image.shape[2] == 3: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4: 
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


"""
From
./BCSS-Mostly-Tumor-master
 ├─images
 └─masks

splitting each image and mask of the dataset above to 512x512 tiled,
and save those tiledly splitted images and masks under
./Mixed-BCSS-Mostly-Tumor-master-M2
 ├─images
 └─masks

"""
  

if __name__ == "__main__":
  try:
    input_dir  = "./BCSS-Mostly-Tumor-master"
    output_dir = "./Mixed-BCSS-Mostly-Tumor-master-M2"
    generator = MixedImageMaskDatasetGenerator()

    generator.generate(input_dir, output_dir)
    
    num_skipped = generator.num_skipped
    print("--- num_skipped {}".format(num_skipped))
  except:
    traceback.print_exc()