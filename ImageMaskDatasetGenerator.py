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
#
# 2024/05/26 sarah@antillia.com

import os
import numpy as np
import cv2
import json
import glob
import shutil
import traceback

# Generate a mask dataset specified by label name and fill_color
# For example, label = "mostlly_tumor"

class ImageMaskDatasetGenerator:

  def __init__(self,  label="mostly_tumor", 
                fill_color=(255, 255, 255)):
    self.label      = label
    self.fill_color = fill_color
  
    #Fixed base_size, which will be used a base size to create a Tiled-imageMask-Dataset
    self.base_size = 512

  def generate(self, images_dir, jsons_dir, output_images_dir, output_masks_dir):
    print("=== generate images_dir {} json_dir {}".format(images_dir, jsons_dir))
    image_files = glob.glob(images_dir + "/*.png")
    image_files = sorted(image_files)
    
    json_files  = glob.glob(jsons_dir + "/*.json")
    json_files  = sorted(json_files)

    num_image_files = len(image_files)
    num_json_files  = len(json_files)
    if num_image_files != num_json_files:
      error = "Unmatched number of images and json files"
      raise Exception(error)
        
    index = 1000
    for image_file in image_files:
      index += 1
      print("--- image_file {}".format(image_file))
      self.generate_image_file(image_file, output_images_dir, index)
      basename = os.path.basename(image_file)
      prefix   =  basename.split("-DX1")[0]
     
      found_json_files = glob.glob(jsons_dir + "/" + prefix + "*.json")
      if len(found_json_files) == 0:
        error = "Not found corresponding json file to " + image_file
        raise Exception(error)
      
      json_file = found_json_files[0]
      print("--- Found corresponding json file {}".format(json_file))
      self.generate_mask_file(json_file,   output_masks_dir,  output_images_dir, index)


  def generate_image_file(self, image_file, output_dir, index):
    print("--- generate_image_file: {}".format(image_file))
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, c = image.shape
    # resize image to be a minimum integral multiple of self.resize
    rh = (h//self.base_size + 1) * self.base_size
    rw = (w//self.base_size + 1) * self.base_size
    image = cv2.resize(image, (rw, rh))
    image_filepath = os.path.join(output_dir, str(index) + ".jpg")
    cv2.imwrite(image_filepath, image)
    print("=== saved {}".format(image_filepath))       

  def generate_mask_file(self, json_file, output_dir, output_images_dir, index):
    print("--- generate_mask_file from json file: {}".format(json_file))
    with open(json_file, "r") as f:
      jdata  = json.load(f)
      header = jdata[0]
      print(header)
      annotation = header["annotation"]
      element    = annotation["elements"][0]
      center = element["center"]
      width  = element["width"]
      height = element["height"]
      print("--- center {}".format(center))
      print("--- width  {}".format(width))
      print("--- height {}".format(height))
      
      # create an empty black mask of width and height 
      mask = np.zeros((height, width, 3),dtype=np.uint8)
      #print("--- mask shape {}".format(mask.shape))
      
      jdata = jdata[1:]
      for data in jdata:
        annotation = data["annotation"]
        elements   = annotation["elements"]
        for element in elements:
          group = element.get("group")
          if group == None:
            continue
          #print("--- group {}".format(group))
          if group == self.label:
             print("---group {}".format(group))
             points = element["points"]
             points = self.get2dpoints(points, center, width, height)
             points = np.array(points, np.int32)

             # fillPoly image by color = (255, 255, 255) white
             cv2.fillPoly(mask, [points], self.fill_color)
        
      mask_filepath = os.path.join(output_dir, str(index) + ".jpg")
      h, w, c = mask.shape

      # resize mask to be a minimum integral multiple of self.base_size
      rh = (h//self.base_size + 1) * self.base_size
      rw = (w//self.base_size + 1) * self.base_size
      mask = cv2.resize(mask, (rw, rh))
      image_filepath = os.path.join(output_images_dir,  str(index) + ".jpg")
      image = cv2.imread(image_filepath)
      h, w, _ = image.shape
      if rw != w or rh != h:
        print("Warning: Unmatched pixel-size of image and mask rw:{} rh:{} w:{} h:{}".format(rw, rh, w, h))
        mask = cv2.resize(mask, (w, h))
      cv2.imwrite(mask_filepath, mask)
      print("=== saved {}".format(mask_filepath))

  def get2dpoints(self, points, center, width, height):
    point_2d = []
    [rx, ry,_] = center
    hw = int(width/2)
    hh = int(height/2)
    for point in points:
       [x, y, z] = point
       x = x - rx + hw
       y = y - ry + hh
       point_2d += [[x, y]]
    return point_2d
  

if __name__ == "__main__":
  try:
    images_dir = "./images"
    jsons_dir  = "./annotations"

    output_dir = "./BCSS-Mostly-Tumor-master"
    output_images_dir = output_dir + "/images"
    output_masks_dir = output_dir  + "/masks"
   
    if os.path.exists(output_dir):
      shutil.rmtree(output_dir)
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)

    os.makedirs(output_images_dir)
    os.makedirs(output_masks_dir)

    label      = "mostly_tumor"
    # mostly_tumor fillColor = (255, 0, 0)
    fill_color = (255, 255, 255)
  
    generator  = ImageMaskDatasetGenerator(label=label, fill_color=fill_color)
    generator.generate(images_dir, jsons_dir, output_images_dir, output_masks_dir)

  except:
     traceback.print_exc()
