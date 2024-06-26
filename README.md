<h2>Tiled-ImageMask-Dataset-Breast-Cancer (Updated: 2024/05/31) </h2>
</h2>
<li>
2024/05/28 Added Python script <a href="PreAugmentedImageMaskDatasetGenerator.py">PreAugmentedImageMaskDatasetGenerator.py</a> 
to create Pre-Augmented-Non-Tiled-Breast-Cancer-ImageMask-Dataset.<br>
</li>
<li>
2024/05/28 Created Pre-Augmented-Non-Tiled-Breast-Cancer-ImageMask-Dataset 
<a href="https://drive.google.com/file/d/1B3QfYxi52UqyVxcfxnRoGYw79KLIn-XA/view?usp=sharing">Non-Tiled-Breast-Cancer-ImageMask-Dataset-V1.zip.</a>
</li>
<li>
2024/05/29 Updated Python script <a href="MixedImageMaskDatasetGenerator.py">MixedImageMaskDatasetGenerator.py</a> 
to create Mixed-Breast-Cancer-ImageMask-Dataset exluding the empty tiled black masks and the corresponding tiled images.
</li>
<li>
2024/05/29 Created Mixed-Breast-Cancer-ImageMask-Dataset-M2 
<a href="https://drive.google.com/file/d/1tkGpCrHGIzzFKjrPhBQ4j1BOKbYGbGO0/view?usp=sharing">Mixed-Breast-Cancer-ImageMask-Dataset-M2.zip</a>
</li>

<br>
This is a Tiled and Non-Tiled Breast Cancer ImageMask Dataset for Image-Segmentation.<br>
<br>
<b>Tiled-ImageMask-Dataset</b> is a microscopic annotation dataset which is created by splitting the large images and masks
of BCSS to 512x512 tiles, and keeping the detailed features on the cancer regions.<br>
<br>
<b>Non-Tiled-ImageMask-Dataset</b> is a macroscopic annotation dataset which is created by reducing the size of 
the large images and masks of BCSS 
to 512x512, and losing a lot of detailed pixel level information of the cancer regions.<br><br>
Probably, you will have to use both Tiled and Non-Tiled Dataset mixing to train a segmentation model.
You may use the latest experimental <b>Mixed-Breast-Cancer-ImageMask-Dataset-M2</b>
 <a href="https://drive.google.com/file/d/1tkGpCrHGIzzFKjrPhBQ4j1BOKbYGbGO0/view?usp=sharing">Mixed-Breast-Cancer-ImageMask-Dataset-M2.zip</a>
<br>

<br>
We created a tiled image and mask dataset from the original large size image and mask files in 

<a href="https://github.com/PathologyDataScience/BCSS">
Breast Cancer Semantic Segmentation (BCSS) dataset
</a>

<br><br>
The pixel-size of the original images and masks in 
image and mask BCSS dataset is from 2K to 7K, 
and too large to use for a training of an ordinary segmentation model.
Therefore we created a dataset of images and masks which were split to the small tiles of 512x512 pixels, 
which can be used for a segmentation model.  
<br>

For example, an image and mask of 4090x4090 pixel-size in BCSS can be split to a lot of tiles of 512x512 as shown below.<br>
<hr>
<b>Image and Mask</b>
<table>
<!--
<tr>
<th>
Image
</th>
<th>Mask</th>
</tr>
-->
<tr>
<td>
<img src="./asset/image-1013.jpg" width="512" height="auto">
</td>
<td>
<img src="./asset/mask-1013.jpg" width="512" height="auto">
</td>
</tr>
</table>
<br>
<b>Tiledly Split Images and Masks</b>
<table>
<!--
<tr>
<th>
Tiled-Image
</th>
<th>Tiled-Mask</th>
</tr>
-->
<tr>
<td>
<img src="./asset/tiled_images_sample.png" width="512" height="auto">
</td>
<td>
<img src="./asset/tiled_masks_sample.png" width="512" height="auto">
</td>

</tr>
</table>
<hr>


<br>
<b>1. Download Tiled-Image-Mask-Dataset</b><br>
You can download our dataset created here from the google drive 
<a href="https://drive.google.com/file/d/1IedbpmttIgY17pPUbS0uliugD7rlkFJQ/view?usp=sharing">Tiled-Breast-Cancer-ImageMask-Dataset-X.zip</a>
<br>
<br>
<b>2. Download Pre-Augmented-Non-Tiled-Image-Mask-Dataset</b><br>
You can download our dataset created here from the google drive 
<a href="https://drive.google.com/file/d/1B3QfYxi52UqyVxcfxnRoGYw79KLIn-XA/view?usp=sharing">Non-Tiled-Breast-Cancer-ImageMask-Dataset-V1.zip</a>

<br>
<br>
<b>3. Download Mixed-Image-Mask-Dataset-M2</b><br>
You can download our dataset created here from the google drive 
<a href="https://drive.google.com/file/d/1tkGpCrHGIzzFKjrPhBQ4j1BOKbYGbGO0/view?usp=sharing">Mixed-Breast-Cancer-ImageMask-Dataset-M2.zip</a>

<br>
<br>


<h3>1. Dataset Citation</h3>
The original dataset used here has been taken from the following github repository.<br>

<a href="https://github.com/PathologyDataScience/BCSS">
Breast Cancer Semantic Segmentation (BCSS) dataset
</a>
<br>
<br>
On detail of this dataset, please refer to the following paper.<br>

<a href="https://academic.oup.com/bioinformatics/article/35/18/3461/5307750?login=false">
<b>Structured crowdsourcing enables convolutional segmentation of histology images</b><br>
</a> 
Bioinformatics, Volume 35, Issue 18, September 2019, Pages 3461–3467, <br>
https://doi.org/10.1093/bioinformatics/btz083<br>
Published: 06 February 2019<br>

Mohamed Amgad, Habiba Elfandy, Hagar Hussein, Lamees A Atteya, Mai A T Elsebaie, Lamia S Abo Elnasr,<br> 
Rokia A Sakr, Hazem S E Salem, Ahmed F Ismail, Anas M Saad, Joumana Ahmed, Maha A T Elsebaie, <br>
Mustafijur Rahman, Inas A Ruhban, Nada M Elgazar, Yahya Alagha, Mohamed H Osman, Ahmed M Alhusseiny,<br> 
Mariam M Khalaf, Abo-Alela F Younes, Ali Abdulkarim, Duaa M Younes, Ahmed M Gadallah, Ahmad M Elkashash,<br> 
Salma Y Fala, Basma M Zaki, Jonathan Beezley, Deepak R Chittajallu, David Manthey, 
David A Gutman, Lee A D Cooper<br>

<br>
<b>Dataset Licensing</b><br>
This dataset itself is licensed under a CC0 1.0 Universal (CC0 1.0) license. 

<h3>2. Download master dataset</h3>
Please clone the following github repository.<br>
<a href="https://github.com/PathologyDataScience/BCSS">
Breast Cancer Semantic Segmentation (BCSS) dataset
</a>
<br><br>
Please set up Python environemnt to download BCSS dataset. 
If you would like to download all dataset, <b>images, masks, annotations</b>, probably you will have to modify <b>PIPELINE </b>
in config.py file as shown below.<br>
<pre>
# What things to download? -- comment out whet you dont want
PIPELINE = (
    'images',
    'masks',
    'annotations',
)
</pre>
Please run the following command to download the BCSS dataset <br>
<pre>
>python download_crowdsource_dataset.py
</pre>
, by which the following folders will be created.<br> 
<pre>
./Tiled-ImageMask-Dataset-Breast-Cancer
├─annotations  : JSON annotation files
├─images       ; PNG image files
├─masks        : PNG mask files
└─meta         
</pre>

<hr>
BCSS images sample<br>
<img src="./asset/bcss_images_sample.png" width=1024 height="auto"><br><br>

BCSS masks sample normalized by us<br>
<img src="./asset/bcss_normalized_masks_sample.png" width=1024 height="auto"><br><br>

<hr>

By checking <b>./meta/gtruth_codes.tsv</b>,
you can identify the labels included in this annotations dataset as shown below.<br>
<pre>
--------------------------------
label                   GT_code
--------------------------------
outside_roi             0
tumor                   1
stroma                  2
lymphocytic_infiltrate  3
necrosis_or_debris      4
glandular_secretions    5
blood                   6
exclude                 7
metaplasia_NOS          8
fat                     9
plasma_cells            10
other_immune_infiltrate 11
mucoid_material         12
normal_acinus_or_duct   13
lymphatics              14
undetermined            15
nerve                   16
skin_adnexa             17
blood_vessel            18
angioinvasion           19
dcis                    20
other                   21
</pre>
As shown above, a lot of labels is included in it. However, for simplicity, we will create ImageMask Dataset for <b>Tumor</b>.<br>

<h3>3. Create Mostly-Tumor Dataset</h3>
To create Mostly Tumor ImageMask dataset, please run the following command for Python script, 
<a href="./ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a>.<br>
<pre>
>python ImageMaskDatasetGenerator.py
</pre>

This script executes the following image processings to create Tumor-ImageMask-Dataset.<br>

Processings for image files in images folder:<br>
<pre>
1. Read a png image file in images folder of BCSS.
2. Resize the width and height of the image to be a minimum integral multiple of 512 respectively.
3. Save the resized image as a jpg file.
</pre>
Processings for the json files in annotations folder:<br>
<pre>
1. Read a json_file in annotations folder of BCSS.
2. Parse the json data, and get a header part of the data.
3. Get center, image_width, image_height from the header part.
4. Create an empty mask of image_width and imagte_height.
5. Find <b>mostly_tumor</b> group in a body part of the json data.
6. Get a set of <b>points</b> of a polygon which represents a mostly_tumor region.
7. Fill the mask with the set of points as a polygon.
8. Resize the width and height of the mask to be a minimum integral multiple of 512 respectively.
9. Save the resized mask as a jpg file.
</pre>
<b>Annotation: Header</b><br>
<img src="./asset/annotation_header_part.png" width=720 height="auto">></br>

<br>
<b>Annotation: Mostly Tumor</b><br>
<img src="./asset/annotation_mostly_tumor.png" width=720 height="auto">></br>

<br>

By using this script, the following dataset will be created.<br>
<pre>
./BCSS-Mostly-Tumor-master
├─images       ; JPG image files
└─masks        : JPG mostly_tumor mask files
</pre>

<hr>
Generateds images sample<br>
<img src="./asset/images_sample.png" width=1024 height="auto"><br><br>

Generated mostly_tumor masks sample<br>
<img src="./asset/mostly_tumor_masks_sample.png" width=1024 height="auto"><br><br>

<hr>

<h3>4. Create Tiled Dataset</h3>
To create Tiledly-Split-ImageMask-Dataset from <b>BCSS-Mostly-Tumor-master</b> , please run the following command for Python script, 
<a href="./TiledImageMaskDatasetGenerator.py">TiledImageMaskDatasetGenerator.py</a>.<br>
<pre>
>python TiledImageMaskDatasetGenerator.py
</pre>
By this command, the following folder will be created.<br>
<pre>
./Tiled-BCSS-Mostly-Tumor-master
├─images       ; Tiledly-split 512x512 jpg image files
└─masks        : Tiledly-split 512x512 jpg mask files
</pre>


<h3>5. Split master</h3>
To split Tiledly-Split-ImageMask-Dataset to test, train, and valid sub datasets,
 please run the following command for Python script, 
<a href="./split_master.py">split_master.py</a>.<br>
<pre>
>python split_master.py
</pre>
By this command, the following folder will be created.<br>
<pre>
./Tiled-Breast-Cancer-ImageMask-Dataset-X
├─test
│  ├─images
│  └─masks
├─train
│  ├─images
│  └─masks
└─valid
    ├─images
    └─masks
</pre>

<b>Dataset Statistics</b><br>
<img src="./Tiled-Breast-Cancer-ImageMask-Dataset-X_Statistics.png" width="540" height="auto"><br>

<hr>
<b>train/images samples:</b><br>
<img src="./asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>train/masks samples:</b><br>
<img src="./asset/train_masks_sample.png"  width="1024" height="auto">
<br>


<h3>6. Create Pre Augmented Non Tiled Dataset</h3>
To create Pre-Augmented-Non-Tiled-ImageMask-Dataset from <b>BCSS-Mostly-Tumor-master</b>, please run the following command for Python script, 
<a href="./PreAugmentedImageMaskDatasetGenerator.py">PreAugmentedImageMaskDatasetGenerator.py</a>.<br>
<pre>
>python PreAugmentedImageMaskDatasetGenerator.py
</pre>
By this command, the following folder will be created.<br>
<pre>
./PreAugmented-Breast-Cancer-master
├─images       ; Pre-augmented 512x512 resized jpg image files
└─masks        : Pre-augmented 512x512 resized jpg mask files
</pre>


<h3>7. Split PreAugmented master</h3>
To split PreAugmented-Breast-Cancer-master to test, train, and valid sub datasets,
 please run the following command for Python script, 
<a href="./split_preaugmented-master.py">split_preaugmented-master.py</a>.<br>
<pre>
>python split_preaugmented-master.py
</pre>
By this command, the following folder will be created.<br>
<pre>
./Non-Tiled-Breast-Cancer-ImageMask-Dataset-V1
├─test
│  ├─images
│  └─masks
├─train
│  ├─images
│  └─masks
└─valid
    ├─images
    └─masks
</pre>

<b>Dataset Statistics</b><br>
<img src="./Non-Tiled-Breast-Cancer-ImageMask-Dataset-V1_Statistics.png" width="540" height="auto"><br>

<hr>
<b>Non-Tiled train/images samples:</b><br>
<img src="./asset/non-tiled-train_images_sample.png" width="1024" height="auto">
<br>
<b>Non-Tiled train/masks samples:</b><br>
<img src="./asset/non-tiled-train_masks_sample.png"  width="1024" height="auto">
<br>


<h3>8. Create Mixed Dataset</h3>
To create Mixed-ImageMask-Dataset, Tiled and Non-Tiled datasets are mixed, from <b>BCSS-Mostly-Tumor-master</b> , please run the following command for Python script, 
<a href="./MixedImageMaskDatasetGenerator.py">MixedImageMaskDatasetGenerator.py</a>.<br>
<pre>
>python MixedImageMaskDatasetGenerator.py
</pre>
By this command, the following folder will be created.<br>
<pre>
./Mixed-BCSS-Mostly-Tumor-master-M2
├─images       ; Pre-augmented resized to 512x512, and tiledly split to 512x512 jpg image files
└─masks        : Pre-augmented resized to 512x512, and tiledly split to 512x512 jpg mask files
</pre>


<h3>7. Split Mixed master</h3>
To split Mixed-Breast-Cancer-master to test, train, and valid sub datasets,
 please run the following command for Python script, 
<a href="./split_mixed-master.py">split_mixed-master.py</a>.<br>
<pre>
>python split_mixed-master.py
</pre>
By this command, the following folder will be created.<br>
<pre>
./Mixed-Breast-Cancer-ImageMask-Dataset-M2
├─test
│  ├─images
│  └─masks
├─train
│  ├─images
│  └─masks
└─valid
    ├─images
    └─masks
</pre>

<b>Dataset Statistics</b><br>
<img src="./Mixed-Breast-Cancer-ImageMask-Dataset-M2_Statistics.png" width="540" height="auto"><br>

<hr>
<b>Mixed train/images samples:</b><br>
<img src="./asset/mixed_train_images_sample.png" width="1024" height="auto">
<br>
<b>Mixed train/masks samples:</b><br>
<img src="./asset/mixed_train_masks_sample.png"  width="1024" height="auto">
<br>

<br>
<h3>Reference</h3>
<b>1. Structured crowdsourcing enables convolutional segmentation of histology images</b><br>
Bioinformatics, Volume 35, Issue 18, September 2019, Pages 3461–3467, <br>
https://doi.org/10.1093/bioinformatics/btz083<br>
Published: 06 February 2019<br>

Mohamed Amgad, Habiba Elfandy, Hagar Hussein, Lamees A Atteya, Mai A T Elsebaie, Lamia S Abo Elnasr,<br> 
Rokia A Sakr, Hazem S E Salem, Ahmed F Ismail, Anas M Saad, Joumana Ahmed, Maha A T Elsebaie, <br>
Mustafijur Rahman, Inas A Ruhban, Nada M Elgazar, Yahya Alagha, Mohamed H Osman, Ahmed M Alhusseiny,<br> 
Mariam M Khalaf, Abo-Alela F Younes, Ali Abdulkarim, Duaa M Younes, Ahmed M Gadallah, Ahmad M Elkashash,<br> 
Salma Y Fala, Basma M Zaki, Jonathan Beezley, Deepak R Chittajallu, David Manthey, 
David A Gutman, Lee A D Cooper<br>

<pre>
https://academic.oup.com/bioinformatics/article/35/18/3461/5307750?login=false
</pre>


