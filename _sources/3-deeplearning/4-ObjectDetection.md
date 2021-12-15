# Object detection

Slides: [pdf](https://www.tu-chemnitz.de/informatik/KI/edu/neurocomputing/lectures/pdf/3.4-ObjectDetection.pdf)


## Object detection

<div class='embed-container'><iframe src='https://www.youtube.com/embed/arzmgsxmpRw' frameborder='0' allowfullscreen></iframe></div>


Contrary to object classification/recognition which assigns a single label to an image, **object detection** requires to both classify object and report their position and size on the image (bounding box).

```{figure} ../img/dnn_classification_vs_detection.png
---
width: 80%
---
Object recognition vs. detection. Source: <https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4>
```

A naive and very expensive method is to use a trained CNN as a high-level filter. The CNN is trained on small images and convolved on bigger images. The output is a **heatmap** of the probability that a particular object is present.

```{figure} ../img/objectdetection.png
---
width: 80%
---
Using a pretrained CNN to generate heatmaps. Source: <https://blog.athelas.com/a-brief-history-of-cnns-in-image-segmentation-from-r-cnn-to-mask-r-cnn-34ea83205de4>
```

Object detection is both a:

* **Classification** problem, as one has to recognize an object.
* **Regression** problem, as one has to predict the coordinates $(x, y, w, h)$ of the bounding box.


```{figure} ../img/localization.png
---
width: 100%
---
Object detection is both a classification and regression task. Source: <https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e>
```

The main datasets for object detection are the **PASCAL** Visual Object Classes Challenge (20 classes, ~10K images, ~25K annotated objects, <http://host.robots.ox.ac.uk/pascal/VOC/voc2008/>) and the MS COCO dataset (Common Objects in COntext, 330k images, 80 labels, <http://cocodataset.org>)


## R-CNN : Regions with CNN features

R-CNN {cite}`Girshick2014` was one of the first CNN-based architectures allowing object detection.

```{figure} ../img/rcnn.png
---
width: 80%
---
R-CNN {cite}`Girshick2014`.
```

It is a pipeline of 4 steps:

1. Bottom-up region proposals  by searching bounding boxes based on pixel info (selective search  <https://ivi.fnwi.uva.nl/isis/publications/2013/UijlingsIJCV2013/UijlingsIJCV2013.pdf>).
2. Feature extraction using a pre-trained CNN (AlexNet).
3. Classification using a SVM (object or not; if yes, which one?)
4. If an object is found, linear regression on the region proposal to generate tighter bounding box coordinates.

Each region proposal is processed by the CNN, followed by a SVM and a bounding box regressor.

```{figure} ../img/rcnn-detail.png
---
width: 70%
---
R-CNN {cite}`Girshick2014`. Source: <https://courses.cs.washington.edu/courses/cse590v/14au/cse590v_wk1_rcnn.pdf>
```

The CNN is pre-trained on ImageNet and fine-tuned on Pascal VOC (transfer learning).

```{figure} ../img/rcnn-training.png
---
width: 90%
---
R-CNN {cite}`Girshick2014`. Source: <https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e>
```

## Fast R-CNN

The main drawback of R-CNN is that each of the 2000 region proposals have to go through the CNN: extremely slow. The idea behind **Fast R-CNN** {cite}`Girshick2015` is to extract region proposals in higher feature maps and to use transfer learning.

```{figure} ../img/fast-rcnn.png
---
width: 90%
---
Fast R-CNN {cite}`Girshick2015`.
```

The network first processes the whole image with several convolutional and max pooling layers to produce a feature map. Each object proposal is projected to the feature map, where a region of interest (RoI) pooling layer extracts a fixed-length feature vector. Each feature vector is fed into a sequence of FC layers that finally branch into two sibling output layers:

* a softmax probability estimate over the K classes plus a catch-all “background” class.
* a regression layer that outputs four real-valued numbers for each class.

The loss function to minimize is a composition of different losses and penalty terms:

$$
    \mathcal{L}(\theta) = \lambda_1 \, \mathcal{L}_\text{classification}(\theta) + \lambda_2 \, \mathcal{L}_\text{regression}(\theta) + \lambda_3 \, \mathcal{L}_\text{regularization}(\theta)
$$


## Faster R-CNN


Both R-CNN and Fast R-CNN use selective search to find out the region proposals: slow and time-consuming. Faster R-CNN {cite}`Ren2016` introduces an object detection algorithm that lets the network learn the region proposals. The image is passed through a pretrained CNN to obtain a convolutional feature map. A separate network is used to predict the region proposals. The predicted region proposals are then reshaped using a RoI (region-of-interest) pooling layer which is then used to classify the object and predict the bounding box.

```{figure} ../img/faster-rcnn.png
---
width: 70%
---
Faster R-CNN {cite}`Ren2016`.
```

## YOLO


<div class='embed-container'><iframe src='https://www.youtube.com/embed/EuAb0UdUnUM' frameborder='0' allowfullscreen></iframe></div>

(Fast(er)) R-CNN perform classification for each region proposal sequentially: slow. YOLO (You Only Look Once) {cite}`Redmon2016` applies a single neural network to the full image to predict all possible boxes and the corresponding classes. YOLO divides the image into a SxS grid of cells.

```{figure} ../img/yolo.png
---
width: 90%
---
YOLO {cite}`Redmon2016`.
```


Each grid cell predicts a single object, with the corresponding $C$ **class probabilities** (softmax). It also predicts the coordinates of $B$ possible **bounding boxes** (x, y, w, h) as well as a box **confidence score**. The SxSxB predicted boxes are then pooled together to form the final prediction.


In the figure below, the yellow box predicts the presence of a **person** (the class) as well as a candidate **bounding box** (it may be bigger than the grid cell itself).

```{figure} ../img/yolo1.jpeg
---
width: 90%
---
Each cell predicts a class (e.g. person) and the (x, y, w, h) coordinates of the bounding box. Source: <https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088>
```

In the original YOLO implementation, each grid cell proposes 2 bounding boxes:

```{figure} ../img/yolo2.jpeg
---
width: 90%
---
Each cell predicts two bounding boxes per object. Source: <https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088>
```

Each grid cell predicts a probability for each of the 20 classes, two bounding boxes (4 coordinates  per bounding box) and their confidence scores. This makes C + B * 5 = 30 values to predict for each cell.


```{figure} ../img/yolo3.jpeg
---
width: 100%
---
Each cell outputs 30 values: 20 for the classes and 5 for each bounding box, including the confidence score. Source: <https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088>
```

### Architecture of the CNN

YOLO uses a CNN with 24 convolutional layers and 4 max-pooling layers to obtain a 7x7 grid. The last convolution layer outputs a tensor with shape (7, 7, 1024). The tensor is then flattened and passed through 2 fully connected layers. The output is a tensor of shape (7, 7, 30), i.e. 7x7 grid cells, 20 classes and 2 boundary box predictions per cell.

```{figure} ../img/yolo-cnn.png
---
width: 100%
---
Architecture of the CNN used in YOLO {cite}`Redmon2016`.
```

### Confidence score

The 7x7 grid cells predict 2 bounding boxes each: maximum of 98 bounding boxes on the whole image. Only the bounding boxes with the **highest class confidence score** are kept.

$$
    \text{class confidence score = box confidence score * class probability}
$$

In practice, the class confidence score should be above 0.25.

```{figure} ../img/yolo4.png
---
width: 100%
---
Only the bounding boxes with the highest class confidence scores are kept among the 98 possible ones. Source: {cite}`Redmon2016`.
```

### Intersection over Union (IoU)

To ensure specialization, only one bounding box per grid cell should be responsible for detecting an object. During learning, we select the bounding box with the biggest overlap with the object. This can be measured by the **Intersection over the Union** (IoU).

```{figure} ../img/iou1.jpg
---
width: 60%
---
The Intersection over Union (IoU) measures the overlap between bounding boxes. Source: <https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/>
```

```{figure} ../img/iou2.png
---
width: 60%
---
The Intersection over Union (IoU) measures the overlap between bounding boxes. Source: <https://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/>
```

### Loss functions

The output of the network is a 7x7x30 tensor, representing for each cell:

* the probability that an object of a given class is present.
* the position of two bounding boxes.
* the confidence that the proposed bounding boxes correspond to a real object (the IoU).

We are going to combine three different loss functions:

1. The **categorization loss**: each cell should predict the correct class.
2. The **localization loss**: error between the predicted boundary box and the ground truth for each object.
3. The **confidence loss**: do the predicted bounding boxes correspond to real objects?

**Classification loss**

The classification loss is the **mse** between:

* $\hat{p}_i(c)$: the one-hot encoded class $c$ of the object present under each cell $i$, and
* $p_i(c)$: the predicted class probabilities of cell $i$.

$$
    \mathcal{L}_\text{classification}(\theta) =  \sum_{i=0}^{S^2} \mathbb{1}_i^\text{obj} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2
$$

where $\mathbb{1}_i^\text{obj}$ is 1 when there actually is an object behind the cell $i$, 0 otherwise (background).

They could also have used the cross-entropy loss, but the output layer is not a regular softmax layer. Using mse is also more compatible with the other losses.

**Localization loss**

For all bounding boxes matching a real object, we want to minimize the **mse** between:

* $(\hat{x}_i, \hat{y}_i, \hat{w}_i, \hat{h}_i)$: the coordinates of the ground truth bounding box, and
* $(x_i, y_i, w_i, h_i)$: the coordinates of the predicted bounding box.

$$
    \mathcal{L}_\text{localization}(\theta) = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^\text{obj} [ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2] \\
    \qquad\qquad\qquad\qquad + \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^\text{obj} [ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2]
$$

where $\mathbb{1}_{ij}^\text{obj}$ is 1 when the bounding box $j$ of cell $i$ "matches" with an object (IoU). The root square of the width and height of the bounding boxes is used. This allows to penalize more the errors on small boxes than on big boxes.

**Confidence loss**

Finally, we need to learn the confidence score of each bounding box, by minimizing the **mse** between:

* $C_i$: the predicted confidence score of cell $i$, and
* $\hat{C}_i$: the IoU between the ground truth bounding box and the predicted one.

$$
    \mathcal{L}_\text{confidence}(\theta) = \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^\text{obj} (C_{ij} - \hat{C}_{ij})^2  \\
    \qquad\qquad\qquad\qquad + \lambda^\text{noobj} \, \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^\text{noobj} (C_{ij} - \hat{C}_{ij})^2
$$

Two cases are considered:

1. There was a real object at that location ($\mathbb{1}_{ij}^\text{obj} = 1$): the confidences should be updated fully.
2. There was no real object ($\mathbb{1}_{ij}^\text{noobj} = 1$): the confidences should only be moderately updated ($\lambda^\text{noobj} = 0.5$)

This is to deal with **class imbalance**: there are much more cells on the background than on real objects.

Put together, the loss function to minimize is:

$$
\begin{align}
    \mathcal{L}(\theta) & = \mathcal{L}_\text{classification}(\theta) + \lambda_\text{coord} \, \mathcal{L}_\text{localization}(\theta) + \mathcal{L}_\text{confidence}(\theta) \\
              & = \sum_{i=0}^{S^2} \mathbb{1}_i^\text{obj} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2 \\
              & + \lambda_\text{coord} \, \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^\text{obj} [ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2] \\
              & + \lambda_\text{coord} \, \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^\text{obj} [ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2] \\
              & + \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^\text{obj} (C_{ij} - \hat{C}_{ij})^2  \\
              & + \lambda^\text{noobj} \, \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^\text{noobj} (C_{ij} - \hat{C}_{ij})^2 \\
\end{align}
$$

### YOLO  trained on PASCAL VOC

YOLO was trained on PASCAL VOC (natural images) but generalizes well to other datasets (paintings...). YOLO runs in real-time (60 fps) on a NVIDIA Titan X.
Faster and more accurate versions of YOLO have been developed: YOLO9000 {cite}`Redmon2016a`, YOLOv3 {cite}`Redmon2018`, YOLOv5 (<https://github.com/ultralytics/yolov5>)...

```{figure} ../img/yolo-result2.png
---
width: 60%
---
Performance of YOLO compared to the state of the art. Source: {cite}`Redmon2016`.
```

Refer to the website of the authors for additional information: <https://pjreddie.com/darknet/yolo/> 

<div class='embed-container'><iframe src='https://www.youtube.com/embed/MPU2HistivI' frameborder='0' allowfullscreen></iframe></div>


## SSD


The idea of SSD (Single-Shot Detector, {cite}`Liu2016`) is similar to YOLO, but:

* faster
* more accurate
* not limited to 98 objects per scene
* multi-scale

Contrary to YOLO, all convolutional layers are used to predict a bounding box, not just the final tensor: **skip connections**. This allows to detect boxes at multiple scales (pyramid).

```{figure} ../img/ssd.png
---
width: 100%
---
Single-Shot Detector, {cite}`Liu2016`.
```

## 3D object detection

It is also possible to use **depth** information (e.g. from a Kinect) as an additional channel of the R-CNN. The depth information provides more information on the structure of the object, allowing to disambiguate certain situations (segmentation).

```{figure} ../img/rcnn-rgbd.png
---
width: 100%
---
Learning Rich Features from RGB-D Images for Object Detection, {cite}`Gupta2014`.
```

Lidar point clouds can also be used for detecting objects, for example **VoxelNet**  {cite}`Zhou2017` trained in the KITTI dataset.


```{figure} ../img/voxelnet.png
---
width: 100%
---
VoxelNet {cite}`Zhou2017`.
```

```{figure} ../img/voxelnet-result.png
---
width: 100%
---
VoxelNet {cite}`Zhou2017`. Source: <https://medium.com/@SmartLabAI/3d-object-detection-from-lidar-data-with-deep-learning-95f6d400399a>
```


```{admonition} Additional resources on object detection

<https://medium.com/comet-app/review-of-deep-learning-algorithms-for-object-detection-c1f3d437b852>

<https://medium.com/@smallfishbigsea/faster-r-cnn-explained-864d4fb7e3f8>

<https://towardsdatascience.com/r-cnn-fast-r-cnn-faster-r-cnn-yolo-object-detection-algorithms-36d53571365e>

<https://medium.com/@jonathan_hui/real-time-object-detection-with-yolo-yolov2-28b1b93e2088>

<https://medium.com/@jonathan_hui/ssd-object-detection-single-shot-multibox-detector-for-real-time-processing-9bd8deac0e06>

<https://towardsdatascience.com/lidar-3d-object-detection-methods-f34cf3227aea>
```