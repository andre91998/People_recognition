# Pedestrian Recognition via OpenCV's HOG + Linear SVM
  Inspired by Andrew Rosebrock @ https://www.pyimagesearch.com/2015/11/09/pedestrian-detection-opencv/

  OpenCV ships with a pre-trained HOG + Linear SVM model based, on Dalal and Triggs method to automatically detect pedestrians in images, that can be used to perform pedestrian detection in both images and video streams. Below we give an outline of the two:

## Dependencies: 
    - OpenCV
    - Numpy
    - argparse
    - imutils (for imutils in Anaconda: conda install -c conda-forge imutils)
  
## Histogram of Oriented Gradients (HOG)

While OpenCV's Cascade Classifiers are fast, they leave much to desire. That's where HOG comes in. But if you want to check out cascade classifiers, a java application for face recognition can be found [here](https://github.com/andre91998/JavaBasics/tree/master/FaceDetection).

### HOG algorithm:

<ul><li><strong>Step 1:</strong> Sample P positive samples from your training data of the object(s) you want to detect and extract HOG descriptors from these samples. </ul></li>

<ul><li><strong>Step 2:</strong> Sample N negative samples from a negative training set that does not contain any of the objects you want to detect and extract HOG descriptors from these samples as well. In practice N >> P.</ul></li>

<ul><li><strong>Step 3:</strong> Train a <strong>Linear Support Vector Machine</strong> on your positive and negative samples.</ul></li>

<ul><li><strong>Step 4:</strong> Apply <em>hard-negative mining</em>. For each image and each possible scale of each image in your negative training set, apply the sliding window technique and slide your window across the image. At each window compute your HOG descriptors and apply your classifier. If your classifier (incorrectly) classifies a given window as an object (and it will, there will absolutely be false-positives), record the feature vector associated with the false-positive patch along with the probability of the classification. </ul></li>

<ul><li><strong>Step 5:</strong> Take the false-positive samples found during the hard-negative mining stage, sort them by their confidence (i.e. probability) and re-train your classifier using these hard-negative samples. (Note: You can iteratively apply steps 4-5, but in practice one stage of hard-negative mining usually [not not always] tends to be enough. The gains in accuracy on subsequent runs of hard-negative mining tend to be minimal.)</ul></li>

<ul><li><strong>Step 6: </strong>Your classifier is now trained and can be applied to your test dataset.</ul></li>

## Linear Support Vector Machines (SVM)

SVMs are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. More formally, a support-vector machine constructs a hyperplane or set of hyperplanes in a high- or infinite-dimensional space, which can be used for classification, regression, or other tasks like outliers detection.

In the **linear** case (ours), the goal is to find the *"maximum-margin hyperplane"* that divides the data points into their correct classes, which means we want the distance between the closest data point of each class to be maximized.


  

