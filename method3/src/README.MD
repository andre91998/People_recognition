# Code Usage and Important Comments

  - If you already had *imutils*  installed, make sure it is at least version v0.3.1, which includes the implementation of the **non_max_suppression**  function, along with a few other minor updates. This method is important for avoiding extra bounding boxes in our output, which helps to reduce *false-positives*. For more information on how Non-Maximum Supression works, refer to [here](https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/).
  
  - Use the following argument format for running the code with images as input:
  
        python ped_detect.py --images path\to\your\directory
  
  - Use the following format for running cod with video file/stream (explained below):
  
        python ped_detect_tuned.py --video path\to\your\directory --buffer buffersize
   
  - The HOG method requires the *detectMultiScale* parameters to be set properly depending on your the images that you're processing and to tune the trade-off between accuracy and performance (especially important for real-time video processing on resource constrained devices: arduino, raspberry Pi, etc..).
  
  - ped_detect.py is the code found [here](), unaltered
  - ped_detect_tuned.py is our altered version for increased speed, accuracy, and real-time application
 
  - To be process a batch of images or a video/video-stream, you must include a loop to run through the images/frames. More specifically, in the case of processing video:
  
    - For real-time processing, *imutils.video*'s  *VideoStream* is the best for handling the camera frames in a threaded approach. But for handling a video file, *cv2.VideoCapture*  does the best job. Notice that in ped_detect_tuned.py, instead of the *--images* switch, we use:
      - *--video* for processing a video file (ommiting results in use of camera)
      - *--buffer* for controlling the maximum size of the *deque* of points. The larger the *deque*  the more (x, y)-coordinates of the object are tracked, essentially giving you a larger “history” of where the object has been in the video stream. The argument default value is **32**, indicating that we’ll maintain a buffer of (x, y)-coordinates of our object for the previous 32 frames.
  
  
 ### The Parameters to the cv2.HOGDescriptor(). DetectMultiScale Function:
 
    - img (required)
    - hitThreshold (optional)
    - winStride (optional)
    - padding (optional)
    - scale (optional)
    - finalThreshold (optional)
    - useMeanShiftGrouping (optional)
    
The values of these parameters are of extreme importance, as they can:

  - Increase the number of false-positive detections.
  
  - Result in missing a detection entirely.

  - Dramatically affect the speed of the detection process.
  
For a detailed description of what each parameter means and how to tune them, see [here](https://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/).

### Tips on speeding up the object detection process:

  - Resize your image or frame to be as small as possible *without sacrificing detection accuracy*: Prior to calling the detectMultiScale  function, reduce the width and height of your image. The smaller your image is, the less data there is to process, and thus the detector will run faster.
  - Tune your *scale*  and *winStride*  parameters. These two arguments have a tremendous impact on your object detector speed. Both scale  and winStride  should be as large as possible, again, *without sacrificing detector accuracy*.
  - If your detector still is not fast enough… you *might* want to look into re-implementing your program in C/C++. Sometimes you need the compiled binary speed of C, especially in resource constrained environments.
