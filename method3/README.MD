# Video Processing with Improved Performance by Frame Drop Algorithm
In video_detection.py, we use Pearson's Correlation Coeficient to evaluate how similar each frame is to the one before. By establishing a dynamic threshold (through the updated mean of the PC coeficients), we can reduce the amount of frames to be processed, by dropping them when they are similar enough to the one before, maintaining a non-deterministic approach (and increasing accuracy and speed).

## Dependencies:
    - OpenCV
    - Numpy
    - argparse
    - imutils (for imutils in Anaconda: conda install -c conda-forge imutils)
