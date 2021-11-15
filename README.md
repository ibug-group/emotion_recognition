# ibug.emotion_recognition
Emotion recgoniser based on [EmoNet](https://rdcu.be/cdnWi) with some pretrained weights. Our training code will be released soon.

## Prerequisites
* [Numpy](https://www.numpy.org/): `$pip3 install numpy`
* [PyTorch](https://pytorch.org/): `$pip3 install torch torchvision`
* [OpenCV](https://opencv.org/) (only needed by the test script): `$pip3 install opencv-python`
* [ibug.face_detection](https://github.com/hhj1897/face_detection) (only needed by the test script). See this repository for details: [https://github.com/hhj1897/face_detection](https://github.com/hhj1897/face_detection).
* [ibug.face_alignment](https://github.com/hhj1897/face_alignment). See this repository for details: [https://github.com/hhj1897/face_alignment](https://github.com/hhj1897/face_alignment)

## How to Install
```
git clone https://github.com/hhj1897/emotion_recognition.git
cd emotion_recognition
pip install -e .
```

## How to Test
* To test on live video: `python emotion_recognition_test.py [-i webcam_index]`
* To test on a video file: `python emotion_recognition_test.py [-i input_file] [-o output_file]`

## How to Use
```python
# Import the libraries
import cv2
from ibug.face_detection import RetinaFacePredictor
from ibug.face_alignment import FANPredictor
from ibug.emotion_recognition import EmoNetPredictor

# Create a RetinaFace detector using Resnet50 backbone, with the confidence 
# threshold set to 0.8
face_detector = RetinaFacePredictor(
    thershold=0.8, device='cuda:0',
    model=RetinaFacePredictor.get_model('resnet50'))

# Create a facial landmark detector
# Note:
#   1. The landmark detector is being used the feature extractor for EmoNet.
#   2. Because of this, you must load the same weights as what were used
#      during the training of the EmoNet model. Fow now, please load 2dfan2
#      in all cases.
landmark_detector = FANPredictor(
    device='cuda:0', model=FANPredictor.get_model('2dfan2'))

# Create a emotion recogniser
emo_rec = EmoNetPredictor(
    device='cuda:0', model=EmoNetPredictor.get_model('emonet248'))

# Load a test image. Note that images loaded by OpenCV adopt the B-G-R channel
# order.
image = cv2.imread('test.png')

# Detect faces from the image
detected_faces = face_detector(image, rgb=False)

# Use the landmark detector to extract features
_, _, feats = landmark_detector(
    image, detected_faces, rgb=False, return_features=True)

# Emotion recognition
emotions = emo_rec(feats)
```

## References
\[1\] Toisoul, Antoine, Jean Kossaifi, Adrian Bulat, Georgios Tzimiropoulos, and Maja Pantic. "[Estimation of continuous valence and arousal levels from faces in naturalistic conditions.](https://rdcu.be/cdnWi)" _Nature Machine Intelligence_ 3, no. 1 (2021): 42-50.
