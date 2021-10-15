import numpy as np
import cv2
import face_alignment
import sys
import matplotlib.pyplot as plt
import os
from glob import glob
# plt.ioff()
# Initialize the chip resolution

# Initialize the face alignment tracker
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=True, device="cpu")

if os.path.isfile(sys.argv[1]):
    _file = [sys.argv[1]]
elif os.path.isdir(sys.argv[1]):
    _file = glob(os.path.join(sys.argv[1],'*.jpg'))
else:
    raise "Please supply an argument"

Frame = []
Chip = []
for idx, f in enumerate(_file):
    frame = cv2.imread(f)
    Frame.append(frame)
    # chipSize = (300, 300)
    # chipSize = (512,512)
    chipSize = frame.shape[:-1][::-1]
    chipCorners = np.float32([[0,0],
                                [chipSize[0],0],
                                [0,chipSize[1]],
                                [chipSize[0],chipSize[1]]])
    imagePoints = fa.get_landmarks_from_image(frame)

    imagePoints = imagePoints[0]

    # Compute the Anchor Landmarks
    # This ensures the eyes and chin will not move within the chip
    rightEyeMean = np.mean(imagePoints[36:42], axis=0)
    leftEyeMean  = np.mean(imagePoints[42:47], axis=0)
    middleEye    = (rightEyeMean + leftEyeMean) * 0.5
    chin         = imagePoints[8]
    #cv2.circle(frame, tuple(rightEyeMean[:2].astype(int)), 30, (255,255,0))
    #cv2.circle(frame, tuple(leftEyeMean [:2].astype(int)), 30, (255,0,255))

    # Compute the chip center and up/side vectors
    mean = ((middleEye * 3) + chin) * 0.25
    centered = imagePoints - mean 
    rightVector = (leftEyeMean - rightEyeMean)
    upVector    = (chin        - middleEye)

    # Divide by the length ratio to ensure a square aspect ratio
    rightVector /= np.linalg.norm(rightVector) / np.linalg.norm(upVector)

    # Compute the corners of the facial chip
    imageCorners = np.float32([(mean + ((-rightVector - upVector)))[:2],
                                (mean + (( rightVector - upVector)))[:2],
                                (mean + ((-rightVector + upVector)))[:2],
                                (mean + (( rightVector + upVector)))[:2]])

    # Compute the Perspective Homography and Extract the chip from the image
    chipMatrix = cv2.getPerspectiveTransform(imageCorners, chipCorners)
    chip = cv2.warpPerspective(frame, chipMatrix, (chipSize[0], chipSize[1]))
    Chip.append(chip)
    if ((idx+1) % 10) == 0 or len(_file) == 1:
        # plt.figure()
        figure, axes = plt.subplots(nrows=2, ncols=len(Frame))
        for k, (_f, c) in enumerate(zip(Frame, Chip)):
            # import ipdb; ipdb.set_trace()
            axes[0, k].imshow(_f[:,:,::-1])
            axes[0, k].axis('off')
            axes[1, k].imshow(c[:,:,::-1])
            axes[1, k].axis('off')
        figure.tight_layout()     
        # figure.show()
        Chip = []
        Frame = []
        input('Press enter to continue.')