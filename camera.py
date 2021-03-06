import cv2
import pickle
import torch
import detectron2
import os
from torch.nn import AdaptiveAvgPool2d
from torch.nn import Linear, Module
from torch.nn.functional import softmax
from torch.nn.functional import relu
from torch.nn import BatchNorm1d
import numpy as np
from detectron2.data.detection_utils import read_image
from torch.utils.data import Dataset, DataLoader


os.environ["CUDA_VISIBLE_DEVICES"]=""
# Define yoga model with the backbone from dense pose
class YogaPoseEstimatorModel(Module):
    def __init__(self, backbone, num_classes, pixel_mean, pixel_std):
        super().__init__()
        self.backbone = backbone
        self.avg_pool = AdaptiveAvgPool2d((1, 1))
        self.fc1 = Linear(256, 64)
        self.bn1 = BatchNorm1d(64)
        self.fc2 = Linear(64, num_classes)
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

    def forward(self, x):
        x = self.preprocess(x)
        x = self.backbone(x)['p6']
        x = self.avg_pool(x)
        x = x.view((-1, 256))
        x = self.fc1(x)
        x = self.bn1(x)
        x = relu(x)
        x = self.fc2(x)
        x = softmax(x, dim=1)
        return x

    def preprocess(self, tensor):
        # Preprocessing from the source code for the original model
        tensor = (tensor - self.pixel_mean) / self.pixel_std
        return tensor

torch.device('cpu')

poses = ['bridge', 'childs', 'downwarddog', 'mountain', 'plank', 'seatedforwardbend', 'tree', 'trianglepose', 'warrior1', 'warrior2']

#Load model with pickle
my_model = pickle.load(open('densepose_model.sav', 'rb'))

# Start Camera App
print(cv2.__version__)
cv2.namedWindow("Yoga-Pose-Estimation")
cam = cv2.VideoCapture(0)

if cam.isOpened():
    rval, frame = cam.read()
else:
    rval = False

count = 0 # Count frames
predictedd = 3
font = cv2.FONT_HERSHEY_SIMPLEX

while rval:
    cv2.imshow("Yoga-Pose-Estimation", frame)
    rval, frame = cam.read()
    count = count+1
    print(count)

    # YOGA POSE ESTIMATION #
    # Get frame and load image
    cv2.imwrite('poseframe.png', frame)
    image = read_image('poseframe.png', format='BGR')
    height, width, _ = image.shape
    transform = detectron2.data.transforms.transform.ResizeTransform(h=height, w=width, new_h=800, new_w=800, interp=2)
    image = transform.apply_image(image)
    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

    test_loader = DataLoader([(image, 0)], batch_size=1, shuffle=False, pin_memory=True)

    if count % 20 == 0:
        # Predict
        with torch.no_grad():
            for data in test_loader:
                x, y = data
                x = x
                y = y
                out = my_model(x) # Pose prediction
                _, predicted = torch.max(out.data, 1)
                predictedd = predicted
                print(poses[predicted])
                # Write predicted pose to frame
                cv2.putText(frame, poses[predicted], (10, 450), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, poses[predictedd], (10, 450), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break
    if key == 32: # space bar
        cv2.imwrite('poseframe.png', frame)
cam.release()
cv2.destroyWindow("Yoga-Pose-Estimation")