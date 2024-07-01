import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
    'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'TV',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def get_prediction(img_path, threshold):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    img = transform(img)
    with torch.no_grad():
        prediction = model([img])
    pred_score = list(prediction[0]['scores'].numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    masks = (prediction[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    masks = masks[:pred_t+1]
    boxes = [[(i[0], i[1]), (i[2], i[3])] for i in prediction[0]['boxes'].detach().cpu().numpy()]
    boxes = boxes[:pred_t+1]
    labels = prediction[0]['labels'].detach().cpu().numpy()
    labels = labels[:pred_t+1]
    return masks, boxes, labels

def draw_segmentation_map(img_path, threshold=0.5):
    masks, boxes, labels = get_prediction(img_path, threshold)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(len(masks)):
        mask = masks[i]
        rgb_mask = np.zeros_like(img)
        color = np.random.randint(0, 255, 3)
        for j in range(3):
            rgb_mask[:, :, j] = mask * color[j]
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color, 2)
        label = COCO_INSTANCE_CATEGORY_NAMES[labels[i]]
        cv2.putText(img, label, (int(boxes[i][0][0]), int(boxes[i][0][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    plt.figure(figsize=(20, 30))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    img_path = "kitchen.jpg"  
    draw_segmentation_map(img_path, threshold=0.5)
