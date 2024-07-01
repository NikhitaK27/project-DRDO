import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
import ORB_SLAM2
from a_star import a_star

# Load YOLO model
def load_yolo():
    net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return net, output_layers

# Load Mask R-CNN model
def load_mask_rcnn():
    model = maskrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model

# Load SLAM system
def load_slam(vocab_path, settings_path):
    slam = ORB_SLAM2.System(vocab_path, settings_path, ORB_SLAM2.Sensor.RGBD)
    return slam

# Object Detection using YOLO
def detect_objects_yolo(img, net, output_layers):
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    boxes, confidences, class_ids = [], [], []
    height, width, channels = img.shape
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [(boxes[i], class_ids[i], confidences[i]) for i in indexes.flatten()]

# Object Segmentation using Mask R-CNN
def segment_objects_mask_rcnn(img, model, threshold=0.5):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
    img_tensor = transform(img)
    with torch.no_grad():
        prediction = model([img_tensor])
    masks, boxes, labels = [], [], []
    pred_score = list(prediction[0]['scores'].numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    masks = (prediction[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()
    masks = masks[:pred_t+1]
    boxes = [[(i[0], i[1]), (i[2], i[3])] for i in prediction[0]['boxes'].detach().cpu().numpy()]
    boxes = boxes[:pred_t+1]
    labels = prediction[0]['labels'].detach().cpu().numpy()
    labels = labels[:pred_t+1]
    return masks, boxes, labels

# Main integration function
def integrate_all(img_path, yolo_net, yolo_output_layers, mask_rcnn_model, slam_system, goal_location):
    # Read input image
    img = cv2.imread(img_path)
    
    # Object Detection
    yolo_detections = detect_objects_yolo(img, yolo_net, yolo_output_layers)
    
    # Object Segmentation
    masks, boxes, labels = segment_objects_mask_rcnn(img, mask_rcnn_model)
    
    # Initialize SLAM and update with current frame
    pose = slam_system.TrackMonocular(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 0)
    
    # Path Planning (A* algorithm)
    current_location = pose[:2]  # Assuming pose contains (x, y) coordinates
    path = a_star(current_location, goal_location)
    
    # Display results
    for box, class_id, confidence in yolo_detections:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    for i in range(len(masks)):
        mask = masks[i]
        rgb_mask = np.zeros_like(img)
        color = np.random.randint(0, 255, 3)
        for j in range(3):
            rgb_mask[:, :, j] = mask * color[j]
        img = cv2.addWeighted(img, 1, rgb_mask, 0.5, 0)
        cv2.rectangle(img, boxes[i][0], boxes[i][1], color, 2)
    
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return path

if __name__ == "__main__":
    # Load models
    yolo_net, yolo_output_layers = load_yolo()
    mask_rcnn_model = load_mask_rcnn()
    slam_system = load_slam('path/to/ORBvoc.txt', 'path/to/Settings.yaml')
    
    # Define goal location (example)
    goal_location = (10, 10)  # Replace with actual goal coordinates
    
    # Integrate all components
    img_path = "kitchen.jpg"  # Change this to the path of your image file
    path = integrate_all(img_path, yolo_net, yolo_output_layers, mask_rcnn_model, slam_system, goal_location)
    
    print("Planned Path:", path)
