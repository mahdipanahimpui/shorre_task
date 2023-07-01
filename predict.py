import cv2
import torch
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
import time


def preprocess_img(img, device, w, h):
    """
        Preprocesses an input image for use with the YOLOv5 object detection model.

        Args:
            img (np.ndarray): The input image
            device (str): The device to run the model on ('cpu' or 'cuda')
            w (int): The width to resize the image to
            h (int): The height to resize the image to

        Returns:
            torch.Tensor: The preprocessed image as a PyTorch tensor
        """

    # Resize the input image and convert it to RGB
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Convert image to torch.Tensor
    img = torch.from_numpy(img).to(device).float()

    # Normalize image
    img /= 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)

    return img


def predict(frame, model, names, w, h, device='cpu', conf_thresh=0.25, iou_thresh=0.45):
    """
        Uses the YOLOv5 object detection model to predict the bounding box, class name, and
        confidence score of an object
        in the input image.

        Args:
            frame (np.ndarray): The input image
            model (torch.nn.Module): The YOLOv5 object detection model
            names (list): The list of class names
            w (int): The width to resize the image to
            h (int): The height to resize the image to
            device (str): The device to run the model on ('cpu' or 'cuda')
            conf_thresh (float): The confidence threshold for predictions
            iou_thresh (float): The IOU threshold for non-maximum suppression

        Returns:
            Detection: The detected object as a Detection object (bbox, class_name, confidence),
            or None if no object is
            detected
        """

    s = time.time()
    # apply the device environment
    device_env = select_device(device)
    img = preprocess_img(frame, device_env, w, h)

    pred = model(img)[0]
    pred = non_max_suppression(pred, conf_thresh, iou_thresh, classes=None, agnostic=False, max_det=1)

    # Process detections
    if pred is not None:
        for det in pred:
            if len(det) > 0:

                cls = int(det[0][-1])
                conf = float(det[0][-2])
                cls_name = names[cls]

                x1, y1, x2, y2 = [int(i) for i in det[0][:4]]

                e = time.time()
                print('predict ok: ', e - s)

                return (x1, y1, x2, y2), cls_name, conf

            else:
                e = time.time()
                print('predict failure: ', e - s)
                return None, None, None