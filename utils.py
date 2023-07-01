import cv2
import time
import os


def compute_roi_pixels(img, bbox):
    """
        Computes the number of black pixels in the region of interest (ROI) defined by the given
         bounding box.

        Args:
            img (np.ndarray): The input image
            bbox (tuple): The bounding box coordinates in the format (x1, y1, x2, y2)

        Returns:
            int: The number of black pixels in the ROI
        """
    # Extract the ROI from the input image
    x1, y1, x2, y2 = bbox
    thresh_img = apply_threshold(img)
    roi = thresh_img[y1:y2, x1:x2]

    # Calculate number of black pixels
    num_black_pixels = cv2.countNonZero(roi)
    print(f'Number of black pixels in ROI: {num_black_pixels}')
    return num_black_pixels


def apply_threshold(img):
    """
        Applies a threshold to the input image to convert it to a binary image.

        Args:
            img (np.ndarray): The input image

        Returns:
            np.ndarray: The thresholded binary image
        """
    # Apply a threshold to the input image
    thresh_value = 195
    max_value = 255
    ret, thresh = cv2.threshold(img, thresh_value, max_value, cv2.THRESH_TOZERO_INV)
    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)

    return gray


def draw(frame, bbox, cls_name='shorre', conf=''):
    """
        Draws a bounding box and class label on the input image.

        Args:
            frame (np.ndarray): The input image
            bbox (tuple): The bounding box coordinates in the format (x1, y1, x2, y2)
            cls_name (str): The predicted class name
            conf (float): The confidence score of the prediction
        """
    if conf != '':
        conf = f'{conf:.2f}'

    img = frame
    x1, y1, x2, y2 = bbox
    # Draw the bounding box and class label on the input image
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, f'{cls_name} {conf}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 2)
    return img


def write_preprocess(cap, filename, w, h, FPS=None, target_path='output'):
    """
        Writes preprocessed video frames to a new video file with the specified name.

        Args:
            cap (cv2.VideoCapture): The input video capture object.
            output_name (str): The name of the output video file to be saved.
            w (int): The width of the video frames.
            h (int): The height of the video frames.
            FPS (float, optional): The frames per second rate of the output video.
            If not provided, it is obtained from the input video.
            folder_target (string, optional): the target path to write video

        Returns:
            writer (cv2.VideoWriter): The video writer object used to write the output video.
        """
    fps = FPS
    # mp4 type for video to write
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    if fps is None:
        fps = cap.get(cv2.CAP_PROP_FPS)

    if not os.path.exists(target_path):
        os.makedirs(target_path)

    writer = cv2.VideoWriter(os.path.join(target_path, f'{filename}.mp4'), fourcc, fps, (w, h))
    return writer


def release_resources(cap, writer=None):
    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
