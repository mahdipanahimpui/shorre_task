import cv2
from yolov5.models.experimental import attempt_load
import time
from predict import predict
from utils import compute_roi_pixels, draw, write_preprocess, release_resources


def detect_track_video(input_path, weights_path, show_video=True, output_name=None):
    """
       Detects and tracks objects in a video using a YOLOv5 object detection model and a CSRT tracker.

       Args:
           input_path (str): The path to the input video file.
           weights_path (str): The path to the weights file for the YOLOv5 model.
           show_video (bool, optional): Whether to display the video output. Default is True.
           output_name (str, optional): The name of the output video file to be saved. If not provided, the output video
           is not saved.

       Yields:
           tuple: A tuple containing the computed region of interest (ROI) pixels and the frame number for every 4th
           frame in the video.
       """
    # Load the YOLOv5 model from the given weights file
    model = attempt_load(weights_path)
    # Get the class names from the model
    names = model.module.names if hasattr(model, 'module') else model.names
    writer = None
    frame_num = 0
    # Open the input video file and get its dimensions
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize variables for object detection and tracking
    first_detect = True
    tracker = None
    success = False

    # If an output file name is provided, create a new video writer object to write the output video
    if output_name:
        writer = write_preprocess(cap, output_name, w, h)

    # Loop through the frames of the input video
    while True:
        # Read the next frame from the input video
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection and tracking every 8th frame or on the first frame
        if frame_num % 8 == 0 or first_detect:
            success = False  # to check is the tracking successful

            # Predict the bounding box, class name, and confidence for each object in the frame
            bbox, cls_name, conf = predict(frame, model, names, w, h)
            # Create a CSRT tracker object and initialize it with the predicted bounding box
            tracker = cv2.TrackerCSRT_create()
            # if detection happens run if block
            if bbox:
                # init the tracking coordinates as t_coord and initialize the tracker
                t_coord = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
                tracker.init(frame, t_coord)
                # success = True if the initialization is ok
                success = True
                # after first detection of video the first_detection flag changes to False,
                # and detection happens every few frames
                if first_detect:
                    first_detect = False

        s = time.time()
        # if the tracker initialization Occurred update the tracking in each frame
        if success:
            success, t_coord = tracker.update(frame)

        # if the tracker.update(frame) Occurred the success value changes to True
        if success:
            # temporary coordinates in format(x_start, y_start, x_end, y_end)
            temp_coord = (t_coord[0], t_coord[1],
                          t_coord[0] + t_coord[2],
                          t_coord[1] + t_coord[3])
            e = time.time()
            print('track_ok time: ', e - s)
            # draw boxes that taken from tracking
            frame = draw(frame, temp_coord)
            # yield the roi every few frames
            if frame_num % 4 == 0: yield compute_roi_pixels(frame, temp_coord), frame_num

        else:
            # yield the empty roi every few frames
            if frame_num % 4 == 0: yield 0, frame_num

        print('frame number: ', frame_num)
        frame_num += 1

        # write the video if the writer is enable
        if writer is not None:
            writer.write(frame)

        # show the video if the show_video is enable
        if show_video:
            cv2.imshow('YOLOv5', frame)
            if cv2.waitKey(1) == ord('q'):
                break

    # release the resources
    release_resources(cap, writer)








# input_path = 'video/v2.mp4'
# weights_path = 'weights/best.pt'
# detect_track_video(input_path, weights_path, show_video=True, output_name=None)







