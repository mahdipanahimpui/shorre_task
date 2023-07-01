import cv2
import os


def convert_videos_to_images(video_directory, image_directory):
    """
    Converts all videos in a directory to images and saves them in a directory with the same name as the video file.

    Args:
        video_directory (str): Path to the directory containing video files.
        image_directory (str): Path to the directory for storing images.
    """
    # Create a directory for storing images if it does not exist
    if not os.path.exists(image_directory):
        os.makedirs(image_directory)

    # Traverse the video directory and convert each video file to images
    for filename in os.listdir(video_directory):
        if not filename.endswith(".mp4"):
            continue

        print(f"Converting {filename} to images...")

        # Path to the video file
        video_path = os.path.join(video_directory, filename)

        # Create a VideoCapture object using the video file path
        cap = cv2.VideoCapture(video_path)

        # Counter for images
        count = 0

        # Check if the video file was opened successfully
        if not cap.isOpened():
            print(f"Could not open video file: {filename}")
            continue

        # Read the first frame from the video
        success, image = cap.read()

        # Create a directory for storing images if it does not exist
        video_name = os.path.splitext(filename)[0]
        video_image_directory = os.path.join(image_directory, video_name)
        if not os.path.exists(video_image_directory):
            os.makedirs(video_image_directory)

        # Save the frames as images until the end of the video
        while success:
            # Filename for the current image
            image_filename = os.path.join(video_image_directory, f"{video_name}_frame{count}.jpg")

            # Save the image
            cv2.imwrite(image_filename, image)

            # Increase the counter
            count += 1

            # Read the next frame
            success, image = cap.read()

        print(f"Finished converting {filename} to {count} images.")

        # Release the VideoCapture object
        cap.release()

    print("All videos converted to images successfully.")



def main():
    video_directory = "./video"
    image_directory = "./img"
    convert_videos_to_images(video_directory, image_directory)

if __name__ == "__main__":
    main()