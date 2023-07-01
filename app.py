from flask import Flask, render_template, request
from flask_socketio import SocketIO
from threading import Lock
from track_detect import detect_track_video
import argparse


thread = None
thread_lock = Lock()

# Set up the Flask app and SocketIO server
app = Flask(__name__)
app.config['SECRET_KEY'] = 'donsky!'
socketio = SocketIO(app, cors_allowed_origins='*')




# Define the necessary and optional arguments using argparse
parser = argparse.ArgumentParser(description='run the shorre detection realtime')
parser.add_argument('video_source', type=str, help='the video source')
parser.add_argument('weights_path', type=str, help='the model weights path')
parser.add_argument('--save', type=str,
                    help='the output video name, (optional, if there is no name, could not save the video)')

args = parser.parse_args()

input_path = args.video_source
weights_path = args.weights_path
output_video = args.save or None


# Define a background thread that runs the detect_track_video function and emits sensor data to clients
def background_thread():
    print("Generating sensor values")
    for shorre_value, frame_num in detect_track_video(input_path, weights_path, show_video=True, output_name=output_video):
        socketio.emit('updateSensorData', {'value': shorre_value, "date": frame_num})
        socketio.sleep(0.000001)


# Set up a route for the index page
@app.route('/')
def index():
    return render_template('index.html')


# Handle the client connecting to the SocketIO server
@socketio.on('connect')
def connect():
    global thread
    print('Client connected')
    with thread_lock:
        if thread is None:
            thread = socketio.start_background_task(background_thread)


# Handle the client disconnecting from the SocketIO server
@socketio.on('disconnect')
def disconnect():
    print('Client disconnected', request.sid)

# Run the SocketIO server when the script is executed
if __name__ == '__main__':
    socketio.run(app)