"""
*******************************************************************
  Copyright (c) 2022, Aerospace Engineering, TU Delft and NLR.
  All rights reserved. This program and the accompanying materials
  are meant solely for educative purposes.
  Author: Erik van Baasbank @erikbaas

  # Tips to improve resolution:
    # - As high as possible FPS (preferably slowmo)
    # - Set lines as far apart as possible
    # - As low speeds as possible
*******************************************************************
"""

import time
import numpy as np
import cv2
from IPython.display import Video
from matplotlib import pyplot as plt
import pandas as pd
import csv


# Todo: Every time you run, check the following parameters.
filename = "yaw0_trim_pass_1.mp4"
drone_label = filename.replace(".mp4", "")
print(drone_label)

fps = 480
distance_between_lines = 1.85              # m
x_first_line = 1428  # highest if from right to left
x_second_line = 627
np.random.seed(32)

# Constants and initialization info
passed_first = False
passed_second = False
f1 = 99999.                                 # Will later in the code be updated by a real value
f2 = 88888.                                 # idem
speed_avg = 9999.


# Routine to fix color to rgb
def fix_color(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# Routine to display image
def show_image(current_frame):
    # Draw the first and second line
    h, w, c = current_frame.shape
    cv2.line(current_frame, (x_first_line, int(0.1*h)), (x_first_line, int(0.9*h)), color=(0, 255, 0), thickness=2)
    cv2.line(current_frame, (x_second_line, int(0.1*h)), (x_second_line, int(0.9*h)), color=(0, 255, 0), thickness=2)

    # Matplotlib show the image
    plt.imshow(fix_color(current_frame))
    plt.show()


def find_background():
    # Import video stream
    cur_video_stream = cv2.VideoCapture(filename)
    # Randomly select 30 frames
    frame_ids = cur_video_stream.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=100)
    # Store selected frames in an array
    frames = []
    for fid in frame_ids:
        cur_video_stream.set(cv2.CAP_PROP_POS_FRAMES, fid)
        _, cur_frame = cur_video_stream.read()
        frames.append(cur_frame)
    cur_video_stream.release()

    # Take median frame (along time axis) and make it grey
    median_frame = np.median(frames, axis=0).astype(dtype=np.uint8)
    grayMedianFrame = cv2.cvtColor(median_frame, cv2.COLOR_BGR2GRAY)

    # show_image(grayMedianFrame)
    return grayMedianFrame


# Write the results to a csv file
def write_to_csv():
    header = ['filename', 'drone label', 'area', 'distance between lines', 'frames', 'time', 'speed (m/s)', 'associated pitch angle']
    data = [filename, drone_label, area, distance_between_lines, f2 - f1, (f2-f1)/fps, speed_avg, None]
    with open('data.csv', 'w', encoding='UTF8') as f:
        writer = csv.writer(f)

        # write the header
        writer.writerow(header)

        # write the data
        writer.writerow(data)


# ######## 1. Get the median image, greyed and blurred ##########
grayMedianFrame = find_background()

# ######## 2. Reopen the video stream and walk through it #######
video_stream = cv2.VideoCapture(filename)
total_frames = video_stream.get(cv2.CAP_PROP_FRAME_COUNT)
print("total frames: ", total_frames)

# Find how many pixels in width there are
ret, frame = video_stream.read()
h, w, c = frame.shape
x_prev = w

# Loop through frames
frameCnt = 0
while frameCnt < total_frames - 1:

    frameCnt += 1
    ret, frame = video_stream.read()

    # # Flip the image to rotate it:
    # cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
    # Convert current frame to grayscale
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate absolute difference of current frame and the median frame
    dframe = cv2.absdiff(gframe, grayMedianFrame)
    # Gaussian blur
    blurred = cv2.GaussianBlur(dframe, (11, 11), 0)
    # Thresholding to binarise (OTSU automatically determines a proper threshold)
    _, tframe = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Identifying contours from the threshold
    (cnts, _) = cv2.findContours(tframe.copy(),
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    # Within one frame, find all cnts and compute their areas and put them in a list
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        areas.append(area)

    # Rescan and only select the largest area
    for cnt in cnts:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if (40000 < area < 70000) and (x_prev-x < 200):  # Optional: change condition

            # Update x_prev
            x_prev = x

            # Draw a box around the found object
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

            # Draw a dot to indicate which point is evaluated
            cv2.drawMarker(frame, (x, y), (0, 255, 255), markerType=1, thickness=2)

            # # ALWAYS SHOW THE IMAGE TO TEST
            show_image(frame)
            print("Frame: ", frameCnt)
            print("Area size: ", area)

            # # When you pass the first line
            # if x < x_first_line and not passed_first:
            #     f1 = frameCnt
            #     print(f"Passed first line on the {f1}th frame")
            #
            #     show_image(frame)
            #
            #     # Set flag so you only execute this code once
            #     passed_first = True

            # # Show what's happening right before crossing the second line
            # if x < (x_second_line + 40):
            #     # Here we plot the last image
            #     print(f"The x loc is {x} and the to-pass x loc is {x_second_line}")
            #     show_image(frame)

            # # When you pass the second line
            # if x < x_second_line and not passed_second:
            #     f2 = frameCnt
            #     frames_delta = (f2-f1)
            #     time_delta = (frames_delta+1)/fps
            #     speed_avg = distance_between_lines/time_delta
            #
            #     # Here we plot the image of the drone passing the second image
            #     show_image(frame)
            #
            #     passed_second = True
            #
            #     # With 60fps:       20.5m           10m
            #     # f = 21 --> v = 105.43 km/h        51.4
            #     # f = 22 --> v = 100.64 km/h        49.1
            #     # f = 23 --> v = 96.26 km/h         47.0
            #
            #     print(f"Passed second line on the {f2}th frame")
            #     print(f"Time difference for area {area} is {f2 - f1} frames, which translates to {(f2-f1)/fps} sec")
            #     print(f"Assuming a distance of {distance_between_lines} m between lines, "
            #           f"v_avg = {round(speed_avg,3)} m/s, or {round(speed_avg*3.6, 3)} km/h")
            #
            #     write_to_csv()

    # writer.write(cv2.resize(frame, (640, 480)))

# Release video object
video_stream.release()
# writer.release()
