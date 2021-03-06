import face_recognition
import cv2
import numpy as np
import os
import argparse
import imutils
import pickle
import math

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
args = vars(ap.parse_args())

data = pickle.loads(open(args["encodings"], "rb").read())

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)


# Create arrays of known face encodings and their names
known_face_encodings = data["encodings"]
known_face_names = data["names"]


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

num_frame = 0
modulus_processing_frame = 1
latest_recognitions = []

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]


    # Only process every other frame of video to save time
    if num_frame % modulus_processing_frame == 0:
        # Find all the faces and face encodings in the current frame of video
        face_names = []
        face_locations = face_recognition.face_locations(rgb_small_frame)
        for (top, right, bottom, left) in face_locations:
            was_it_before = False
            name = "Intruso"
            for (old_location_center, old_frame, old_name) in latest_recognitions[:]:
                was_it_before = abs(old_location_center[0] - (right + left)/2) < 15 and abs(old_location_center[1] - (top + bottom)/2) < 15 and old_frame + 10 >= num_frame
                if was_it_before:
                    name = old_name
                    latest_recognitions.remove((old_location_center, old_frame, old_name))
                    latest_recognitions.append((((right + left)/2, (top + bottom)/2), num_frame, name))
                    break
                
            if (not(was_it_before)):
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    latest_recognitions.append((((right + left)/2, (top + bottom)/2), num_frame, name))

            face_names.append(name)
        for (old_location_center, old_frame, old_name) in latest_recognitions[:]:
            if (old_frame + 10 < num_frame):
                latest_recognitions.remove((old_location_center, old_frame, old_name))


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        top -= 25
        left -= 25
        bottom += 45
        right += 25

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    # Display the resulting image
    cv2.imshow('Video', frame)

    num_frame = num_frame + 1

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()