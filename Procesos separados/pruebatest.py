import face_recognition
import cv2
import numpy as np
import os
import argparse
import imutils
import pickle
import dlib

dlib.DLIB_USE_CUDA = True

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--encodings", required=True,
    help="path to serialized db of facial encodings")
ap.add_argument("-o", "--option", default=0,
    help="")
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
if args["option"] == 1:
    video_captures = [cv2.VideoCapture(0), cv2.VideoCapture(2)]
else:
    video_captures = [cv2.VideoCapture(args["option"])]


def getColor(c):
    switcher = {
        0: (0, 0, 255),
        1: (92, 234, 53),
        2: (228, 236, 62),
        3: (26, 182, 133),
        4: (47, 95, 205),
        5: (165, 47, 295),
        6: (205, 47, 157),
        7: (138, 12, 12)
    }

    return switcher.get(c)

def getSizeText(w):
    return (w* 1.0 )/ 200


# Create arrays of known face encodings and their names
known_face_encodings = data["encodings"]
known_face_names = data["names"]


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
face_prob = []
process_this_frame = True


while True:

    frameList = []

    for i in video_captures:

        # Grab a single frame of video
        ret, frame = i.read()

        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = small_frame[:, :, ::-1]

        # Only process every other frame of video to save time
        if process_this_frame:
            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                prob = 0.0

                # # If a match was found in known_face_encodings, just use the first one.
                # if True in matches:
                #     first_match_index = matches.index(True)
                #     name = known_face_names[first_match_index]

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                    #prob = float(round(prob / len(face_distances),2))
                    prob = float(round((100-best_match_index),2))

                face_names.append(name)
                face_prob.append(prob)

        #process_this_frame = not process_this_frame

        # Display the results
        index = 0
        for (top, right, bottom, left), name, prob in zip(face_locations, face_names, face_prob):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            top -= 28
            left -= 0
            bottom += 28
            right += 0


            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), getColor(index), 2)

            sizeText = getSizeText(right-left)

            if name == "Unknown":
                texto = str(name)
            else:
                texto = str(name)+' '+str(prob)

            # Draw a label with a name below the face
            #cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, texto, (left + 1, bottom - 1), font, sizeText, getColor(index), 2) #shadow
            cv2.putText(frame, texto, (left + 3, bottom - 3), font, sizeText, getColor(index), 2) #shadow
            cv2.putText(frame, texto, (left + 2, bottom - 2), font, sizeText, (255, 255, 255), 1)

            index +=1
            if index == 8:
                index = 0

        face_locations.clear()
        face_names.clear()
        face_locations.clear()
        face_encodings.clear()

        #cv2.imshow('Video {}'.format(idx), frame)
        frameList.append(frame)

    process_this_frame = not process_this_frame

    if "option"==1:
        numpy_vertical_concat = np.concatenate((frameList[0], frameList[1]), axis=1)
        cv2.imshow('Video', numpy_vertical_concat)
    else:
        cv2.imshow('Video', frameList[0])

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
for video_capture in video_captures:
    video_capture.release()
cv2.destroyAllWindows()
