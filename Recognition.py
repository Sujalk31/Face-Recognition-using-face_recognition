#python Output.py --input video.mp4 --output result.mp4

import cv2
import face_recognition
import numpy as np
import argparse
import os
import sys
import time

DATABASE_FILE = "face_database.npz"

TOLERANCE = 0.50
PROCESS_EVERY_N = 2
SCALE = 0.5


def load_database(path):

    if not os.path.exists(path):
        sys.exit("Face database not found. Run enrollment first.")

    data = np.load(path, allow_pickle=True)
    names = list(data["names"])
    encodings = list(data["encodings"])

    print("Database loaded:", names)
    return names, encodings


def run_inference(video_path, output_path):

    known_names, known_encodings = load_database(DATABASE_FILE)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        sys.exit("Cannot open video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0
    last_results = []

    start = time.time()

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1

        if frame_number % PROCESS_EVERY_N == 0:

            small = cv2.resize(frame, (0, 0), fx=SCALE, fy=SCALE)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, face_locations)

            last_results = []

            for loc, encoding in zip(face_locations, encodings):

                distances = face_recognition.face_distance(known_encodings, encoding)

                name = "Unknown"
                confidence = 0

                if len(distances) > 0:

                    best_match = np.argmin(distances)

                    if distances[best_match] <= TOLERANCE:
                        name = known_names[best_match]
                        confidence = (1 - distances[best_match]) * 100

                top, right, bottom, left = [int(v / SCALE) for v in loc]

                last_results.append((top, right, bottom, left, name, confidence))

        for (top, right, bottom, left, name, confidence) in last_results:

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

            label = f"{name} {confidence:.0f}%" if name != "Unknown" else "Unknown"

            cv2.putText(
                frame,
                label,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        out.write(frame)

    cap.release()
    out.release()

    print("Output saved:", output_path)
    print("Time taken:", round(time.time() - start, 2), "seconds")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="result.mp4", help="Output video")

    args = parser.parse_args()

    run_inference(args.input, args.output)