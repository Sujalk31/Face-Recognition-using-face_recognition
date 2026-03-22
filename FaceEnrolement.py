"""
STEP 1 - FACE ENROLLMENT
========================
Captures your photo via webcam, extracts the face encoding,
and saves it to a 'face_database.npz' file.

Usage:
    python FaceEnrolement.py --name "YourName"

Requirements:
    pip install face_recognition opencv-python numpy
"""

import cv2
import face_recognition
import numpy as np
import argparse
import os
import sys

DATABASE_FILE = "face_database.npz"


def enroll_person(name: str, num_samples: int = 5):
    """Capture face from webcam, compute encoding, save to NPZ database."""

    print(f"\n[INFO] Enrolling: {name}")
    print("[INFO] Press SPACE to capture a sample, Q to quit early.\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        sys.exit("[ERROR] Cannot open webcam. Check your camera connection.")

    collected_encodings = []

    while len(collected_encodings) < num_samples:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame, retrying...")
            continue

        display = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect faces in current frame
        face_locations = face_recognition.face_locations(rgb)

        for top, right, bottom, left in face_locations:
            cv2.rectangle(display, (left, top), (right, bottom), (0, 255, 0), 2)

        status = f"Samples: {len(collected_encodings)}/{num_samples} | SPACE=capture  Q=quit"
        cv2.putText(display, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        cv2.imshow("Enrollment - " + name, display)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[INFO] Enrollment cancelled by user.")
            break

        if key == ord(' '):
            if not face_locations:
                print("[WARN] No face detected in frame. Try again.")
                continue

            if len(face_locations) > 1:
                print("[WARN] Multiple faces detected. Please be alone in frame.")
                continue

            encodings = face_recognition.face_encodings(rgb, face_locations)
            if encodings:
                collected_encodings.append(encodings[0])
                print(f"  [OK] Sample {len(collected_encodings)} captured.")

    cap.release()
    cv2.destroyAllWindows()

    if not collected_encodings:
        sys.exit("[ERROR] No face samples collected. Enrollment failed.")

    # Average all samples for a robust encoding
    mean_encoding = np.mean(collected_encodings, axis=0)

    # Load existing database if it exists
    names = []
    encodings = []

    if os.path.exists(DATABASE_FILE):
        data = np.load(DATABASE_FILE, allow_pickle=True)
        names = list(data["names"])
        encodings = list(data["encodings"])
        print(f"[INFO] Loaded existing database with {len(names)} person(s).")

    # Check if name already exists → update
    if name in names:
        idx = names.index(name)
        encodings[idx] = mean_encoding
        print(f"[INFO] Updated existing entry for '{name}'.")
    else:
        names.append(name)
        encodings.append(mean_encoding)
        print(f"[INFO] Added new entry for '{name}'.")

    np.savez_compressed(
        DATABASE_FILE,
        names=np.array(names),
        encodings=np.array(encodings)
    )

    print(f"\n[SUCCESS] Database saved to '{DATABASE_FILE}'")
    print(f"          Total persons enrolled: {len(names)}")
    for n in names:
        print(f"            • {n}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll a face into the recognition database.")
    parser.add_argument("--name", required=True, help="Name of the person to enroll")
    parser.add_argument("--samples", type=int, default=5,
                        help="Number of photo samples to capture (default: 5)")
    args = parser.parse_args()

    enroll_person(args.name, args.samples)