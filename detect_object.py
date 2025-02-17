import json
import os
import shutil
import sys
from logging import config, getLogger

import cv2
from ultralytics import YOLO


def init_logger():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_config_path = os.path.join(script_dir, 'log_config.json')
    with open(log_config_path, 'r') as f:
        log_conf = json.load(f)

    config.dictConfig(log_conf)


def process_videos_in_directory(directory_path, destination_dir_path,
                                object_name, model, logger):
    logger.info(f"check video in {directory_path}")
    os.makedirs(destination_dir_path, exist_ok=True)
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            lower_name = file.lower()
            if lower_name.endswith('_r.mp4'):
                video_path = os.path.join(root, file)
                if detect_object_in_video(video_path, object_name,
                                          model, logger):
                    dest_path = os.path.join(
                        destination_dir_path, lower_name)
                    logger.debug(
                        f"Copying video: {video_path} to {dest_path}")
                    if os.path.exists(dest_path):
                        raise Exception(f"File already exists: {dest_path}")
                    shutil.copy(video_path, dest_path)

                    front_file_path = os.path.join(
                        root, file.replace('_R.MP4', '_F.MP4'))
                    front_file_path = front_file_path.replace(
                        '_r.mp4', '_f.mp4')
                    new_file_name = lower_name.replace('_r.mp4', '_f.mp4')

                    dest_path = os.path.join(
                        destination_dir_path, new_file_name)
                    logger.debug(
                        f"Copying video: {front_file_path} to {dest_path}")
                    if os.path.exists(dest_path):
                        raise Exception(f"File already exists: {dest_path}")
                    shutil.copy(front_file_path, dest_path)


def detect_object_in_video(video_path, object_name, model, logger):
    logger.debug(f"Processing video: {video_path}")
    # Open video file
    object_detected = False

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)
        boxes = results[0].boxes
        for box, cls_idx, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            class_name = results[0].names[cls_idx.item()]
            if class_name == object_name:
                width = box[2].item() - box[0].item()
                height = box[3].item() - box[1].item()
                if width >= 300 or height >= 300:
                    object_detected = True
                    break

        if object_detected:
            break

    cap.release()

    if object_detected:
        logger.debug(
            f"Object({object_name}) is detected in video: {video_path}")
    return object_detected


if __name__ == "__main__":
    init_logger()
    logger = getLogger(__name__)
    # Load a pretrained model
    model = YOLO("yolo11s.pt")
    # model = YOLO("yolo11m.pt")

    if len(sys.argv) < 3:
        print("usage detect_object.py <video_dir_path> <destination_dir_path>")
    else:
        try:
            process_videos_in_directory(
                sys.argv[1], sys.argv[2], "person", model, logger)
        except Exception as e:
            logger.error(f"An error occurred: {e}")
