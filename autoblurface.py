import os

from ultralytics import YOLO
import cv2
import time


class AutoBlurFace:
    """
       This class provides functionality to automatically detect faces in a video stream
       and apply a blurring effect to them.
       """

    def __init__(self, model_path, show_video=False, save_video=False, apply_blur=False, enlargement_factor=10,
                 output_video_path='output.mp4'):
        """
        Initializes the AutoBlurFace class with the specified parameters.

        Parameters:
            model_path (str): Path to the YOLO model file.
            output_video_path (str): Path for saving the output video.
            enlargement_factor (int): Factor to enlarge the detected face area.
            show_video (bool): Flag to display the video during processing.
            save_video (bool): Flag to save the processed video.
            apply_blur (bool): Flag to apply blurring effect on detected faces.
        """

        self.model = self._load_model(model_path)
        self.output_video_path = output_video_path
        self.enlargement_factor = enlargement_factor
        self.show_video = show_video
        self.save_video = save_video
        self.apply_blur = apply_blur
        self.video_writer = None

    def _load_model(self, model_path):
        """
        Loads the YOLO model from the specified path.

        Parameters:
            model_path (str): Path to the YOLO model file.

        Returns:
            YOLO: Loaded YOLO model.
        """
        try:
            model = YOLO(model_path)
            print('YoloFaceModel loaded successfully.')
            return model
        except Exception as e:
            print(f'Failed to load the model: {e}')
            return None

    def _apply_blur_to_faces(self, frame, detections):
        """
        Applies blurring to areas around the detected faces, handling image edges.

        Parameters:
            frame (numpy.ndarray): The current video frame.
            detections: Detected objects in the frame.

        Returns:
            numpy.ndarray: The frame with blurred faces.
        """
        height, width = frame.shape[:2]
        for detection in detections:
            for xyxy in detection.boxes.xyxy:
                x1, y1, x2, y2 = map(int, xyxy)
                width_enlargement = int((x2 - x1) * self.enlargement_factor / 100)
                height_enlargement = int((y2 - y1) * self.enlargement_factor / 100)

                x1 = max(0, x1 - width_enlargement)
                y1 = max(0, y1 - height_enlargement)
                x2 = min(width, x2 + width_enlargement)
                y2 = min(height, y2 + height_enlargement)

                face_region = frame[y1:y2, x1:x2]
                blurred_face = cv2.GaussianBlur(face_region, (99, 99), 30)
                frame[y1:y2, x1:x2] = blurred_face

        return frame

    def _process_frame(self, frame):
        """
        Processes a single frame to detect faces and apply blurring.

        Parameters:
            frame (numpy.ndarray): The current video frame.

        Returns:
            numpy.ndarray: The processed frame.
        """
        detections = self.model(frame)
        if self.apply_blur:
            frame = self._apply_blur_to_faces(frame, detections)
        return frame

    def run(self, video_path, use_webcam=False):
        """
        Runs the face detection and blurring process on a video file or webcam feed.

        Parameters:
            video_path (str): Path to the input video file.
            use_webcam (bool): Flag to use webcam as input instead of a file.
        """
        if use_webcam:
            video = cv2.VideoCapture(0)
        else:
            video = cv2.VideoCapture(video_path)

        if not video.isOpened():
            raise Exception('Failed to load the video or webcam.')

        if self.save_video:
            frame_rate = int(video.get(cv2.CAP_PROP_FPS))
            width, height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            self.video_writer = cv2.VideoWriter(self.output_video_path, fourcc, frame_rate, (width, height))

        start_time = time.time()
        frame_count = 0

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            processed_frame = self._process_frame(frame)
            if self.show_video:
                cv2.imshow('Processed Frame', cv2.resize(processed_frame, (1280, 720)))

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if self.save_video:
                self.video_writer.write(processed_frame)

            frame_count += 1

        elapsed_time = time.time() - start_time
        print(
            f'Processed {frame_count} frames in {int(elapsed_time // 60)} minutes and {int(elapsed_time % 60)} seconds.')

        if self.save_video and self.video_writer is not None:
            self.video_writer.release()
        video.release()
        cv2.destroyAllWindows()


# Usage example
auto_blur_face = AutoBlurFace('yolov8n-face.pt', output_video_path='face-video-blurred.mp4', show_video=False, save_video=True,
                              apply_blur=True)
auto_blur_face.run('face-video.mp4', use_webcam=False)
