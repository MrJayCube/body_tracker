import numpy as np
from typing import Tuple, Union
import math
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
class headTracker():
    def __init__(self):
        model_fileHead = open(r"./blaze_face_short_range.tflite", "rb")
        model_dataHead = model_fileHead.read()
        model_fileHead.close()
        base_optionsHead = python.BaseOptions(model_asset_buffer=model_dataHead)
        optionsHead = vision.FaceDetectorOptions(base_options=base_optionsHead)
        self.detectorHead = vision.FaceDetector.create_from_options(optionsHead)

    def _normalized_to_pixel_coordinates(
            self,
            normalized_x: float, normalized_y: float, image_width: int,
            image_height: int) -> Union[None, Tuple[int, int]]:
        """Converts normalized value pair to pixel coordinates."""

        # Checks if the float value is between 0 and 1.
        def is_valid_normalized_value(value: float) -> bool:
            return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                              math.isclose(1, value))

        if not (is_valid_normalized_value(normalized_x) and
                is_valid_normalized_value(normalized_y)):
            # TODO: Draw coordinates even if it's outside of the image bounds.
            return None
        x_px = min(math.floor(normalized_x * image_width), image_width - 1)
        y_px = min(math.floor(normalized_y * image_height), image_height - 1)
        return x_px, y_px
    def visualize(
            self,
            image,
            detection_result
    ) -> np.ndarray:
        """Draws bounding boxes and keypoints on the input image and return it.
        Args:
          image: The input RGB image.
          detection_result: The list of all "Detection" entities to be visualize.
        Returns:
          Image with bounding boxes.
        """
        annotated_image = image.copy()
        height, width, _ = image.shape

        #### Head params #####
        MARGIN = 10  # pixels
        ROW_SIZE = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        TEXT_COLOR = (255, 0, 0)  # red

        for detection in detection_result.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

            # Draw keypoints
            for keypoint in detection.keypoints:
                keypoint_px = self._normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                               width, height)
                color, thickness, radius = (0, 255, 0), 2, 2
                cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            category_name = '' if category_name is None else category_name
            probability = round(category.score, 2)
            result_text = category_name + ' (' + str(probability) + ')'
            text_location = (MARGIN + bbox.origin_x,
                             MARGIN + ROW_SIZE + bbox.origin_y)
            cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                        FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        return annotated_image
    def headDetector(self, image):
        image_mp = mp.Image(mp.ImageFormat.SRGB, data=image)
        detection_result = self.detectorHead.detect(image_mp)
        image = self.visualize(image, detection_result)
        return image

class gestureTracker():
    def __init__(self):
        with open(r'./gesture_recognizer.task', 'rb') as f:
            model = f.read()

        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        optionsGestures = vision.GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_buffer=model),
            running_mode=VisionRunningMode.IMAGE, )
        self.recognizerGestures = vision.GestureRecognizer.create_from_options(optionsGestures)

    def gestureDetector(self, image):
        image_mp = mp.Image(mp.ImageFormat.SRGB, data=image)
        recognition_result = self.recognizerGestures.recognize(image_mp)
        if len(recognition_result.gestures) > 0 and recognition_result.gestures[0][0].category_name != "None":
            top_gesture = recognition_result.gestures
            hand_landmarks = recognition_result.hand_landmarks
            print((top_gesture, hand_landmarks))
            cv2.putText(image, str(top_gesture[0][0].category_name), (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 0, 0), 2, cv2.LINE_AA)
        return image
class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5,modelComplexity=1,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)


        #rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        return image

    def positionFinder(self,image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h,w,c = image.shape
                cx,cy = int(lm.x*w), int(lm.y*h)
                lmlist.append([id,cx,cy])
            if draw:
                cv2.circle(image,(cx,cy), 15 , (255,0,255), cv2.FILLED)

        return lmlist

def main():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()
    head_tracker = headTracker()
    gesture_tracker = gestureTracker()

    while True:
        success,image = cap.read()
        image = gesture_tracker.gestureDetector(image)
        image = tracker.handsFinder(image)
        lmList = tracker.positionFinder(image)
        image = head_tracker.headDetector(image)
        if len(lmList) != 0:
            #print(lmList[4])
            pass

        cv2.imshow("Video", image)
        cv2.waitKey(1)
if __name__ == "__main__":
    main()
