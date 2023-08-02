import cv2
import time
import random
import numpy as np

import mediapipe as mp


class CatchTheDot:

    def __init__(self, num_players=2, num_dots=1, dot_size=10, model_path = 'hand_landmarker.task', DEBUG=False) -> None:
        """Initialization of the CatchTheDot class.

        Args:
            num_players (int, optional): Number of players. Defaults to 2.
            num_dots (int, optional): Number of dots. Defaults to 1.
            dot_size (int, optional): Size of the dots. Defaults to 10.
            model_path (str, optional): Path to the hand landmarker model. Defaults to 'hand_landmarker.task'.
            DEBUG (bool, optional): DEBUG flag. Defaults to False.
        """

        self.DEBUG = DEBUG

        self.num_players = num_players
        self.num_dots = num_dots
        self.dot_size = dot_size

        self.model_setup(model_path=model_path)

    def model_setup(self, model_path: str) -> None:
        """Set up the hand landmarker model.

        Args:
            model_path (str): Path to the hand landmarker model
        """

        self.options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
            num_hands=self.num_players,
            result_callback=self.hand_landmarks_detected)
    
    def start(self) -> None:
        """Start the game. Reset all the dots, fingers and scores.
        """

        self.dot_coo = []
        self.finger_coo = []
        self.score = [0 for _ in range(self.num_players)]

        self.run()
    
    def run(self) -> None:
        """Run the gameloop.
        """

        HandLandmarker = mp.tasks.vision.HandLandmarker
        with HandLandmarker.create_from_options(self.options) as landmarker:

            cap = cv2.VideoCapture(0)
            self.frame_resolution = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            start_timestamp = time.time()

            while True:

                _, frame = cap.read()
                frame = cv2.flip(frame, 1) # Flip frame for mirror effect
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

                timestamp_ms = round((time.time() - start_timestamp)*1000)
                landmarker.detect_async(mp_image, timestamp_ms)

                self.game_mechanics()
                
                frame = self.write_text(frame)
                cv2.imshow('Catch The Dot' , frame)

                if cv2.waitKey(1) == ord('q'): # Press Q to quit the game
                    break
                    
            cap.release()
            cv2.destroyAllWindows()
    
    def game_mechanics(self) -> None:
        """Check if finger collides with a dot. Regenerate dots if necessary.
        """

        for finger_index, finger in enumerate(self.finger_coo):
            for dot in reversed(self.dot_coo):

                if abs(finger['x'] - dot['x']) < self.dot_size and abs(finger['y'] - dot['y']) < self.dot_size:
                    # Finger collides with dot
                    self.score[finger_index] += 1
                    self.dot_coo.remove(dot)
        
        # Regenerate the removed dots
        while len(self.dot_coo) < self.num_dots:
            self.dot_coo.append({
                'x': random.randint(100, self.frame_resolution[0]-100),
                'y': random.randint(100, self.frame_resolution[1]-200),
            })
    
    def write_text(self, frame: np.ndarray) -> np.ndarray:
        """Write the scores and draw the dots on top of the frame.

        Args:
            frame (np.ndarray): Frame from the webcam

        Returns:
            np.ndarray: Frame from the webcam with the text and dots drawn on top
        """

        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (20, 50)
        fontScale = 1
        color = (255, 255, 255)
        thickness = 1

        # Write the scores
        for index, score in enumerate(self.score):
            frame = cv2.putText(frame, f'Score player {index}: {score}', org, font, fontScale, color, thickness, cv2.LINE_AA)
            org = (20, org[1] + 50)

        # Draw the dots
        for dot in self.dot_coo:
            frame = cv2.circle(frame, (dot['x'], dot['y']), self.dot_size, (0, 0, 255), -1)
        
        if self.DEBUG:
            # Draw dots on the index finger tips for debugging
            for finger in self.finger_coo:
                frame = cv2.circle(frame, (finger['x'], finger['y']), self.dot_size, (0, 255, 0), -1)

        return frame

    def hand_landmarks_detected(self, result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int) -> None:
        """Callback function from the hand landmark model. Sets the finger coordinates.

        Args:
            result (mp.tasks.vision.HandLandmarkerResult): Hand landmarker result from the hand landmark model
            output_image (mp.Image): Frame on which the detection took place
            timestamp_ms (int): Timestamp in milliseconds of when the detection took place
        """

        INDEX_FINGER = 8

        self.finger_coo = [] # Clear previous finger coordinates

        for hand_landmark in result.hand_landmarks:
            # Append detected finger coordinates in pixel coordinates
            self.finger_coo.append(dict({
                'x': round(hand_landmark[INDEX_FINGER].x * self.frame_resolution[0]),
                'y': round(hand_landmark[INDEX_FINGER].y * self.frame_resolution[1]),
            }))


if __name__ == "__main__":

    CatchTheDot().start()
