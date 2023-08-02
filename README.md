# Catch The Dot
Catch The Dot game created for an Ordina case by Arno Van Eetvelde.

## Game
This is a game played with your webcam. You have to 'touch' as many dots as possible with the tip of your index finger.
Each time you 'touch' a dot with your index finger the touched dot disappears and a new one shows up. The game can be played individually or against multiple people.

You can set the numer of players, number of dots and dot size in the CatchTheDot class.

For the detection and tracking of the index finger the hand landmarks model is used from [MediaPipe](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker).