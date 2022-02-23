# Hand Gesture Recognition

Hand gesture recognition using mediapipe. You can download folder "**gesture recognition model**" and test latest model by yourself. Main gestures:

## Fire
![Image alt](https://github.com/GargoDan/HandGestureRecognition/imgs/fire.jpg)

## Air
![Image alt](ttps://github.com/GargoDan/HandGestureRecognition/imgs/air.jpg)

## Earth
![Image alt](ttps://github.com/GargoDan/HandGestureRecognition/imgs/earth.jpg)

## Water
![Image alt](ttps://github.com/GargoDan/HandGestureRecognition/imgs/water.jpg)

# Descriotion
Firstly, main idea was to recognize dynamic gestures, so "**collect_data.py**" collect short videos (30 frames) of different gestures. After collecting data and testing different models (example of lstm you can find here: "**lstm.ipynb**"), it was decided to recognize static gestures because of big varaity of dynamic representation of gestures. Following this, all videos was parsed to images in "**video_frames.ipynb**" and converted to mediapipe features in "**images_to_mp.ipynb**". File "**catboost.ipynb**" is needed to training the catboost model and "**test_model.ipynb**" needed to test the model.

The next step will be add more trash data for decrease fake number of missclassifications. After this, speed up the main model. Final step will be to deploy the model on IoS.