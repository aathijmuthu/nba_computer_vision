# NBA Player Detection/Tracking and Analysis

## Introduction

The goal of this project is to detect and track players, referees, and basketballs in a video using YOLO, an AI object detection model. Additionally, we will assign players to teams based on the colors of their t-shirts using Kmeans for pixel segmentation and clustering. With this information, I measured a team's ball acquisition percentage in a match. I also used optical flow to measure camera movement between frames, to accurately measure a player's movement. Furthermore, I implemented perspective transformation to represent the scene's depth and perspective, which allows me to measure a player's movement in meters rather than pixels. Finally, I calculated a player's speed and the distance covered.

## Training

The YOLOv8 model was trained on a [RoboFlow Dataset](https://universe.roboflow.com/asas-annotations/ai-sports-analytics-system/dataset/7)

<img width="1183" alt="image" src="https://github.com/aathijmuthu/nba_computer_vision/assets/65832367/a7fc37c8-ff84-47f3-9336-f2e747f829de">
