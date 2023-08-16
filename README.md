# Face Mask Detection

## Introduction
The "Face Mask Detection" project is an application designed to detect and recognize individuals not wearing face masks in images, videos, and real-time webcam streams. This is a problem that can be applied in various fields such as healthcare, security, public surveillance, and many other applications.

## Features
- **Image and Video Classification**: The project allows uploading images or videos, as well as utilizing webcam feeds, containing individuals. It will classify whether they are wearing face masks or not and process the information in real-time.
- **High Accuracy**: The project employs machine learning models to ensure high accuracy in detecting the presence of face masks.
- **Sending Warning Alerts via Telegram Chatbot**: The system can send alerts through a Telegram chatbot when individuals are detected not wearing face masks.
- **Alert Sound**: An alert sound can be triggered when individuals without face masks are detected.

## Installation
1. **Prerequisites**: Make sure you have Python 3.8 or higher installed, along with the necessary libraries. You can use pip to install them:
    ```bash
    pip install -r requirements.txt
    ```

2. **Clone the Repository**: Clone the project from GitHub using the following command:
    ```
    git clone https://github.com/minh1311/PMMNM_FaceMask.git
    ```

3. **Run the Application**: Run the application using the following command:
    ```bash
    python detect.py --weights D:\DeepLearning\yolov5\runs\train\exp13\weights\best.pt --source 0
    ```
    or
   ```bash
    python main.py
    ```

## Contribution
We welcome contributions to the project from the community. If you wish to provide feedback, report issues, or submit new feature requests, you can follow these steps:
1. Fork the project and create a new branch for your work.
2. Implement your changes and add new commits.
3. Create a pull request to the main branch of the project.
<img width="100%" src="[https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/yolo-comparison-plots.png](https://github.com/minh1311/PMMNM_FaceMask/blob/master/1.jpg)"></a>
## Authors
This project was implemented by Nguyen Hoang Minh, Nguyen Thi Thu Thuy, Hoang Thi Sao Mai.

## Contact
If you have any questions, suggestions, or proposals, please contact us via email at: minhnguyeny2002@gmail.com.
