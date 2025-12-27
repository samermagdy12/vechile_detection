# Vehicle Detection and Tracking Application

## 🎯 Project Goal
The primary goal of this project is to provide a robust, real-time solution for vehicle detection and tracking in video streams and static images. This application is designed to be user-friendly, leveraging a modern web interface for easy interaction and visualization of results.

## ✨ Features
*   **Real-Time Detection:** Utilizes the YOLOv11 model for high-accuracy, real-time object detection.
*   **User Interface:** Built with Streamlit for a clean, modern, and interactive web application experience.
*   **Multi-Media Support:** Supports detection on both video files and static images.
*   **Modern Design:** Features a custom dark UI theme for improved aesthetics and user experience.

## 🛠️ Tech Stack
| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Model** | YOLOv11 | State-of-the-art object detection model. |
| **Frontend** | Streamlit | Web application framework for data science projects. |
| **Backend** | Python | Core programming language. |
| **Dependencies** | `requirements.txt` | Manages all necessary Python packages. |

## 🚀 Setup and Run Instructions

### Prerequisites
*   Python 3.8+

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/samermagdy12/vehicle_detection.git
    cd vehicle_detection
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application
Execute the following command to start the Streamlit web application:
```bash
streamlit run app.py
```
The application will open in your default web browser, typically at `http://localhost:8501`.

## 💡 Example Usage
Once the application is running, you can:
1.  Upload a video file or an image through the sidebar.
2.  The application will process the media, displaying bounding boxes and confidence scores for all detected vehicles.
3.  The results can be viewed directly in the web interface, demonstrating the model's performance in real-time.

## 📁 Project Structure
```
vehicle_detection/
├── app.py              # Main Streamlit application file
├── best_2.pt           # Pre-trained YOLOv11 model weights
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation (this file)
```
