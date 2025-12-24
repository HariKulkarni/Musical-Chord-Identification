🎵 Musical Chord Identification Using Deep Learning

This project focuses on automatic musical chord identification from audio files using deep learning models. Audio signals are transformed into image-based representations and classified using multiple CNN architectures such as VGG19, ResNet50V2, InceptionV3, and MobileNetV2.

📁 Project Structure
├── Analysis/
│   └── (Model performance analysis, metrics, and results)
│
├── AudioFiles/
│   └── (Input audio files used for chord classification)
│
├── GUI/
│   └── (Graphical User Interface files for user interaction)
│
├── output/
│   └── (Predicted results, logs, and generated outputs)
│
├── MobileNetV2.ipynb
├── ResNet50v2 (1).ipynb
├── VGG19 (1).ipynb
├── inceptionV3_Upd.ipynb
│
└── README.md

🎯 Objective

To build an intelligent system that:

Accepts audio input

Converts audio into frequency-based image representations

Classifies musical chords using deep learning CNN models

Compares model performance to identify the best architecture

🧠 Models Used

VGG19

ResNet50V2

InceptionV3

MobileNetV2

Each model is implemented and trained separately using Jupyter Notebooks.

⚙️ Technologies & Libraries

Python

TensorFlow / Keras

NumPy

Pandas

Librosa

Matplotlib

Seaborn

Streamlit (for GUI, if applicable)

🔄 Workflow

Load audio files from AudioFiles/

Convert audio signals into image representations (spectrograms / DFT)

Train deep learning models on generated images

Evaluate models using accuracy, precision, recall, and F1-score

Display predictions and analysis results

📊 Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Model performance comparison is available in the Analysis/ folder.

🖥️ GUI Module

The GUI/ folder contains files that allow users to:

Upload audio files

View predicted chord output

Interact with the trained models easily

🚀 How to Run

Clone the repository

git clone <repository-url>


Install required dependencies

pip install -r requirements.txt


Open any model notebook (.ipynb) in Jupyter Notebook

Run all cells step by step

📌 Future Enhancements

Real-time chord detection

Improved accuracy with larger datasets

Full deployment using Streamlit

Support for more musical instruments

👤 Author

Harish H Kulkarni
📧 kulkarniharish4102000@gmail.com
