**🎬 Movie Genre Classification Using Posters and Subtitles
📖 Overview**

This project implements a hybrid deep learning approach for movie genre classification, inspired by the paper “Movie Genre Classification Based on Poster and Subtitles Using Hybrid Combination of CNN.”
The model leverages visual features from movie posters and textual features from subtitles to predict genres more accurately.

**🚀 Features**

Extracts visual features from posters using Convolutional Neural Networks (CNNs).

Processes subtitle text data using text preprocessing and embedding techniques (e.g., Word2Vec / TF-IDF).

Combines visual and textual features for hybrid genre classification.

Provides evaluation metrics such as Accuracy, Precision, Recall, and F1 Score.

**🗂️ Dataset**

The dataset includes:

Poster images — movie promotional posters (poster_path)

Subtitle files — .srt subtitle text files (subtitle_path)

Genre labels — target output classes (genre)

**📁 Example dataset structure:**

dataset/
│
├── posters/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   └── ...
│
├── subtitles/
│   ├── movie1.srt
│   ├── movie2.srt
│   └── ...
│
└── final_dataset.csv

**⚙️ Installation**

Clone this repository:

git clone https://github.com/Pushpa-surapureddy/movie-genre-classification.git
cd movie-genre-classification


Install required dependencies:

pip install tensorflow==2.10.0 scikit-learn pandas numpy pillow opencv-python gensim nltk tqdm


(Optional) If running in Google Colab, mount Google Drive to access dataset.

**🧠 Model Architecture**

Poster Module: CNN (Convolutional Neural Network) extracts image features.

Subtitle Module: Text preprocessing → Tokenization → Embedding → Dense Layers.

Fusion Layer: Combines both feature sets (image + text).

Output Layer: Fully connected softmax layer for genre classification.

**📊 Architecture Overview:**

Poster (CNN) ─┐
               ├── Concatenate ──> Dense ──> Output (Genre)
Subtitles (Text Embedding) ─┘

**🧾 Performance Metrics**

The model reports:

Accuracy

Precision

Recall

F1 Score (Macro / Weighted)

Example:

Accuracy: 85%
F1 Score (Macro): 0.82

🧩 Usage

Run the notebook to train and evaluate the model:

movie_gener_code2.ipynb


To predict genres for a new movie:

model.predict([poster_image, subtitle_text])

**📈 Results**

Hybrid model improves classification accuracy compared to single-modality models.

Shows better F1 performance when combining both visual and textual features.

**🔍 Future Enhancements**

Include audio features (spectrograms or MFCCs).

Extend dataset with movie plots and trailers.

Deploy as a web application with Flask or Streamlit.

**📚 References**

Movie Genre Classification Based on Poster and Subtitles Using Hybrid Combination of CNN

OpenSubtitles Dataset

IMDb Posters and Metadata

**👩‍💻 Author**

Puspa
Movie Genre Classification Project — 2025
Feel free to fork, improve, or cite this repository.
