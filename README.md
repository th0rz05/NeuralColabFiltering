# Neural Collaborative Filtering Recommendation System

This repository contains an end-to-end implementation of a recommendation system for an online casino using Neural Collaborative Filtering (NCF). The system leverages both offline and real-time recommendation strategies and includes a detailed production pipeline.

## Overview

The project includes:

- **Model Training:** A neural network implemented in PyTorch for learning user-slot interactions with a ranking-based loss (BPR Loss).
- **Recommendation Serving:** Functions to generate offline recommendations, hybrid recommendations (combining model predictions with recent user activity), and real-time recommendations using a FAISS index.
- **Data Preparation:** Notebooks to preprocess and prepare your data.
- **Production Pipeline:** A document detailing the pipeline and best practices for daily operations, real-time serving, and handling new users/games.

## Repository Structure

The project is organized as follows:

```plaintext
├── NeuralCollabFiltering.ipynb       # Final Jupyter Notebook for the recommendation system
├── Pipeline.md                       # Detailed production pipeline guidelines
├── data.csv                          # Sample dataset for training and evaluation
├── data_preparation.ipynb            # Notebook for data cleaning and preparation
├── main.py                           # Production-ready script for training and serving recommendations
├── model.py                          # Module containing the NCFModel, loss function, and Trainer class
├── recommender.py                    # Module containing the Recommender class and utility functions
└── slot_game_data.csv                # Additional dataset (if applicable)
```

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/your_username/NeuralCollabFiltering.git
   cd NeuralCollabFiltering
   ```

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training and Serving Recommendations

- **Training the Model:**
  Run the `main.py` script to load data, train the model with early stopping, extract embeddings, and build the FAISS index.

  ```bash
  python main.py
  ```

  The script will output offline recommendations (both model-only and hybrid) as well as real-time recommendations based on the FAISS index.

- **Exploring Notebooks:**
  - `NeuralCollabFiltering.ipynb` – Contains a complete walkthrough of the model training and recommendation process.
  - `data_preparation.ipynb` – Provides steps for cleaning and preparing the dataset.

## Production Pipeline

Refer to the [Pipeline.md](Pipeline.md) document for detailed guidelines on:

- Daily data collection and preprocessing.
- Model retraining, validation, and early stopping.
- Real-time recommendation serving using FAISS.
- Handling new users and new games.
- Monitoring, evaluation, and continuous improvement.

## Future Improvements

- **Hyperparameter Tuning:** Experiment with different model architectures, loss functions, and training parameters.
- **Enhanced Data Signals:** Integrate additional features (e.g., session duration, recency) for better personalization.
- **Scalable Deployment:** Refactor into microservices, implement API endpoints, and use containerization (Docker/Kubernetes) for production deployment.
- **User Feedback Loop:** Incorporate user feedback and A/B testing to refine recommendations further.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any questions or contributions, please contact [tiago.filipe.barbosa@gmail.com](mailto:tiago.filipe.barbosa@gmail.com).
