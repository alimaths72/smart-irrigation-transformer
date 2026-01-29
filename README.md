# Smart Irrigation Requirement Prediction Using Machine Learning

## Abstract / Overview
This project focuses on predicting irrigation requirements using machine learning techniques based on environmental and soil-related parameters. The aim is to support efficient water management in agriculture by providing data-driven irrigation recommendations. A baseline Logistic Regression model is compared with a Transformer-based deep learning model, followed by controlled hyperparameter tuning to improve predictive performance.

---

## 1. Introduction
Efficient irrigation is a major challenge in modern agriculture, particularly in regions affected by water scarcity. Traditional irrigation practices often rely on fixed schedules or manual judgment, which can result in over-irrigation or under-irrigation. This project addresses the problem by developing a machine learning–based system that predicts irrigation requirements using environmental data, enabling smarter and more sustainable irrigation decisions.

### Project Objectives
- Predict irrigation requirements using machine learning
- Compare baseline and deep learning models
- Improve model performance through hyperparameter tuning
- Support sustainable water management practices

---

## 2. Dataset Description
The dataset used in this project is an agricultural dataset containing approximately 16,000 records.

### Features
- Moisture Index (MOI)
- Temperature
- Humidity

### Target Variable
- Irrigation requirement level (classification)

### Preprocessing Steps
- Feature selection
- Train–test split (80% training, 20% testing)
- Feature scaling using standardization

---

## 3. Methodology
The project follows a structured machine learning pipeline:

1. Data loading and preprocessing
2. Feature scaling
3. Train–test split
4. Model training
5. Model evaluation
6. Model saving
7. Inference using a separate prediction script

### Models Used
- Baseline Model: Logistic Regression
- Advanced Model: Transformer-based Deep Learning Model

The Transformer architecture was selected due to its ability to capture non-linear relationships between environmental features more effectively than traditional machine learning models.

---

## 4. Experimental Setup
- Train–Test Split: 80% / 20%
- Loss Function: Cross-Entropy Loss
- Optimizer: Adam
- Evaluation Metric: Accuracy
- Frameworks: Python, Scikit-learn, PyTorch
- Execution Environment: CPU-based

---

## 5. Results and Limitations

### Results
The following performance results were obtained:

- Logistic Regression: ~81% accuracy
- Transformer (Before Tuning): ~84% accuracy
- Transformer (After Tuning): ~87% accuracy

A bar chart visualization comparing model performance is included in the repository.

### Limitations
- The dataset contains a limited number of environmental features
- Accuracy is used as the primary evaluation metric
- The system is trained on historical data and does not yet integrate real-time sensor inputs

---

## 6. Hyperparameters
The performance of the Transformer-based model depends on several key hyperparameters. Controlled tuning was performed after ensuring error-free execution to balance accuracy and model stability.

| Hyperparameter | Value |
|---------------|-------|
| Hidden dimension (hidden_dim) | 32 |
| Number of attention heads (nhead) | 4 |
| Number of Transformer layers | 1 |
| Learning rate | 0.0005 |
| Training epochs | 20 |

The final configuration was selected to achieve stable performance without overfitting. Experiments with higher epochs and larger hidden dimensions were conducted, but the chosen configuration provided the best balance between accuracy and generalization.

---

## 7. Usage and How to Run

### Installation
Install the required dependencies using:
pip install -r requirements.txt

### Training the Model
To train the model and generate evaluation results:
python train_model.py

### Running Inference
To perform irrigation prediction using the trained model:
python predict.py

---

## 8. Repository Structure
├── data/
├── models/
│   ├── transformer_model.pth
│   └── scaler.pkl
├── results/
│   ├── accuracy_comparison.png
│   └── results.txt
├── train_model.py
├── predict.py
├── README.md
├── requirements.txt
├── Short_Project_Report_Smart_Irrigation.docx
├── Project_Documentation_Smart_Irrigation.docx
├── Smart_Irrigation_Presentation.pptx

---

## 9. Sustainable Development Goals (SDG) Alignment
This project aligns with the following United Nations Sustainable Development Goals:
- SDG 2: Zero Hunger
- SDG 6: Clean Water and Sanitation
- SDG 12: Responsible Consumption and Production

---

## 10. Conclusion and Future Work
This project demonstrates the effective application of machine learning for irrigation requirement prediction. The comparison between traditional machine learning and Transformer-based models highlights the benefits of advanced architectures for agricultural decision support systems. Future work may include incorporating additional environmental features, using advanced evaluation metrics, and integrating real-time sensor data for deployment in smart irrigation systems.

---

## 11. References
Relevant research papers and online resources on smart irrigation systems and machine learning were consulted during the development of this project.

---

## 12. Acknowledgments
Special thanks to Sir Talha for guidance and supervision throughout the project.
