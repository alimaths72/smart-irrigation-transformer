# Smart Irrigation Requirement Prediction using Machine Learning

This project focuses on predicting irrigation requirements using environmental and soil-related features such as Moisture Index (MOI), temperature, and humidity. A baseline machine learning model and a Transformer-based deep learning model are developed and compared to determine irrigation needs effectively.

The system is designed to follow a complete machine learning pipeline, including data preprocessing, model training, evaluation, and real-time inference.
## Problem Statement

Efficient water usage is a major challenge in agriculture. Over-irrigation leads to water wastage, while under-irrigation negatively affects crop yield. Traditional irrigation decisions are often based on fixed schedules rather than real environmental conditions.

This project aims to address this problem by using machine learning techniques to predict irrigation requirements based on real-time environmental data.

## Project Objectives

- To analyze environmental and soil-related data relevant to irrigation decisions  
- To develop a baseline machine learning model for irrigation prediction  
- To implement a Transformer-based model for improved performance  
- To compare baseline and deep learning model results  
- To provide a real-time prediction system for irrigation requirements  
## Dataset Description

The dataset used in this project contains environmental and soil-related attributes relevant to irrigation decision-making. It consists of over 16,000 samples collected under varying environmental conditions.

### Features Used
- **Moisture Index (MOI):** Represents soil moisture level (range: 0.0 – 1.0)  
- **Temperature:** Ambient temperature measured in degrees Celsius  
- **Humidity:** Relative humidity expressed as a percentage  

### Target Variable
- **Irrigation Requirement:** A multi-class label representing different levels of irrigation need  

The dataset is structured and relatively clean, requiring minimal preprocessing beyond feature selection and scaling.

---

## Methodology

The project follows a structured machine learning pipeline to ensure correctness, reproducibility, and real-world applicability.

### 1. Data Preprocessing
- Selected relevant numerical features (MOI, temperature, humidity)  
- Applied train-test split to avoid data leakage  
- Used standard scaling to normalize feature values  

### 2. Baseline Model
A Logistic Regression model was implemented as a baseline to establish initial performance and provide a comparison point for the deep learning model.

### 3. Transformer-Based Model
A lightweight Transformer architecture was designed for tabular data. The model consists of:
- A linear embedding layer  
- A Transformer encoder block  
- A fully connected output layer  

This architecture allows the model to capture feature interactions more effectively than the baseline model.

### 4. Training and Evaluation
- Models were trained using the training subset of the data  
- Accuracy was used as the primary evaluation metric  
- Random seeds were fixed to ensure reproducibility and stable results  

### 5. Inference Pipeline
Training and inference were separated into different scripts. The trained Transformer model and the scaler are saved after training and loaded during inference, enabling real-time prediction without retraining the model.
## Results and Limitations

### Results
The performance of the baseline and Transformer-based models was evaluated using accuracy as the primary metric.

- **Baseline Logistic Regression Accuracy:** ~81%  
- **Transformer Model Accuracy:** ~84%  

The Transformer-based model consistently outperformed the baseline model, indicating its improved capability to learn complex relationships among environmental features relevant to irrigation decision-making.

### Limitations
- The dataset exhibits class imbalance, which can bias the model toward majority irrigation classes during prediction  
- Accuracy was used as the primary evaluation metric; incorporating additional metrics such as precision, recall, and F1-score could provide deeper insights  
- The model was trained on a specific dataset and may require retraining or fine-tuning before deployment in different geographic regions or environmental conditions  

Despite these limitations, the proposed system demonstrates reliable performance and provides a practical foundation for smart irrigation decision support.
## Usage and How to Run

This project is divided into two main components: model training and real-time prediction.

### 1. Training the Model
To train the models and save the trained Transformer model and scaler, run the following command:

python train_model.py

This script performs the following steps:
- Loads and preprocesses the dataset
- Trains a baseline Logistic Regression model
- Trains a Transformer-based deep learning model
- Evaluates and compares model performance
- Saves the trained Transformer model and scaler for inference

Note: Training needs to be performed only once unless the dataset or model configuration is changed.

---

### 2. Running Real-Time Prediction
To perform irrigation prediction using user-provided input values, run:

python predict.py

The script will prompt the user to enter:
- Moisture Index (MOI)
- Temperature (°C)
- Humidity (%)

Based on the input values, the trained Transformer model predicts the irrigation requirement level along with a confidence score.

---

### 3. Example Input
MOI: 0.45  
Temperature (°C): 32  
Humidity (%): 60  

### Example Output
Prediction: Irrigation LIKELY REQUIRED  
Prediction confidence: 0.82  

This separation of training and inference ensures efficient execution and reflects real-world machine learning deployment practices.
## Sustainable Development Goals (SDG) Alignment

This project contributes to the following United Nations Sustainable Development Goals:

- **SDG 2: Zero Hunger**  
  By enabling efficient irrigation decisions, the system supports improved agricultural productivity and crop yield.

- **SDG 6: Clean Water and Sanitation**  
  The model promotes responsible water usage by reducing over-irrigation and minimizing water wastage through data-driven decision-making.

Smart irrigation systems powered by machine learning can play a significant role in sustainable agriculture and water resource management.

---

## Conclusion and Future Work

This project successfully demonstrates the application of machine learning for smart irrigation requirement prediction. A Transformer-based model was developed and compared with a baseline Logistic Regression model, achieving improved and reproducible performance.

The separation of training and inference pipelines ensures efficient execution and reflects real-world machine learning deployment practices.

### Future Work
- Incorporating additional environmental features such as rainfall and soil type  
- Addressing class imbalance using advanced resampling techniques  
- Evaluating the model using additional performance metrics  
- Deploying the model as a web or mobile-based decision support system  
