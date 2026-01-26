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
