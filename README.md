# Brain Tumor Classification Using CNN

This project demonstrates the implementation of a Convolutional Neural Network (CNN) to classify brain tumor images. The notebook guides through the process of loading the dataset, preprocessing it, constructing the CNN model, training it, and evaluating its performance. This README provides an overview of the notebook's structure, key features, and requirements.

## Project Overview
Brain tumor classification is a critical task in medical image analysis. This project focuses on using a deep learning approach (CNN) to automatically identify the presence of tumors in MRI scans.

### Key Features
1. **Dataset Preprocessing:**
   - Image loading, resizing, and normalization.
   - Splitting the dataset into training, validation, and testing sets.

2. **CNN Architecture:**
   - Model definition using Keras with TensorFlow backend.
   - Includes convolutional, pooling, and dense layers.

3. **Model Training and Evaluation:**
   - Compiling the model with a suitable optimizer and loss function.
   - Training the model while monitoring performance metrics.
   - Visualizing accuracy and loss trends.

4. **Results Visualization:**
   - Displaying sample predictions.
   - Confusion matrix and classification report.

## Notebook Structure
1. **Import Libraries:**
   - Required libraries such as TensorFlow, Keras, NumPy, Matplotlib, and others are imported.

2. **Data Preprocessing:**
   - Loading MRI images.
   - Applying transformations for better model performance.

3. **CNN Model Design:**
   - Constructing a multi-layer CNN for feature extraction and classification.

4. **Training the Model:**
   - Using training and validation datasets.
   - Employing early stopping or checkpoint callbacks (if implemented).

5. **Evaluation:**
   - Assessing the model on the test dataset.
   - Generating visualizations for insights.

6. **Conclusion:**
   - Key observations and model performance summary.

## Dependencies
Ensure the following Python libraries are installed:
- `TensorFlow`
- `Keras`
- `NumPy`
- `Matplotlib`
- `Pandas`
- Any other library specified in the notebook

## How to Use
1. Clone this repository or download the notebook file.
2. Install the dependencies using `pip install -r requirements.txt` (if provided).
3. Run the notebook sequentially to reproduce the results.

## Results
- Achieved classification accuracy: **94%**.
- Observations:
  - High performance on most classes.
  - Slight misclassification observed in glioma and no_tumor classes.

### Confusion Matrix
![image](https://github.com/user-attachments/assets/74ff75c0-9dc5-43c5-ac0d-cead36096823)


### Classification Report
```
                  precision    recall  f1-score   support

glioma_tumor       0.94      0.87      0.91       300
    no_tumor       0.90      0.90      0.90       306
meningioma_tumor   0.97      1.00      0.98       405
pituitary_tumor    0.95      0.99      0.97       300

    accuracy                           0.94      1311
   macro avg       0.94      0.94      0.94      1311
weighted avg       0.94      0.94      0.94      1311
```

## Future Work
- Improving model accuracy with advanced architectures.
- Expanding the dataset for better generalization.
- Applying techniques like transfer learning.

## Acknowledgments
- Dataset source: [Brain Tumor Classification Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
- Frameworks and tools used: TensorFlow, Keras, etc.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---
Feel free to contribute or suggest improvements to this project!
