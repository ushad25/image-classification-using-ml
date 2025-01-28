

## **Author Information**  
| **Author**        | **Usha Rahul**              |  
|--------------------|-----------------------------|  
| **Date**          | January 7, 2025            |  
| **Company**       | CodeTech IT Solutions      |  
| **Intern ID**     | CT0806HT                   |  
| **Domain**        | Machine Learning           |  
| **Mentor**        | Neela Santhosh Kumar       |  
| **Batch Duration**| December 30, 2024 – February 14, 2025 |  

---
# **Pet Classifier: Cats vs. Dogs(TASK 3)**

## **Overview**  
This project involves building a machine learning model to classify images of cats and dogs. It uses a Convolutional Neural Network (CNN) to distinguish between the two classes. The dataset comprises labeled images of cats and dogs, divided into training, testing, and validation sets.  

The model was trained using TensorFlow and Keras and evaluated based on its accuracy and loss metrics. While the current accuracy is 54.29%, further improvements are recommended to achieve better performance.  

---

## **Dataset**  
- **Source**: Custom dataset containing images of cats and dogs.  
- **Structure**:  
  ```
  ├── train/
  │   ├── cats/
  │   └── dogs/
  ├── test/
  │   ├── cats/
  │   └── dogs/
  ```


## **Model Architecture**  
The Convolutional Neural Network (CNN) was used with the following layers:  
- **Conv2D Layers**: Extract spatial features.  
- **MaxPooling Layers**: Reduce spatial dimensions.  
- **Dense Layers**: Classification based on extracted features.  

---

## **Steps to Run the Project**  

### **1. Environment Setup**  
Ensure you have the following installed:  
- Python (>=3.8)  
- TensorFlow (>=2.6.0)  
- Keras  
- NumPy  
- Matplotlib  

Install the required libraries:  
```bash  
pip install tensorflow numpy matplotlib  
```  

### **2. Dataset Preparation**  
Organize your dataset in the following directory structure:  
```
project_directory/
├── train/
├── test/
```  
Place images of cats and dogs in their respective folders.  

### **3. Training the Model**  
Run the training script to train the CNN on the dataset:  
```python  
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)
```  

### **4. Evaluating the Model**  
Evaluate the model on the test dataset:  
```python  
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
```  

### **5. Single Image Prediction**  
Use the following script to predict the class of a single image:  
```python  
from tensorflow.keras.preprocessing import image  
import numpy as np  

img_path = r"C:\path_to_image\dog_or_cat.jpg"  
img = image.load_img(img_path, target_size=(150, 150))  
img_array = image.img_to_array(img) / 255.0  
img_array = np.expand_dims(img_array, axis=0)  

prediction = model.predict(img_array)  
if prediction[0][0] > 0.5:  
    print("It's a Dog!")  
else:  
    print("It's a Cat!")  
```  

---

## **Results**  
- **Test Loss**: 0.6901  
- **Test Accuracy**: 54.29%  

---

## **Conclusions**  
The model achieved moderate performance, with an accuracy of 54.29%.  
### **Limitations**:  
- The model underperforms compared to expectations, likely due to insufficient training data or inadequate model complexity.  

### **Suggestions for Improvement**:  
1. **Data Augmentation**: Add transformations like rotations, flips, and shifts to increase data diversity.  
2. **Hyperparameter Tuning**: Experiment with learning rates, batch sizes, and optimizers.  
3. **Architecture Changes**: Use pre-trained models (e.g., VGG16, ResNet) for better feature extraction.  

---

## **Visualization**  
![image](https://github.com/user-attachments/assets/9a5e7bb0-6e36-4d06-8f56-a7d516c6ee5c)

---

## **Future Work**  
1. Implement pre-trained models using transfer learning.  
2. Increase the dataset size for better generalization.  
3. Improve handling of class imbalance.  

---

## **Contact**  
For questions or suggestions, feel free to reach out:  
- **Email**: ushamuth18@gmail.com

---  

