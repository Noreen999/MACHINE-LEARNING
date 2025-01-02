# MACHINE-LEARNING


Dataset
The dataset used in this study consisted of:
1.Original Images: Raw images categorized as "Monkeypox" and "Non-Monkeypox."
2.Augmented Dataset: Images enhanced through rotation, zooming, flipping, and other transformations to improve generalization.
3.Metadata: A CSV file containing annotations for the images.
The dataset was split into training (70%), validation (20%), and testing (10%) sets.
Baseline CNN Model
We implemented a baseline CNN with three convolutional layers, followed by max-pooling, fully connected layers, and a sigmoid output layer. Despite extensive data augmentation and hyperparameter tuning, the model achieved a maximum accuracy of 70.15% on the test set.
MobileNetV2 Model
To overcome the limitations of the CNN model, we adopted MobileNetV2. The architecture included:
1.A pre-trained MobileNetV2 model with weights initialized from ImageNet.
2.Custom classification layers, including a global average pooling layer, dense layers with ReLU activation, dropout for regularization, and a sigmoid output layer.
3.Optimization using Adam with a learning rate of 0.0001.
The model was trained for 10 epochs using the same dataset split as the baseline model. Class weights were applied to address dataset imbalance.
Evaluation Metrics
Both models were evaluated using metrics such as accuracy, precision, recall, and F1-score. Computational efficiency was also analyzed by comparing training times.

Results
The MobileNetV2 model significantly outperformed the baseline CNN model:
Accuracy: MobileNetV2 achieved an accuracy of 88.50%, compared to the CNN's 70.15%.
Precision and Recall: MobileNetV2 exhibited higher precision (91.2%) and recall (86.8%), indicating better reliability in detecting true positives.
Training Time: MobileNetV2 required fewer epochs to converge, making it computationally efficient.
The confusion matrices highlighted the superior classification performance of MobileNetV2, particularly in reducing false negatives.

Discussion
The results validate the effectiveness of transfer learning in medical imaging tasks. MobileNetV2's ability to leverage pre-trained weights enabled it to extract more discriminative features, addressing the overfitting issues observed in the CNN model.
Key improvements observed:
1.Feature Extraction: MobileNetV2 captured fine-grained features of lesions more effectively.
2.Regularization: The use of dropout and data augmentation reduced overfitting.
3.Computational Efficiency: The lightweight architecture of MobileNetV2 facilitated faster training and inference.
Despite these advantages, the study acknowledges limitations such as dataset size and potential biases in image quality. Future work will focus on expanding the dataset and exploring ensemble techniques.

Conclusion
This study demonstrates that MobileNetV2 significantly enhances the accuracy and reliability of Monkeypox classification compared to traditional CNN models. The findings underscore the potential of transfer learning in addressing emerging public health challenges. By leveraging MobileNetV2, healthcare professionals can achieve more accurate diagnoses, contributing to timely interventions and containment efforts.

References
1.Howard, A. G., et al. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications.
2.Litjens, G., et al. (2017). A Survey on Deep Learning in Medical Image Analysis.
3.Existing CNN-based Monkeypox Classification Study (Include citation).
