# Image Classification of 225 Aves Species

In this project, we utilize various ImageNet Pre-trained Neural Network models for classification of 225 Bird species. 
Additionally, we also use Machine Learning models such as Logistic Regression and SVM to classify the images using the features extracted using various techniques.

For feature extraction, we implemented the following techniques:

	1. HU Moments
	2. Haralik Textures
	3. Color Historgram
	4. VGG16 for feature extraction
	5. Bag of Visual Words

The dataset consists of over 33 thousand 224x224 color images of 225 different bird species.

For a detailed study, kindly refer the document: "DS5220_Project_Report.pdf"

The following are the code files:

	1. Log_Reg_Flattened.ipynb - Logistic Regression using flattened feature vectors

	2. Log_Reg_Reduced.ipynb - Logistic Regression using only 50 images per class VGG16 extracted features

	3. Saving_Extracted_Features.ipynb - Saving the VGG16 extracted features to files

	4. SVM_Flattened.ipynb - Attempt to train SVM on flattened features - did not execute even after 40 hours

	5. SVM_GRIDSEARCH_Reduced.ipynb - Hyperparameter tuning for reduced data - only 50 images per class

	6. SVM_LogReg_Load_Saved_Models.ipynb - Loading the best performing saved SVM and Logistic Regression models and displaying accuracy

	7. SVM_training.ipynb - Training the SVM model with complete VGG 16 extracted features
	
	8. pretrained_cnn_models - Contains code to train various ImageNet pre-trained models offered by PyTorch. They include variations of densenets, squeezenets, VGG, resnets etc.
