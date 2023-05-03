
# Breast Cancer Detection project



![GitHub watchers](https://img.shields.io/github/watchers/d-elicio/Breast-Cancer-Detection?style=social)
![GitHub forks](https://img.shields.io/github/forks/d-elicio/Breast-Cancer-Detection?style=social)
![GitHub Repo stars](https://img.shields.io/github/stars/d-elicio/Breast-Cancer-Detection?style=social)
![GitHub last commit](https://img.shields.io/github/last-commit/d-elicio/Breast-Cancer-Detection?style=plastic)

Design and implementation of a machine learning **Spark application** for breast cancer detection

## 🚀 About Me
I'm a computer science Master's Degree student and this is one of my university project. 
See my other projects here on [GitHub](https://github.com/d-elicio)!

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://d-elicio.github.io)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/domenico-elicio/)

# 💻 The project

- The project consists in the design and the implementation of a Spark application that analyzes the [dataset](https://www.kaggle.com/uciml/breast-cancerwisconsin-data) and applies various Machine Learning techniques to create a **classification model** for the *detection* and *prediction* of breast cancer.

- To build the ML models the **Pipeline approach** of the *MLib Spark API* has been used.  

- **Five** different **classification models** have been used (*Logistic Regression*, *Decision Three*, *Random Forest*, *Linear Support Vector*, *Naïve Bayes*)

- All these models have been evaluated by using different **accuracy** metrics to choose the best model.



## Description
![Breast_cancer_cells](https://user-images.githubusercontent.com/96207365/183132955-af2237f6-9cd1-4381-b075-25b1aca4293c.jpg)

The columns of the used [dataset](https://www.kaggle.com/uciml/breast-cancerwisconsin-data) represent all the features computed from
a digitalized image of a [fine-needle aspiration (FNA)](https://www.cancer.org/cancer/breast-cancer/screening-tests-and-early-detection/breast-biopsy/fine-needle-aspiration-biopsy-of-the-breast.html) of a breast mass. After all these exams, digital images are computed and all the features of the cell nuclei have been collected into the datased that will be used to predict the diagnosis (M=malignant, B=benign).

### Explorative analysis
First phase of every machine learning project regards dataset analysis, attributes understanding, missing values search and their fixing and the verification of dataset's classes balance.

**Heatmaps**, **pairplots** and **histograms** have been used in this phase.

![CROSS CORRELATION](https://user-images.githubusercontent.com/96207365/183138105-8aaeee9e-c264-4257-8d02-f5304ca32357.png)


### Pipeline approach
A Pipeline is specified as a sequence of stages, and each stage is either a **Transformer** or an
**Estimator**. These stages are run in order, and the input DataFrame is transformed as it passes
through each stage.

Various classification methods have been used in this work, but the first three steps of the pipeline are in common for all classification methods:
- **StringIndexer**: used to transform the string values (M or B) of the “diagnosis” column in numeric binary values (0=B and 1=M) 
- **VectorAssembler**: a feature-transformer that merges multiple columns (30, in this case --> the initial 32 columns minus the first two: the ID column (it's irrelevant) and the diagnosis column (it's the target attribute to evaluate) into a vector column.
- **StandardScaler**: used to standardize features by removing the mean and scaling to unit variance using column summary statistics on the samples in the training set.

![Pipeline](https://user-images.githubusercontent.com/96207365/183140908-1ce31cdf-8690-4e0e-9562-870dd5aca320.jpg)

### Classification
Five different machine learning models have been used in this work:
- *Logistic Regression*
- *Decision Three*
- *Random Forest*
- *Linear Support Vector* 
- *Naïve Bayes*

### Evaluation

Different evaluation metrics have been used in this project to evaluate the accuracy of the models:
- **Precision** = how many are correctly classified among that class
- **Recall** = how many of a certain class you find over the whole number of element of this class
- **F1-score** = the harmonic mean between *precision* and *recall*
- **Support** = number of occurrences of a given class in your dataset (if you have 37.5K instances of class 0 and 37.5K of class 1, it is a really well balanced dataset)
- **Accuracy** = rate of positive predicted values
- **Test Error** = 1 - *Accuracy*
- **ROC Curve** and **AUC** value

![RANDOM FOREST](https://user-images.githubusercontent.com/96207365/183143231-37541e66-a571-4b49-a0b4-8fa336e8fa03.png)

![ROC Random Forest](https://user-images.githubusercontent.com/96207365/183143248-1ea70fc9-e8cb-450d-9e86-b9dff894a969.png)
## Support

For any support, error corrections, etc. please email me at domenico.elicio13@gmail.com 

