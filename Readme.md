# **README**

## **Capstone Project**

Mario Albuquerque

May 31th, 2018

## **Data Source**

The project used data provided through a Kaggle competition: ["Avito Demand Prediction Challenge"](https://www.kaggle.com/c/avito-demand-prediction). Note that the location of the necessary files are assumed to have a root folder where the Jupyter Notebook Python files are located. There are two data files to be extracted:

* **_train.csv.zip_** which has a file named *train.csv* with a total of 1,503,424 ads totaling around 931,000 KB. This is the main data source with the ads. This file should be in the folder *"./Data/"*.

* **_train_jpg.zip_** which has images corresponding to the ads in the **_train.csv_** dataset. There are a total of 1,390,836 images in the zipped folder and it totals around 52,000,000 KB. Note that not all ads in the **_train.csv_** dataset have an image. This file should be unzipped in the folder *"./Data/Images/"*.

## **Python version and package requirements**

This project was done with Python 3.5.3 and needs the following packages (outside of the Python Standard Library):

* pandas 0.22.0
* numpy 1.11.3
* textblob 0.15.1
* pillow 5.0.0
* matplotlib 1.5.1 
* nltk 3.2.4
* keras 2.1.4
* opencv 3.2.0 
* ipython 6.1.0
* scikit-learn 0.19.1

## **Jupiter Notebook Python Files**

The implementation of the project was done through three Jupyter Notebooks:

* **_EDA.ipynb_**: Exploratory data analysis.

* **_Feature Engineering.ipynb_**: Feature engineering.

* **_Model Development.ipynb_**: Model development and evaluation.

