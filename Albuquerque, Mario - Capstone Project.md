# **Machine Learning Engineer Nanodegree**
## **Capstone Project**
Mario Albuquerque  
May 31th, 2018

## **I. Definition**
### **Project Overview**
The subject of the project was taken from a Kaggle competition named ["Avito Demand Prediction Challenge"](https://www.kaggle.com/c/avito-demand-prediction). Today's retail environment is increasingly dominated by online platforms (e.g. Amazon, eBay, Alibaba, etc.) that want to reach as wider audience as possible in order to sell goods and services. In this business it is crucial to determine how the product is advertised so that the right balance between maximizing demand (marketing spending) and profit (pricing optimization) can be achieved. The goal of this project is to apply machine learning in a diversified dataset, that was sourced in a Russian online retail platform, to predict the likelihood of an ad resulting in a deal.
This project is extremely interesting as it allows the combination of text, numerical, and imaging data in one data science challenge so that a more expanded set of skills can be showcased.

The dataset was provided by Avito (through Kaggle) and it represents advertisements (ads) along with the images (when available) of the products/services being sold.

The data can be downloaded in this website: https://www.kaggle.com/c/avito-demand-prediction/data.  
For the elaboration of this project the relevant files are the **_train.csv_** which is named **_train.csv.zip_**, and the **_train_jpg_** folder with images which is named **_train_jpg.zip_**.

### **Problem Statement**

The problem is one of determining the demand for an online advertisement given its endogenous characteristics (eg. advertisement title, and item price), geographical location (region and city), and similarity to other items (category and parent category). These are the features, or inputs for the problem.  
The demand for the ad is expressed as a probability (named **_deal_probability_** in the dataset) and this is the target variable or the output of the model. Without any further transformation, the problem is a supervised regression problem as the output variable is continuous (probabilities). However, it will also be attempted to transform the output variable (**_deal_probability_**) into a 2-class supervised classification problem, where the model will assign **_Unlikely_** and **_Likely_** labels depending on whether the probability of the deal is below 50% or not, respectively.

The solution for this problem will take two approaches:

**A classification-based approach**: Where the dependent variable is going to be discretized into two labels based on having a **_deal_probability_** below 50%, or not. The labels will be named **_Likely_** and **_Unlikely_**, based on the **_deal_probability_** value being equal or above 50%, and below 50%, respectively. In this approach, multiple supervised classification models will be tried and combined so that the F1-Score is maximized in a cross validation exercise. The final performance evaluation on the best model will be assessed in the testing subset.

**A regression-based approach**: Where the dependent variable is going to be predicted through multiple regression models. In this approach, multiple models will be tried and combined so that the Root Mean Squared Error is minimized in a cross validation task. The final performance evaluation on the best model will be assessed in the testing subset.

### **Metrics**

As both classification and regression-based approaches are going to be tested, there is going to be one metric for each type of problem.

**For classification-based models**: The evaluation metric is the F1-Score as it balances the precision and recall trade-off. This is especially relevant in cases where the target variable is imbalanced, such as in this project (Out of a total 1,503,424 ads, 1,321,411 ads or 88%, have **_deal_probability_** lower than 0.5, while 182,013 ads or 12%, have **_deal_probability_** equal or higher than 0.5). This is why the F1-Score was selected as the metric for Supervised Classification in this project: the F1-Score represents a harmonic mean between precision and recall and will be closer to the lowest value between the two. The equation is given by:  
![f1_score](./images/f1_Score.png)   
Source: https://en.wikipedia.org/wiki/F1_score

**For regression-based models**: The evaluation metric is the root mean squared error (RMSE) which is a quantitative way to express the average deviation of the predicted **_deal_probability_** from the actual value. The equation is given by:  
![rmse](rmse.png)  
Source: https://www.kaggle.com/c/avito-demand-prediction#evaluation

## **II. Analysis**

### **Data Exploration and Visualization**

The composition of the data is as follows:

* **_train.csv_**: Contains 1,503,424 ads with the following columns:  
![column_desc](columns_desc.png)  
Source: https://www.kaggle.com/c/avito-demand-prediction/data

The approach taken for this capstone project is to further divide the **_train.csv_** file into train and testing subsets. Although Avito provides additional records assigned to the testing subset in the Kaggle challenge, they do not have the dependent variable (**_deal_probability_**) that would make it possible to evaluate machine learning models. The Kaggle challenge sponsor holds back the **_deal_probability_** column from the testing group so that it remains a competitive and truly out-of-sample exercise.

* **_train_jpg_**: A folder containing 1,390,836 images that correspond to the classified ads in the **_train.csv_** file that had an image. The column named *image* in the *train.csv* file has the filename of the image associated with a specific ad.

The dataset in the **_train.csv_** file is composed of 1,503,423 ads and 18 columns. 11 columns have complete records (zero nulls), while 7 columns have at least one null value. Three columns are of type **_float64_**, one column is of type **_int64_**, and the remaining columns are of type **__object__**.

Here are the first 5 ads of the dataset (broken down into two images due to space limitations):  
![dataset_head_1](dataset_head_1.png)

The columns hold different ad characteristics that range from numerical data (e.g. **_price_**, **_image_top_1_**) to text objects (e.g. **_title_**, **_description_**). The text objects are expressed in the Russian language.

The last column (**_deal_probability_**) is the quantity to be forecasted (dependent variable).

Note that the dataset is augmented by images shown in the ads (the **_image_** column has the filename of the picture). The location of the images is in a separate folder. Each ad has at most one image, and it can have no image at all.

#### **Distribution of the dependent variable**
Key to this problem is to understand the distribution of **_deal_probability_**:  
![dealprobability_hist](dealprobability_hist.png)

Note that the majority of the records have a deal probability of zero. In fact, 974,618 records (or 64.83% of the data) has a deal probability of zero.

#### **Data breakdown across categorical variables**

##### **Region**

The ads are fairly diversified across regions with the largest concentration at 9.4% in the Krasnodar region. The pie chart below has more detail:

![region_breakdown](region_breakdown.png)

In the **_EDA.ipynb_** file, a table with statistics on **_deal_probability_** per region is shown (it is not presented here due to space limitations). In all of the regions, more than 50% of deal probabilities are zero. Furthermore, the maximum 75th percentile across regions is 20%, which is a fairly low probability. Looking at the numbers it seems that a lot of deals do not go through completion. At the region level, the average deal probability is within a somewhat compressed range (12.04% to 15.59%).

##### **City**

There are a total of  1733  unique cities. The least represented city is  Eagle-Emerald  with  0.00% and the most represented city is  Krasnodar  with 4.23%.  
Given the widespread distribution of ads per cities across 1733 cities, it is more relevant to aggregate data per region.

##### **Parent Category**

Ads are concentrated in the *Personal things* category, with about 46% of all ads, as the following pie chart shows:

![parentcategory_breakdown](parentcategory_breakdown.png)

The following table will show that a more interesting picture emerges from looking at **_deal_probability_** through the **_parent_category_name_** column:

* *The services* parent category is characterized by the largest mean deal probability (40%).
* *Personal things* parent category is the most frequent parent category in the dataset but it has the lowest average deal probability (7.59%).
* The intermediate parent categories (in terms of deal probability) are concentrated in a fairly narrow range (11.10% to 26.33%).
* The **_parent_category_name_** feature has some potential in separating high vs. low deal probability ads.

![dealprobability_parentcategory](dealprobability_parentcategory.png)

##### **Category**

The **_category_name_** feature allows for a more detailed dive on the **_parent_category_name_** and shows some interesting dynamics in terms of **_deal_probability_**:

* *The services* does not have a more detailed breakdown.
* *Animals* ranks 3rd in terms of average deal probability within the parent_category_name filter (26.60%). However, the sub-categories have a somewhat wide dispersion of average deal probabilities (e.g. "Cats" with 29.73% and "Goods for pets" with 13.36%).

##### **Activation date**

The distribution of ads per **_activation_date_** is very close to an equal weight (1/unique dates), with the exception of the latest 7 days in the dataset (from 2017-03-29 onwards) where a low number of ads are present.

![date_breakdown](date_breakdown.png)

Even though there are days where **_deal_probability_** is extremely high, it occurs in situations where sample size is very low. A weekday vs. weekend feature might be thought of as something that would differentiate deal probability across activation dates, but it would not be a very robust rule to look at (e.g. 2017-04-01 had an average 80.32% deal probability but only 3 deals).

##### **User type**

The data is clearly dominated by the "Private" user type as opposed to professional user types like "Company" or "Shop": "Private" users represent 74.7% of the ads, followed by "Company" users with 21.8%, and finally "Shop" users with 3.5%.

Surprisingly, the average **_deal_probability_** of the *Private* user type is the highest at 14.96%. My expectation was that professional users like *Company* and *Shop* would post higher quality ads positioned in front of the website as to increase deal probability, but this does not appear to be the case.

It is also interesting to note that the *Shop* category type has a low average **_deal_probability_** (6.28%). This might be explained by many items sold by *Private* user types being second hand, and hence, having a lower price relative to the *Shop* user price.

![dealprobability_user](dealprobability_user.png)

#### **Data breakdown across continuous variables**

As the following table shows, it is interesting to learn that not all ads have a price (price count is different than total number of rows in dataset). The **_price_** variable is heavily influenced by extreme outliers (see the 'max' row and how the standard deviation is much greater than the mean value).

The **_image_top_1_** variable has as many counts as there are ad images, as expected (1,390,836).  
![continuous_stats](continuous_stats.png)

In order to conveniently represent the **_price_** information in a histogram, a log transformation is made so that the range of values is more compressed:  
![log_price](log_price.png)

Note how there is one peak in the log of **_price_** distribution, corresponding to the majority of ads having a mid-range price.

The column **_image_top_1_** has a clear concentration of ads at lower values:  
![imagetop1_hist](imagetop1_hist.png)

Finally, the **_item_seq_number_** variable is also characterized by a large concentration of values in the lower region:  
![itemseqnumber_hist](itemseqnumber_hist.png)

#### **Behavior of **_deal_probability_** across continuous variables**

In the EDA.ipynb file, three scatter plots are shown that do not show a clear linear relationship between either **_price_**, **_image_top_1_**, or **_item_seq_number_** with **_deal_probability_**.

However, it is interesting to note three characteristics:

* The **_price_** variable at high values has less points on the upper range of **_deal_probability_**. Notice how the number of points in the upper range of **_deal_probability_** decays (points become scarce) as **_price_** increases.

* Looking at **_item_seq_number_**, there appears to be a somewhat concave relationship, ie, the higher the **_item_seq_number_** , the more narrow is the range of the **_deal_probability_** variable towards lower values.

* There is a gap on the **_deal_probability_** variable from values around 0.9 to 1.0 (in all variables).  

#### **Correlation matrix between continuous variables**

The only noteworthy aspect of the linear correlations is that the **_image_top_1_** variable has the largest postive correlation with **_deal_probability_** (0.21). This variable might be a quantitative metric of image quality, as the better the image quality, the higher the likelyhood that a deal occurs.

![corr](corr.png)

### **Algorithms and Techniques**

The problem will be tackled through two approaches: supervised classification and supervised regression. The dependent variable on the supervised regression is **_deal_probability_**, and in the supervised classification it is a discretized version of **_deal_probability_** where ads at or above 0.5 are flagged as *Likely*, and the ones below 0.5 are flagged as *Unlikely*. As such, each approach will have their own algorithm implementation.

**Supervised classification models:** The models described below are of the type that estimate a function that link continuous input features to a data label, by discretizing the output that usually comes expressed as a probability of belonging to a particular class. Unless otherwise noted, the parameters used in doing cross validation checks were the scikit-learn default ones. Also, depending on the model, the *random_state* option was set in order to achieve replicability of results.

* Gaussian Naive Bayes: Assumes independence of all features, which in the case at hand is reasonable. The probability distribution of features is assumed to be normal.
  * Logistic Regression: Despite the name, logistic regression is a classification algorithm that models the relationship between the discreet dependent variable with multiple continuous independent variables. It does so using a probabilistic model that uses a function to map continuous variables in the range of ]-inf, +inf[ to a range of [0, 1].
* Support Vector Machines: This algorithm seeks to find the closest points belonging to opposing classes and draw a decision boundary on that zone. This model has a lot of flexibility as it separates the data points across the classes by using either linear or non-linear boundary curves. 
* Random Forests: This is an ensemble algorithm that combines multiple tree-based models that attempt to establish a series of questions that most clearly divides the data into the two classes, at each decision stage, until a final label assignment is done. The grouping of multiple trees and taking the majority vote of the class is an exercise in reducing overfitting, because no specific tree dominates the final output. This model is excelent at handling multiple types of variables as is the case in this project.
* Gradient Boosting: Combines several weak learners that result in a series of models specialized in classifying a certain aspect of the dataset, that when put together perform much better. This algorithm lends itself nicely to create decisions based on multiple types of features, as in this project. 
* Ada Boost: This model is an extension of the Gradient Boosting model in such a way that each sequential learner will be tuned to the errors made so far by previous learners. It gets smarter in addressing errors as more learners are combined.
* K-Neighbors: This algorithm will return the class of the k nearest neighbors (in terms of proximity to all the feature values) of the data point of interest. 

**Supervised regression models:** In this type of approach, the function that links feature inputs to the output generates a continuous output and it is appropriate to forecast actual values, as the **_deal_probability_** column in this project.
* Linear Regression: Establishes a linear link between all the features (independent variables) and **_deal_probability_** by finding the best linear combination of the features that result in the least sum of squared errors.
* Lasso Regression: This is an extension to the linear regression approach where a regularization parameter on the coefficients is enforced. The coefficient penalty is expressed as an absolute sum of coefficients.
* Ridge Regression: This is an extension to the linear regression approach similar to Lasso regression where the coefficient penalty is expressed as the squared sum of coefficients.
* Gradient Boosting: This algorithm was described in the previous section and it can also be applied in a supervised regression setting.
* Multi-layer Perceptron: This is a neural network-based approach that uses fully connected layers between the input features and the output. This approach represent a non-linear link between inputs and outputs and it is useful to uncover non-linearities in the data.

### **Benchmark**

The benchmark models were developed by selecting the most intuitively significant feature from the available columns in the *train.csv* file. That feature is *price* as it is, intuitively, an important driver of deal probability. The higher the price, the lower the deal probability, and vice-versa.

**Classification-based benchmark**: For each item category, compute whether the price of a specific ad is above or below the **_category_name_** mean. If it is equal or above, then the predicted label is "Unlikely", if it is below the **_category_name_** mean, then the predicted label is "Likely". Notice that it is important to condition by **_category_name_**, otherwise big ticket items like cars would be systematically flagged as "Unlikely", even though they could be excellent deals.

The Jupyter Notebook named **_Model Development.ipynb_** has a section that handles the benchmark construction for the supervised classification problem:  
![bench_class_code](bench_class_code.png)

The classification benchmark rule achieves an F1-Score of 0.21 on the test dataset. Below is the classification report:

![bench_class_report](bench_class_report.png)

**Regression-based benchmark**: A linear regression with **_price_** as the independent variable and **_deal_probability_** as the dependent variable. The parameters of the model are going to be estimated across the full dataset to predict **_deal_probability_**.

The Jupyter Notebook named **_Model Development.ipynb_** has a section that handles the benchmark construction for the supervised regression problem:  
![bench_reg_code](bench_reg_code.png)

The regression benchmark model achieves a root mean squared error of 0.263 in the test dataset.

## **III. Methodology**

### **Data Preprocessing**

The dataset in **_train.csv_** is fairly cleaned, with just two alterations needed. Besides that, there is some data processing to create a zscored numerical variable, express categorical variables, and extract features from text and image data. More details below.

#### **Outliers**

From the **_price_** distribution plot presented earlier, it is clear that there are outliers in the data. A common and reasonable approach to handle outliers is to remove records which have **_price_** above (below) 1.5 times the inter-quartile range above (below) Q3 (Q1). This method excludes 247,266 ads (16% of the data).

#### **Ads with no **_price_****

There are 85,362 ads without price. The distribution of **_deal_probability_** among those ads is fairly wide, and it is counterintuitive, as it was expecting that the absence of a price would correspond to either an extremely high likely hood of the deal to occur (free item) or an extremely low likelihood (invalid ad without the price information). Records with no price information were removed from the dataset.

#### **Price ZScore per category**

Intuitively, the price variable should be related to **_deal_probability_**. However, as was seen in the Exploratory Data Analysis section, the relationship is far from being a linear one, and different items will have different price magnitudes. For this reason, conditioning the price data by category should improve the information contained in the price column. 

An additional column named **_price_zscore_** was created by taking the difference that each price has to the category mean and dividing by the category standard deviation. This is a quantitative way to normalize price by category, and control different magnitudes of prices across categories. This metric will be positive, the more expensive an item is relative to the category mean, and negative, the more cheap an item is relative to the category mean.

#### **Categorical data**

The data has multiple categorical data objects. In order to be able to use that information in machine learning, the unique values of each categorical feature is assigned a new column in the dataset. For each record, a *1* flags the presence of a specific value in a categorical variable, otherwise it is marked with *0*. The dataset column number was expanded by the number of unique value in the following features: **_region_**, **_category_name_**, and **_user_type_**.

An interesting aspect to explore is to assess **_deal_probability_** between a weekday and a weekend day. It is hypothesized that during weekends, there are more users logged into the website checking the ads, generating more deals for the platform.

For this reason, an additional categorical variable is extracted from **_activation_date_** that identifies each ad belonging to a weekday or a weekend day.

#### **Text data**

For an ad to be effective, it has to be succinct and convey all the information that the potential buyer needs to make an informed decision. If a large quantity of capitalized letters and punctuation (example would be exclamation points) are used, or if the title/description is too long, it is reasonable to expect that **_deal_probability_** decays, as the ad becomes similar to SPAM-type communication.

The following variables are used in order to capture clear and succinct communication:

* Proportion of capitalized letters used in title and description
* Proportion of punctuation used in title and description
* Proportion of Russian stop words used in title and description
* Length of title
* Length of description

#### **Image data**

A high quality image attached to an ad is expected to increase its **_deal_probability_**. Three different metrics were constructed in order to describe some aspect of image quality:

**The unambiguous identification of the object the image contains:** the ResNet-50 model arquitecture with pre-trained ImageNet weights was used to identify the object in the ad images provided. The highest likelihood outputted by the model is used as a metric of unambiguous object identification (example: 0.9 likelihood that the object is a chair encodes a higher quality of picture than if the highest class likelihood is 0.1).

**Image blur metric:** the approach followed was taken from this website: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/. The technique works by detecting fast intensity changes within the picture and computing the variance of those responses. A blurred image is characterized by having less variability as details are more connected in the image, and there are less edges.

**Image brightness metric:** it is intuitive that photos with poor lightning convey low quality than if the image is well lit. The methodology taken here is to convert an image to grayscale, and then take the average pixel value across the photo as a way to measure image brightness.

#### **Feature cleaning**

Ads with at least one missing (*nan* values) datapoint were replaced by zero. There are four columns were this happened: **_image_top_1_**, **_prop_caps_description_** (proportion of capitalized letters in the **_description_** field), **_prop_pun_description_** (proportion of punctuation in the **_description_** field), and **_prop_stop_description_** (proportion of stop words in the **_description_** field). This is necessary for machine learning algos to be able to be trained.

This replacement does not contradict the meaning of the variables:

* When **_image_top_1_** is missing, it means that there is no picture for the ad.

* The other columns (**_prop_caps_description_**, **_prop_pun_description_**, and **_prop_stop_description_**) have nan values as a result of division by zero when there are no cases of capitalized letters, use of punctuation, or stop words in the description of the ad. It follows that if such cases are not present, a replacement by zero is appropriate.

#### **Features and targets**

In total, there are 95 features (independent variables) and 2 target variables (dependent variables: one for the classification problem and another for the regression problem).

### **Implementation**

The Jupyter Notebook file named **_Model Development.ipynb_** aggregates the relevant data modeling tasks, starting from train/test splitting:

#### **Train/Test Split**

The dataset was split into a training and testing subsets. The size of the training set is 702,477, and the size of the testing dataset is 468,319. This was made such that the testing dataset represents around 40% of the total ads in the dataset.

![code_train_test_split](code_train_test_split.png)

#### **Model Comparison with 10-fold cross validation**

For each type of approach (classification vs. regression) a 10-fold cross validation was done on the training dataset for each model and the performance metric (F1-Score and RMSE) was recorded across the 10 folds.

#### **Supervised Classification Implementation**  
![code_class_crossval](code_class_crossval.png)
#### **Supervised Regression Implementation**  
![code_reg_crossval](code_reg_crossval.png)

Note that in the case of supervised regression, there were situations where the default hyper-parameters were override due to computing performance issues:

* Ridge Regression: set *solver* parameter to *auto*.

* Gradient Boosting Regressor: set *max_features* to 10.

* Multi-layer Perceptron Regressor: set *hidden_layer_sizes* to 3 hidden layers of 10 nodes each. Also increased *max_iter* to 500 in order to achieve convergence.


### **Refinement**

The best model from the cross validation section above was further optimized through hyper-parameter grid search. The tuned model was then used to generate predictions on the test subset and performance metrics were computed.

In the case of **supervised classification**, the best cross validated model was **Naive Bayes**, and as such no tuning was made as there are no hyper-parameters to tune.

With regards to **supervised regression**, the best cross validated model was **Gradient Boosting regressor** and the main hyper-parameter chosen to improve performance was the learning rate as that determines the speed at which weights are updated in the model. The grid search was performed across three values for the learning rate: 0.1, 0.5, and 1.0. The reason for not including more values or not including other parameters to optimize is related to how sensitive this problem case is to overfitting (as seen in the erratic performance scores of the more complex models), and also computing time, which would increase greatly.

The best learning rate turns out to be 1.0. In the cross validation exercise, the Gradient Boosting Regressor model with default hyper-parameter values achieves an average RMSE of 0.242, whereas the model with learning rate set at 1.0 (holding everything else constant) achieves an average RMSE of 0.235 (a -2.9% reduction in RMSE).

The out-of-sample RMSE is 0.236 (vs. 0.243 RMSE for the model with default learning rate), in line with the cross validation task (-2.9% reduction in RMSE on the testing subset). This is the behavior that one wants to see: performance being roughly the same in the testing subset as in the training subset. This is evidence that there was no attempt to overfit.

## **IV. Results**

### **Model Evaluation and Validation**

#### **Supervised Classification**  
![class_crossval](class_crossval.png)

Interestingly, it is the simpler model (Gaussian Naive Bayes) that performs the best across the 10 folds.

Linear Support Vector Classifier is the most inconsistent with the highest standard deviation of F1-Score across folds (0.08).

Logistic Regression is the worst model with a mean F1-Score of 0.02.

In terms of out-of-sample performance, Gaussian Naive Bayes achieves a F1-Score of 0.35, which is in line with results seen in the cross validation exercise. This is a robust model that is fast and easy to train. Sometimes, complexity does not work, as in this case.

#### **Supervised Regression**  
![reg_crossval](reg_crossval.png)

The model that achieves the lowest average RMSE (0.242) across the 10 folds is the Gradient Boosting Regressor.

Again, the most basic model (linear regression) achieves solid performance against more complex alternatives. This case is very prone to data overfiting and the most simple approach seems to work better.

The Multi-layer Perceptron Regressor, which can handle non-linearities and be complex, is the most erratic across the 10 folds with the highest standard deviation of RSME (0.007).

The testing performance of the best cross validation model (Gradient Boosting Regressor), after hyper-parameter tuning, is a RMSE of 0.236, which is in line with the RMSE achieved in the cross validation exercise (0.242).

### **Justification**

#### **Supervised Classification**

In the case of classification, only the Naive Bayes algorithm achieves consistent superior performance against the benchmark (as evidenced by the F1-Score across 10 folds, above). 
In the testing subset, the benchmark achieves an F1-Score of 0.21, where the Naive Bayes model hits 0.35 (a 66.67% improvement).

#### **Supervised Regression**

The benchmark in the case of Supervised Regression achieves the worst performance in terms of RMSE (0.26). 
The best model across the 10 folds is the Gradient Boosting Regressor which averages a RMSE of 0.24 in the cross validation exercise. The out-of-sample RMSE of the Gradient Boosting Regressor, without hyper-parameter optimization yields a RMSE of 0.243 (a -6.5% reduction in RMSE). After tuning the learning rate of the model, the RMSE further decreases to 0.236, pushing the improvement over the benchmark to -9.2% reduction in RMSE. 

## **V. Conclusion**

### **Free-Form Visualization**

#### **Class imbalance in Supervised Classification**

Looking at the confusion matrix of the Supervised Classification solution, it is clear that due to the target imbalance (411,497 Unlikely deals and 56,822 Likely deals), the Gaussian Naive Bayes achieves a higher recall and precision values for Unlikely vs. the Likely labels:  
![confusion_matrix_1](confusion_matrix_1.png)

In other words, out of 468,319 records in the test dataset, 411,497 have *deal_probability* below 50% (*Unlikely*). This represents 87.87% of the test dataset.

A naive model that would flag every ad as *Unlikely* would have a high precision value (assuming that a *positive* is *Unlikely*: 411,497 / (411,497+56,822) = 87.87%) as well as a high recall value (411,497 / (411,497+0) = 100%). However, this would not be a useful model. 

If a *positive* is considered *Likely* then it becomes clear that the Gaussian Naive Bayes improves upon the naive model mentioned in the previous paragraph.

#### **Feature Importance in Supervised Regression**

An interesting visualization is to analyze feature contribution to the overall decision boundary in the Gradient Boosting Regressor algorithm. Given that the total number of features are 95, the most importance features were considered to be the ones that cumulatively total about 80% of the feature importance metric in *sklearn*:  
![feature_importance](feature_importance.png)

Interesting obsevations from the plot above:

* Feature importance resembles an exponential distribution, with a small group of features representing most of the overall feature importance. 18 features are responsible for roughly 80% of feature importance while representing about 19% of total features (18 out of 95).
* The top three features are **_image_top_1_**, **_cat_a service offer_**, and **_price_**, and they represent a third of total feature importance. It is not surprising that a mix of data objects (image, text, and numerical) contribute to the problem's solution.
* Regional information is not significant as it doesn't appear in the most important features.
* Out of the 18 features, four of them were engineered from the original dataset (**_price_zscore_**, **_length_description_**, **_img_brightness_**, and **_object_prob_**). This shows the importance of feature engineering.
* Eight features are associated with the category of the item. It is crucial to get the category of the item well assigned.

### **Reflection**

This project was structured in the following main parts:

* **Exploratory Data Analysis:** Where the data was explored to gain an understanding of the underlying problem. There was some light data cleaning where outliers, and ads without *price* were excluded from the original dataset.

* **Feature Engineering:** The original data contained multiple types of data (numerical, categorical, text, and image) which had to be processed for the machine learning algorithms. Also, the original data was expanded through construction of quantitative metrics that conveyed different aspects of the ads (e.g. image quality, clarity of communication, richness/cheapness of an item vs. item category). Any invalid records (*nan* values) were replaced by zero.

* **Model Development:** In this section, benchmark models for both Supervised Classification, and Supervised Regression were built. The dataset was separated into train/test subsets. The training subset was used to do a 10-fold cross validation of the two types of models considered for this problem (Supervised Classification and Supervised Regression). The best model from the cross validation exercise was then optimized for the best hyper-parameter (this only applied to Supervised Regression). In the end, the best tuned model was used to generate predictions on the testing dataset, and performance metrics were computed.

Throughout the development of this project, several key insights were uncovered:

**Feature engineering matters:** the final solution had an important contribution from features that were engineered from the original dataset.

**Data diversity matters:** different data object types have a significant contribution to the overall solution (image, text, numerical, and categorical).

**Complexity can deteriorate consistency:** The most complex models with most hyper-parameters to tune are the most irregular across folds in a cross validation exercise. This was evident in the standard deviation of the performance measures across the 10 folds for the Linear SVC model in classification, and the Multi-layer Perceptron model in regression.

**Complexity is not always better:** Not always the most complex models achieves the best out-of-sample performance. Evidence of this surfaced in the out-of-sample evaluation.

### **Improvement**

This project can be further extended through the following tasks:

* Improved text feature engineering: feature extraction with Tf-idf; sentiment analysis (for example: proportion of *good* and *bad* words in the **_title_** and **_description_** fields).

* Improved image feature engineering: additional image quality criteria (for example: contrast, definition, colors).

* Expand features by using external datasets: city/region population, income, shipping costs (amount and party responsible for them).

## **VI. References**

Dataset: https://www.kaggle.com/c/avito-demand-prediction/data

F1-Score Evaluation Metric: https://en.wikipedia.org/wiki/F1_score

Kaggle Avito Demand Prediction Challenge: https://www.kaggle.com/c/avito-demand-prediction

Root Mean Squared Error Evaluation Metric: https://www.kaggle.com/c/avito-demand-prediction#evaluation 

Image blur metric: https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/