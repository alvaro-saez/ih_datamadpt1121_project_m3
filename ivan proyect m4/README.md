# Diamond Price Predictor

In this project, a Supervised Machine Learning algorithm has been built, which will allow us to predict the price of a Diamond.

![Image](https://www.genesisdiamonds.net//media/wysiwyg/DIAMOND-BUYING-PAGE-banner.jpg)

---
## **Description**
Given a certain features of a Diamond, the model will predict the sales price of that one.

To better Diamonds understanding as well as data cleaning process, check my [previous project](https://github.com/ivanrepi/data_visualization_project_m2) !

&nbsp;
## **Diamonds Dataset** :

The price of a diamond has a direct correlation with its carat. It is not a straight linear correlation but an exponential one. There are other relevant features which also influence its price, such as color, clarity and cut.

<p align="center"><img src="https://i.imgflip.com/3l8sg9.jpg"></p>


This classic dataset contains the prices and other attributes of almost 44,000 diamonds. It's a great dataset for machine learning and work with data analysis and visualization.

Dataset Columns :
- **price** : price in US dollars ( $326 - $18,823 )

- **carat** : weight of the diamond ( 0.2 - 5.01 )

- **cut** : quality of the cut ( Fair , Good , Very Good , Premium , Ideal )

- **color** : diamond colour, from J ( worst ) to D ( best )

- **clarity** : a measurement of how clear the diamond is ( I1 (worst) ,SI2 ,SI1 ,VS2 ,VS1 ,VVS2 ,VVS1 ,IF (best))

- **x** : length in mm ( 0 - 10.74 )

- **y** : width in mm ( 0 - 58.9 )

- **z** : depth in mm ( 0 - 31.8 )

- **depth** : total depth percentage = z / mean( x, y ) = 2 * z / ( x + y ) --> ( 43 - 79 )

- **table** : width of top of diamond relative to widest point ( 43 - 95 )

![Image](https://4cs.gia.edu/wp-content/uploads/2016/08/PUBL160028_SC_4Cs_Spokes_990x500_ENG_v1.jpg)


---
## **Getting Started**
### :baby: **Status**
This is part of Kaggle Competition. The main goal is to build a complete ML model.

The EDA (Exploratory Data Analysis) notebook is located [here](https://github.com/ivanrepi/predict_diamond_prices_project_m3/blob/master/notebooks/EDA.ipynb)

The main study of this project (model, test, train, predicition, etc.) is located [here](https://github.com/ivanrepi/predict_diamond_prices_project_m3/blob/master/notebooks/diamond_price_prediction.ipynb)

### :computer: **Dependencies**

- This repository is tested on **Python 3.7+**.
- Create a virtual environment with the version of Python you're going to use and activate it.

- Install [pandas](https://pandas.pydata.org/docs/user_guide/index.html) library. Copy and paste next command in your master branch:
    ```
    conda install pandas
    ```
- Install [scikit-learn](https://scikit-learn.org/stable/) library. Copy and paste next command in your master branch:
    ```
    pip install -U scikit-learn
    ```
- Install [XGBoost](https://xgboost.readthedocs.io/en/stable/) library. Copy and paste next command in your master branch:
    ```
    pip install xgboost
    ```
- Install [matplotlib](https://matplotlib.org/stable/index.html) library. Copy and paste next command in your master branch:
    ```
    pip install matplotlib
    ```
- Install [seaborn](https://seaborn.pydata.org/) library. Copy and paste next command in your master branch:
    ```
    pip install seaborn
    ```



&nbsp;

---


### :clipboard: **Modelling with XGBoost**
After several tests (which are located in the discarded folder) the model with a slower RMSE result is XGBoost one.

**XGBoost**, which stands for Extreme Gradient Boosting, is a scalable, distributed gradient-boosted decision tree (GBDT) machine learning library. It provides parallel tree boosting and is the leading machine learning library for regression, classification, and ranking problems.

It’s vital to an understanding of XGBoost to first grasp the machine learning concepts and algorithms that XGBoost builds upon: supervised machine learning, decision trees, ensemble learning, and gradient boosting.

Supervised machine learning uses algorithms to train a model to find patterns in a dataset with labels and features and then uses the trained model to predict the labels on a new dataset’s features.

![image](https://www.nvidia.com/content/dam/en-zz/Solutions/glossary/data-science/xgboost/img-1.png)

### :bar_chart: **Scale**
Machine learning models learn a mapping from input variables to an output variable.

As such, the scale and distribution of the data drawn from the domain may be different for each variable.

Input variables may have different units (e.g. feet, kilometers, and hours) that, in turn, may mean the variables have different scales.

**Standardizing** a dataset involves rescaling the distribution of values so that the mean of observed values is 0 and the standard deviation is 1.

This can be thought of as subtracting the mean value or centering the data.


### :wrench: **Label**
Data labeling in Machine Learning (ML) is the process of assigning labels to subsets of data based on its characteristics. Data labeling takes unlabeled datasets and augments each piece of data with informative labels or tags. 

Next image represents a simple example about what label is:

<p align="center"><img src="https://miro.medium.com/max/386/1*Yp6r7m82IoSnnZDPpDpYNw.png"></p>

### :diamond_shape_with_a_dot_inside: **Train & Test**
The train-test split procedure is used to estimate the performance of machine learning algorithms when they are used to make predictions on data not used to train the model.

It is a fast and easy procedure to perform, the results of which allow you to compare the performance of machine learning algorithms for your predictive modeling problem. Although simple to use and interpret, there are times when the procedure should not be used, such as when you have a small dataset and situations where additional configuration is required, such as when it is used for classification and the dataset is not balanced.

![image](https://miro.medium.com/max/1400/1*-NC7sX7kzjJ_UKfUNYbh6Q.png)

### :diamond_shape_with_a_dot_inside: **RMSE**
RMSE: Root Mean Square Error is the measure of how well a regression line fits the data points. RMSE can also be construed as Standard Deviation in the residuals.

In this prediction, the RMSE is **522,06**.

&nbsp;



---

### :file_folder: **Folder structure**
```
└── project
    ├── .git
    │ 
    ├── .gitignore
    │ 
    ├── README.md
    │ 
    ├── main_user.py
    │ 
    ├── notebooks
    │    ├── discarded
    │    ├── diamond_price_prediction.ipynb
    │    └── EDA.ipynb
    │
    └── data
        ├── raw
        ├── train
        ├── test
        └── prediction
```



&nbsp;

---


