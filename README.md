# ih_datamadpt1121_project_m3
This Kaggle competition of Ironhack consists of predicting the price of a data set of Diamonds. 

For it, we have to understand the different data sets, prepare and clean the data and find the best model that targets the lowest deviation from the predicted price.

## **STEPS:**
<ol>
  <li>Importing Libraries</li>
  
  <li>Load the Data available</li>

  <li>Inspect it</li>

  <li>Exploratory Data Analysis (EDA)</li>

  <li>Create the train and the test dataframe</li>

  <li>Test different models</li>

  <li>Optimize the hyperparameters of the model chosen</li>

  <li>Make the prediction</li>

  <li>Evaluate the error of the prediction</li>
</ol>

## **1 - Importing Libraries:**
```
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import plotly.express as px
import seaborn as sns

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
```

## **2 - Load the Data available:**

We have two different data sources:

  **A ) DATABASE FILE:** “diamonds_train.db”, which contains 40455 entries to train the model.  

¿How to read a “db file”?. We will use SQLite, a relational database management system, and SQL to read it. 
```
 def conection(database):
    connection = create_engine("sqlite:///"+ database).connect()
    return connection
con = conection("../files/diamonds_train.db")
```

```
def query_diamond(query):
    df_diamonds_full_train = pd.read_sql_query(query,con)
    return df_diamonds_full_train

query_diamonds_full_train = '''
SELECT 
    d_prop.index_id,
    
    d_tran.carat,
    d_cut.cut,
    d_col.color,
    d_clar.clarity,

    d_dim.depth,
    d_dim.'table',
    d_dim.x,
    d_dim.y,
    d_dim.z,
    
    d_city.city,

    d_tran.price
    
FROM diamonds_properties as d_prop
    INNER JOIN diamonds_clarity as d_clar ON d_prop.clarity_id = d_clar.clarity_id
    INNER JOIN diamonds_color as d_col ON d_prop.color_id = d_col.color_id
    INNER JOIN diamonds_cut as d_cut ON d_prop.cut_id = d_cut.cut_id
    INNER JOIN diamonds_dimensions as d_dim ON d_prop.index_id = d_dim.index_id
    INNER JOIN diamonds_transactional as d_tran ON d_prop.index_id = d_tran.index_id
    INNER JOIN diamonds_city as d_city ON d_tran.city_id = d_city.city_id
'''

df_diamonds_train = query_diamond(query_diamonds_full_train)
```

**B ) CSV FILE:** “diamonds_test.csv”, which contains 13485 entries to make the prediction:

¿How to read a CSV file?. With a Pandas method (pd.read_csv())
```
def import_csv_diamonds(location):
    df_diamonds_test = pd.read_csv(location)
    return df_diamonds_test

df_diamonds_test = import_csv_diamonds("../files/diamonds_test.csv")
```

#### FEATURES TO STUDY:
- price in US dollars ($326--$18,823). This will be our the target column. It is only in the tarin dataframe.

**The 4 Cs of Diamonds:**

- carat (0.2--5.01) The carat is the weight of the diamond.  One carat equals 1/5 gram and is subdivided into 100 points. Carat weight is the most important feature of the 4Cs to predict the price. 

- cut (Fair, Good, Very Good, Premium, Ideal) It is the quality of the cut. A well-cut diamond will direct more light through the crown. A diamond with a depth that's too shallow or too deep will allow light to escape through the sides or the bottom of the stone.

- color, from J (worst) to D (best). The diamond’s color ranges from an icy white colorless to a light yellow. Colorless is the most rare and therefore the most expensive. Yellow is the least rare and therefore the least expensive.

- clarity (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best)) The clarity refers to the diamond's tiny markings. Flawless (IF) is the most rare and therefore the most expensive. Included (I) is the least rare and therefore the least expensive.

**Dimensions:**

- x length in mm (0--10.74)

- y width in mm (0--58.9)

- z depth in mm (0--31.8)

**Other Features:**
- depth: total depth percentage = z / mean(x, y) = 2 * z / (x + y) (43--79)

- table: width of top of diamond relative to widest point (43--95)

- city: city where the diamonds is reported to be sold.

## **3 - Inspect our data**

REMEMBER OUR GOAL --> To predict the price, so we are going to study how the different features are correlated with the price. I have use:
- TABLEAU
- SEABORN.

**A) Correlation coeficient:**
```
def correlation_diamonds(df_diamonds_enriched):
    df_diamonds_corr = df_diamonds_enriched.corr().reset_index()
    return df_diamonds_corr
df_diamonds_corr = correlation_diamonds(df_diamonds_train)
```

```
def correlation_visualization(df_diamonds_enriched):
    df_diamonds_corr2 = df_diamonds_enriched.corr()
    sns.set(rc={'figure.figsize':(12,8)})
    return sns.heatmap(df_diamonds_corr2, annot=True);
corr_graf = correlation_visualization(df_diamonds_corr)
```
<p align="center"><img src="https://github.com/alvaro-saez/ih_datamadpt1121_project_m3/blob/main/images/CORRELATION.png"></p>

So we see a clearly positive correlation between carat, x, y, z with the price. Let´s go to see it with Tableau:
<p align="center"><img src="https://github.com/alvaro-saez/ih_datamadpt1121_project_m3/blob/main/images/carat.png"></p>
<p align="center"><img src="https://github.com/alvaro-saez/ih_datamadpt1121_project_m3/blob/main/images/x.png"></p>
<p align="center"><img src="https://github.com/alvaro-saez/ih_datamadpt1121_project_m3/blob/main/images/y.png"></p>
<p align="center"><img src="https://github.com/alvaro-saez/ih_datamadpt1121_project_m3/blob/main/images/z.png"></p>
<p align="center"><img src="https://github.com/alvaro-saez/ih_datamadpt1121_project_m3/blob/main/images/depth.png"></p>
<p align="center"><img src="https://github.com/alvaro-saez/ih_datamadpt1121_project_m3/blob/main/images/table.png"></p>

We are going to use this information to ponder those features giving more importance to carat, x, y and z.

## **4 - Exploratory Data Analysis (EDA)**

We need to prepare our dataframes to fit it to the best model.

**a) Missing and cero values:** The is no any nulls, but we have different raws with "0" in "x", "y" and "z" columns. There are different ways to see it, but we use the next code (we have to reapeat it for each column):

```
df_diamonds_train[df_diamonds_train["x"]<1]
```

For our case, we are going to drop the "Z" column (making different tests it decrease the RMSE oof our predictions).

¿How to deal with the cero values? There are different techniques. We are going to use the cero values in our prediction beacause I have done many tests and the one with the lowest RMSE was using "0" values, but the next picture is the best way for this data frame to deal with those cero values:
<p align="center"><img src="https://github.com/alvaro-saez/ih_datamadpt1121_project_m3/blob/main/images/mean.png"></p>

**b) Drop columns**
**AN ADVISE!**
in a EDA process is very common to drop columns, so you can create a function to save your time. Here you have an example:
```
def delete_column(df, column_name):
    del df[column_name]
    return "column deleted"
```

I have delete the next columns:
 - Z
 - index id
 - id

**c) Remove outliers** In this case, all the raws are usefull for our model, but here you have a function if you want to remove outliers of a normal distribution.
```
def remove_outlier(df_in, col_name):
    q1 = df_in[col_name].quantile(0.30)
    q3 = df_in[col_name].quantile(0.70)
    iqr = q3-q1 #Interquartile range
    fence_low  = q1-1.5*iqr
    fence_high = q3+1.5*iqr
    df_out = df_in.loc[(df_in[col_name] > fence_low) & (df_in[col_name] < fence_high)]
   return df_out
```
**d) Create new columns**: It could be counterproductive, because it is a good practise to remove columns highly correlated (Feature selection"), but for our model (you will see it later) will be usefull.
 - one new feature multiplying "x" and "y"
 - another feature dividing depth and table

**e) Encoding the categorical features**:  I have test the next methods:

- One hot encoding (discarded because of the model peculiarity): using the pd.get_dummmies() method of Pandas

- Label encoding (which give us a better core in our model)
```
X_train_cat[["cut","color","clarity","city"]] = X_train_cat[["cut","color","clarity","city"]].astype("category")

X_train_cat["cut"] = X_train_cat["cut"].cat.codes
X_train_cat["color"] = X_train_cat["color"].cat.codes
X_train_cat["clarity"] = X_train_cat["clarity"].cat.codes
X_train_cat["city"] = X_train_cat["city"].cat.codes
```

**f) Manual ponderation**:  I have tried three different methods of scaling:
- StandardScaler()
- MinMaxScaler()
- RobustScaler()

The reality was it didn´t decrease ed RMSE, so I tried to make a manual ponderation to assign more "power" to carat, x and y by different ways:

 1. multiplying carat x1000
 2. multiplying "x" and" and creating a new column called "xy"
 3. also I have multiply "x" and "y" x10

 The final result is the next dataframe:
 
 <p align="center"><img src="https://github.com/alvaro-saez/ih_datamadpt1121_project_m3/blob/main/images/tablefinal.png"></p>
 
** This process have been applied to both data sources (train and test)**
 
 ## **5 - Create X_train, y_train and X_test**
 
 - X_train --> It has all the features except the Price (also except the id, index_id and z)
 - y_train --> It has only the price (the target for our model)
 - X_test --> It has the same features of X_train, but the source is different (“diamonds_test.csv”)

 ## **6 - Test different models**
 
 The models tested were:
 - linear_model.Lasso()
 - ElasticNet()
 - Ridge()
 - SVR()
 - SGDRegressor()
 - LinearRegression()
 - XGBRegressor
 - RandomForestRegressor()

The best one for our dataframes and our goal was RandomForestRegressor(), beacuase it obtained the lowest RMSE through two different methods.

#### RandomForestRegressor() DEFINITION (source: https://www.geeksforgeeks.org/random-forest-regression-in-python/)

A Random Forest is an ensemble technique capable of performing both regression and classification tasks with the use of multiple decision trees and a technique called Bootstrap and Aggregation, commonly known as bagging. The basic idea behind this is to combine multiple decision trees in determining the final output rather than relying on individual decision trees. 
Random Forest has multiple decision trees as base learning models. We randomly perform row sampling and feature sampling from the dataset forming sample datasets for every model. This part is called Bootstrap.

ADVANTAGES:

- They are able to select predictors automatically.
- They can be applied to regression and classification problems.
- Trees can, in theory, handle both numerical and categorical predictors without having to create dummy variables or one-hot-encoding. In practice, this depends on the implementation of the algorithm that each library has.
- As they are non-parametric methods, it is not necessary that any specific type of distribution be met.
- They generally require much less data cleaning and pre-processing compared to other statistical learning methods (for example, they do not require standardization).
- They are not very influenced by outliers.
- If for some observation, the value of a predictor is not available, despite not being able to reach any terminal node, a prediction can be obtained using all the observations belonging to the last node reached. The accuracy of the prediction will be reduced but at least it can be obtained.
- They are very useful in data exploration, they allow the most important variables (predictors) to be identified quickly and efficiently.
- Thanks to the Out-of-Bag Error, its validation error can be estimated without the need to resort to computationally expensive strategies such as cross-validation. This does not apply in the case of time series.
- They have good scalability, they can be applied to data sets with a high number of observations.

MAIN DESADVANTAGE:

- They are not able to extrapolate outside the range of the predictors observed in the training data.

## **7 - Optimized the hyperparameters of the model chosen**

To can know the best hyperparameters I have used the GridSearchCV() method, chosen n_estimators= 1024,and max_depth= 16:

```
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)

random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth
               }
```

```
rf = RandomForestRegressor()

rf_random = RandomizedSearchCV(estimator = rf, 
                               param_distributions = random_grid, 
                               cv = 5, 
                               verbose=3, 
                               random_state=42, 
                               n_jobs = -1)
# Fit the random search model
rf_random.fit(X_train_cat, y_train_cat)
rf_random.best_params_
```

## **8 - Make the prediction**

```
model = RandomForestRegressor(n_estimators= 1024,
 max_depth= 16)
model.fit(X_train_cat, y_train_cat)
predictions = model.predict(X_test).clip(350,18000)
id_predictions = [i for i in range(0,len(predictions))]
predictions_df = pd.DataFrame({"id":id_predictions , "price":predictions })
predictions_df.head()
predictions_df.to_csv("p3_alvaro_model_diamonds.csv", sep=",", index=False)
```

## **9 - Evaluate the error**

Through 2 different methods:

**a) train_test_split() and RMSE formula:**
Using a 80% for the training and a 20% for the test:

```
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_train_cat, y_train_cat, test_size=0.2, random_state=42)
model2 = RandomForestRegressor()
model2.fit(X_train2, y_train2)
predictions2 = model2.predict(X_test2).clip(350,18000)
```

```
check = pd.DataFrame({'Ground truth':y_test2, 'Predictions':predictions2, 'Diff':y_test2-predictions2})
rmse = mean_squared_error(y_test2, predictions2)**0.5
rmse
```

Plotting a chart with the las 10 predictions we can see how precise it is:
<p align="center"><img src="https://github.com/alvaro-saez/ih_datamadpt1121_project_m3/blob/main/images/graf1.png"></p>

**b) cross-validation method:**
```
scores = cross_val_score(model, 
                         X_train_cat, 
                         y_train_cat, 
                         scoring='neg_root_mean_squared_error', 
                         cv=5,
                         n_jobs=-1)

print(type(model), '\n')
print(scores, '\n')
print(np.mean(-scores), '\n')
```

The result was very similar:

[-541.06307357 -554.78944972 -552.36873579 -596.55870621 -547.44943305] 

558.4458796694024 

# WORK IN PROGRESS
I will update this readme with a way to ampliate our data source, having a higest train dataframe. With this method I obtained a RMSE of 76.

<p align="center"><img src="https://github.com/alvaro-saez/ih_datamadpt1121_project_m3/blob/main/images/graf2.png"></p>

# THANK YOU VERY MUCH
