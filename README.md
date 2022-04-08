# ih_datamadpt1121_project_m3
This Kaggle competition of Ironhack consists of predicting the price of a data set of Diamonds. 

For it, we have to understand the different data sets, prepare and clean the data and find the best model that targets the lowest deviation from the predicted price.

## **STEPS:**
<ol>
  <li>Load the Data available</li>

  <li>Inspect it</li>

  <li>Exploratory Data Analysis (EDA)</li>

  <li>Create the train and the test dataframe</li>

  <li>Test different models</li>

  <li>Optimized the hyperparameters of the model chosen</li>

  <li>Make the prediction</li>

  <li>Evaluate the error of the prediction</li>
</ol>


## **1 - Load the Data available:**

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
