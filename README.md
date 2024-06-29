# MovieLens-Data-Analytics-and-Recommendation-System-Simulation

## Introduction
In this notebook, we will use an Alternating Least Squares (ALS) algorithm with Spark APIs to predict the ratings for the movies in MovieLens small dataset.

## Data and Code
The dataset source is [here](https://grouplens.org/datasets/movielens/latest/). For the code, you can download the html file and open it in the browser or you could open by this [link](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/1772353219017266/1167986716657297/105392983207357/latest.html).

## Notes
- About ALS: 

## Summary
- Built data ETL pipeline to analyze movie rating dataset and conducted online analytical processing (OLAP) with Spark SQL.
- Used Spark ML to predict ratings, leveraging Alternating Least Square (ALS) algorithm (matrix factorizarion based) for a large-scale dataset with resulting RMSE = 0.65.
- Utilized above trained model to provide personalized movie recommendation (top k liked) and developed user-based approaches to handle system cold-start problems. 
