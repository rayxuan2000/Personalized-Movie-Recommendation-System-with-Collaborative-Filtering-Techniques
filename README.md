# Personalized Movie Recommendation System with Collaborative Filtering Techniques

## Introduction
In this notebook, we will use an Alternating Least Squares (ALS) algorithm with Spark APIs to predict the ratings for the movies in MovieLens small dataset.

## Data and Code
The dataset source is [here](https://grouplens.org/datasets/movielens/latest/). For the code, you can download the html file and open it in the browser or you could open by this [link](https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/1772353219017266/1167986716657297/105392983207357/latest.html).

## Notes

 1. About matrix factorization and ALS:
   - In the case of collaborative filtering, matrix factorization algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices. One matrix can be seen as the user matrix where rows represent users and columns are latent factors. The other matrix is the item matrix where rows are latent factors and columns represent items. （Imagine it in your mind）.
   - With matrix factorization, less-known movies can have rich latent representations as much as popular movies have, which improves recommender’s ability to recommend less-known movies.
   - Alternating Least Square (ALS) is also a matrix factorization algorithm (solving loss function) and it runs itself in a parallel fashion. ALS alternates between fixing one matrix and solving for the other. ALS is doing a pretty good job at solving scalability and **sparseness** of the Ratings data, and it’s simple and scales well to very large datasets. Specifically: First, it fixes the item factors and solves for the user factors using least squares optimization. Then, it fixes the user factors and solves for the item factors using least squares optimization. Based on some notes, use non-null value in original user-item matrix to train the model. Numbers in user matrix and item matrix could be random at first. After training, multiply back to get the missing value.
 2. Als parameter tuning is important, for example: *rank* denote the number of latent factors in the model. ALS has APIs in spark to automatically recommend movies. Each movie will be represented by a itemfactor (vector) in the end with length *rank*.
 3. When getting similar movies, I used three metrics: Euclidean distance, cosine similarity and LSH(Locality-sensitive hashing). Locality-Sensitive Hashing (LSH) is used in recommendation systems to efficiently find similar items or users. LSH is particularly useful for large-scale data, where calculating pairwise similarities directly would be computationally expensive. LSH is a technique for hashing high-dimensional data points so that similar points map to the same hash bucket with high probability. This allows for efficient approximate nearest neighbor search.
 4. About autoencoder: First do indexing for user and movie (actually encoding from category to numerical), convert it to sparse matrix (row denotes user [each input] and column denotes item), build autoencoder (the encoder reduces the dimensionality of the input, while the decoder reconstructs the input from the encoded representation, with data normalization before, encoding_dim is a hyperparameter), train, do recommendation.

Original data:
```
+------+-------+------+
|userId|movieId|rating|
+------+-------+------+
|     1|      1|   4.0|
|     1|      3|   4.0|
|     1|      6|   4.0|
|     1|     47|   5.0|
|     1|     50|   5.0|
+------+-------+------+
```

Do indexing:
```
+------+-------+------+---------+----------+
|userId|movieId|rating|userIndex|movieIndex|
+------+-------+------+---------+----------+
|     1|      1|   4.0|    111.0|      11.0|
|     1|      3|   4.0|    111.0|     422.0|
|     1|      6|   4.0|    111.0|     129.0|
|     1|     47|   5.0|    111.0|      15.0|
|     1|     50|   5.0|    111.0|      14.0|
+------+-------+------+---------+----------+
```

Convert to a sparse matrix:
```
movieIndex	0	  1	  2	  3	  4	  5	  6	  7	  8	  9	 ...	9714	9715	9716	9717	9718	9719	9720	9721	9722	9723
userIndex																					
0	        5.0	5.0	5.0	4.0	5.0	5.0	4.0	5.0	5.0	4.0	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
1	        3.5	4.0	5.0	3.0	5.0	5.0	4.0	3.5	4.5	0.0	...	0.0	0.0	0.0	0.0	3.0	0.0	0.0	0.0	0.0	0.0
2	        3.0	5.0	4.0	4.5	4.5	4.0	4.5	3.0	4.0	5.0	...	3.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
3	        3.0	0.0	5.0	5.0	2.0	5.0	3.0	0.0	3.0	0.0	...	0.0	2.0	3.0	0.0	0.0	0.0	0.0	3.0	0.0	3.0
4	        4.5	4.5	5.0	4.0	4.0	3.0	3.5	4.5	4.5	4.0	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
...	      ...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
605	      0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
606	      0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
607	      3.0	0.0	5.0	0.0	0.0	0.0	4.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
608	      0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
609	      0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	...	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0	0.0
```

Notice that this matrix is sparse, each user vector (horizontal) or item vector (vertical) is also sparse. If we want to build an item-based(user-based) autoencoder, then we take input as each partially observed item vector(user vector), project it into a low-dimensional latent space, and then reconstruct in the outer space to predict missing values for purposes of recommendation. Also pay attention! **We account for the fact that each input is partially observed by only updating during backpropagation those weights that are associated with observed inputs, as is common in matrix factorisation approaches.**

## Summary
- Built data ETL pipeline with parameterized code to analyze movie rating dataset for personalized recommendation;
conducted OLAP with Spark SQL APIs to summarize data at various levels of granularity.
- Leveraged Spark ML APIs to predict ratings using ALS matrix factorization machine, effectively addressing sparsity
issue and uncovering latent information; developed an autoencoder structure to achieve more precise outcomes.
- Employed the trained model with 3 customized distance metrics to group similar items and developed item-based ap-
proaches to tackle user cold-start problems; reduced rating RMSE loss by 20% with resulting RMSE = 0.65.
