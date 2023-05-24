# Movie-Recommender
Unsupervised learning

![MovieRecommender-screenshot](https://github.com/TunaHim/Movie-Recommender/assets/122125136/e48780ae-3748-43df-8a17-2b86bb456e75)

    Based on user's movie choices recommendations are made by this recommender system. There are three ways recommendations are made namely Random, NMF (Non-Negative Matrix factorization) and Neighbourhood based (Cosine Similarity method). 
The 'WebApp/app.py' need to run ($ python app.py) in the terminal which generates a http address. Open this http address in a web browser and get favorite recommendations.  Below are the description of the files and the file system. The file structure needs to remain the same for the Recommender to work. 
## File structure:
#### 'EDA.ipynb' file:
Exploratory data analysis is done on a movie dataset. Data analysis is done to explore number of unique users, movies, duplicate entries, average rating, top-rated movies, etc. Accessory file generation are done here and saved in savedFiles folder.
#### 'savedFiles' folder:
nmf_MovieModel.pkl is the NMF model; mean_movies.pkl mean of the movie rating by all users; ratmov.csv is the merged movies and rating file. All the files are generated in EDA.ipynb script and are required by recommender.py script.

#### 'WebApp' folder:
+ 'data' folder: Contains the csv file for the movies called movies.csv, ratings.csv and a README file downloaded from https://grouplens.org/datasets/movielens/. This is the movie dataset.
+ 'static' folder: style.css & background pictures for the website.
+ 'templates' folder: index.html & recommend.html are used in the website design.
+ 'app.py' file: Run this python file in a terminal which will generates a http address. Open the http address in a web browser.
+ 'recommender.py' file: Contains various recommendation implementations of all three algorithms and returns a list of recommended movies.
+ 'util.py' file: Contains supporting functions.
