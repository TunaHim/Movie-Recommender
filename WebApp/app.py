
# Import Dependencies

from flask import Flask,render_template,request
from recommender import recommend_random, recommend_with_NMF, recommend_neighborhood
from utils import movies

app = Flask(__name__)

@app.route('/')
def hello():
    return render_template('index.html', name='Film', movies=movies.title.to_list())

@app.route('/movies')
def recommendation():
    print(request.args)

    titles = request.args.getlist('title')
    ratings = request.args.getlist('rating')

    query = dict(zip(titles,ratings))

    for movie in query:
        query[movie] = float(query[movie])

    if request.args.get('option') =='Random':
        recommendation_list = recommend_random()
        print(recommendation_list)

        return render_template('recommend.html', recommendation=recommendation_list)
    
    if request.args.get('option')=='NMF':
        
        recommendation_list = recommend_with_NMF(query)
        return render_template('recommend.html', recommendation=recommendation_list)
    
    else:
        recommendation_list = recommend_neighborhood(query)
        return render_template('recommend.html', recommendation=recommendation_list)

if __name__=='__main__':
    app.run(port=5000,debug=True)
    
