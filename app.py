# dependencies
from flask import Flask, render_template, redirect
from flask_pymongo import PyMongo
import scrape_mars

# create the flask app
app = Flask(__name__, template_folder='templates')

# declare the database
app.config["MONGO_URI"] = "mongodb://localhost:27017/mars_db"
mongo = PyMongo(app)

# the base route
@app.route('/')
def home():
    mars = mongo.db.mars_facts.find_one()
    return render_template('index.html', mars=mars)

# the scraping route, which redirects to the base route
@app.route('/scrape')
def scraper():
    mars_facts = mongo.db.mars_facts
    mars_scrape = scrape_mars.scrape()
    mars_facts.update({}, mars_scrape, upsert=True)
    return redirect("/")

if __name__ == "__main__":
    app.run(debug=True)