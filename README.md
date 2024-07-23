Indian Election Sentiment Analysis
This project analyzes sentiments in tweets related to Indian politicians, specifically Narendra Modi and Rahul Gandhi. Using the TextBlob library, the analysis classifies tweets into positive, negative, or neutral categories. The results are visualized using Plotly.

Project Structure
modi_reviews.csv: CSV file containing tweets related to Narendra Modi.
rahul_reviews.csv: CSV file containing tweets related to Rahul Gandhi.
Main.py: The main script for performing sentiment analysis and visualization.
sentiment-analysis.ipynb: Jupyter Notebook providing an interactive exploration of sentiment analysis.
sentiment-analysis.py: Python script for sentiment analysis.
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/gokhulraaj/Election-Twitter-Sentiment-Analysis.git
cd Election-Twitter-Sentiment-Analysis
Install the required packages:

Ensure you have Python installed, then install the necessary libraries:

bash
Copy code
pip install numpy pandas textblob plotly
Usage
Prepare the Data:

Place the CSV files (modi_reviews.csv and rahul_reviews.csv) in the root directory of the project.

Run the Analysis:

Execute the script to perform sentiment analysis and visualize the results:

bash
Copy code
python Main.py
Alternatively, open sentiment-analysis.ipynb in Jupyter Notebook and run the cells interactively.

Code Overview
Data Loading:

python
Copy code
modi = pd.read_csv("./modi_reviews.csv")
rahul = pd.read_csv("./rahul_reviews.csv")
Sentiment Analysis:

Sentiments are computed using TextBlob to find polarity and classify tweets into positive, negative, or neutral.

python
Copy code
def find_polarity(review):
    return TextBlob(review).sentiment.polarity

modi['Polarity'] = modi['Tweet'].apply(find_polarity)
rahul['Polarity'] = rahul['Tweet'].apply(find_polarity)
Data Visualization:

A bar chart shows the percentage of positive and negative sentiments for tweets about each politician.

python
Copy code
fig = go.Figure(
    data=[
        go.Bar(name='Positive', x=politicians, y=pos_list),
        go.Bar(name='Negative', x=politicians, y=neg_list)
    ]
)
fig.update_layout(barmode='group')
fig.show()
Results
The analysis yields insights into public sentiment regarding Narendra Modi and Rahul Gandhi based on tweet data. The visualizations illustrate the distribution of positive and negative sentiments.



