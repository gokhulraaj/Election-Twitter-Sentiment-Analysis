markdown
Copy code
# Indian Election Sentiment Analysis

This project performs sentiment analysis on tweets related to Indian politicians, specifically Narendra Modi and Rahul Gandhi. The analysis uses the TextBlob library to classify tweets into positive, negative, or neutral categories and visualizes the results using Plotly.

## Project Structure

- `modi_reviews.csv`: CSV file containing tweets related to Narendra Modi.
- `rahul_reviews.csv`: CSV file containing tweets related to Rahul Gandhi.
- `sentiment-analysis.py`: Python script for sentiment analysis.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/gokhulraaj/Election-Twitter-Sentiment-Analysis.git
   cd Election-Twitter-Sentiment-Analysis
Install the required packages:

Make sure you have Python installed. Then, install the necessary libraries using:

bash
Copy code
pip install numpy pandas textblob plotly
Usage
Prepare the Data:

Ensure that modi_reviews.csv and rahul_reviews.csv are in the root directory of the project.

Run the Analysis:

Execute the Main.py script to perform sentiment analysis and visualize the results:

bash
Copy code
python Main.py
Alternatively, you can open sentiment-analysis.ipynb in Jupyter Notebook and run the cells interactively to explore the analysis.

Code Overview
Data Loading:

python
Copy code
modi = pd.read_csv("./modi_reviews.csv")
rahul = pd.read_csv("./rahul_reviews.csv")
Sentiment Analysis:

Sentiments are computed using TextBlob to find the polarity of each tweet, and tweets are classified as positive, negative, or neutral.

python
Copy code
def find_polarity(review):
    return TextBlob(review).sentiment.polarity

modi['Polarity'] = modi['Tweet'].apply(find_polarity)
rahul['Polarity'] = rahul['Tweet'].apply(find_polarity)
Data Visualization:

A bar chart displays the percentage of positive and negative sentiments for each politician.

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
The analysis provides insights into public sentiment towards Narendra Modi and Rahul Gandhi based on tweet data. The visualizations depict the distribution of positive and negative sentiments for each politician.

Contributing
Contributions are welcome! Feel free to fork the repository, make changes, and submit a pull request.

License
