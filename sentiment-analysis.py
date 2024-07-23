import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import textblob
import numpy as np
import pandas as pd
from textblob import TextBlob
import plotly.graph_objects as go
import plotly.express as xp

modi = pd.read_csv("./modi_reviews.csv")
rahul = pd.read_csv("./rahul_reviews.csv")

modi.head()

modi.shape

rahul.head()

rahul.shape

TextBlob(modi['Tweet'][0]).sentiment

TextBlob(rahul['Tweet'][10]).sentiment

modi['Tweet'] = modi['Tweet'].astype(str)
rahul['Tweet'] = rahul['Tweet'].astype(str)

def find_polarity(review):
    return TextBlob(review).sentiment.polarity

modi['Polarity'] = modi['Tweet'].apply(find_polarity)
rahul['Polarity'] = rahul['Tweet'].apply(find_polarity)

modi

modi['Label'] = np.where(modi['Polarity']>0,'positive','negative')
modi['Label'][modi['Polarity']==0]='Neutral'

rahul['Label'] = np.where(rahul['Polarity']>0,'positive','negative')
rahul['Label'][rahul['Polarity']==0]='Neutral'

neutral_modi = modi[modi['Polarity']==0.0000]
remove_neutral_modi = modi['Polarity'].isin(neutral_modi['Polarity'])
modi.drop(modi[remove_neutral_modi].index,inplace=True)
print(neutral_modi.shape)
print(modi.shape)

neutral_rahul = rahul[rahul['Polarity']==0.0000]
remove_neutral_rahul = rahul['Polarity'].isin(neutral_rahul['Polarity'])
rahul.drop(rahul[remove_neutral_rahul].index,inplace=True)
print(neutral_rahul.shape)
print(rahul.shape)

print(modi.shape)
print(rahul.shape)
np.random.seed(10)
remove_n = 8481
drop_indices = np.random.choice(modi.index,remove_n,replace=False)
df_modi = modi.drop(drop_indices)
np.random.seed(10)
remove_n = 367
drop_indices1 = np.random.choice(rahul.index,remove_n,replace=False)
df_rahul = rahul.drop(drop_indices1)

print(df_modi.shape)
print(df_rahul.shape)

modi_count = df_modi.groupby('Label').count()
neg_modi = (modi_count['Polarity'][0] / 1000) * 100
pos_modi = (modi_count['Polarity'][1] / 1000) * 100


modi_count

rahul_count = df_rahul.groupby('Label').count()
neg_rahul = (rahul_count['Polarity'][0] / 1000) * 100
pos_rahul = (rahul_count['Polarity'][1] / 1000) * 10


rahul_count


politicians = ['Modi','Rahul']

neg_list = [neg_modi,neg_rahul]
pos_list = [pos_modi,neg_modi]


fig = go.Figure(
data = [
    go.Bar(name='Positive',x=politicians,y=pos_list),
    go.Bar(name='Negative',x=politicians,y=neg_list)
]
)
fig.update_layout(barmode='group')
fig.show()
