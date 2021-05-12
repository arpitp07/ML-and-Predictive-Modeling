# %% [markdown]
#  # Machine Learning and Predictive Modeling - Assignment 6
#  ### Arpit Parihar
#  ### 05/11/2021
#  ****
# %% [markdown]
#  **Importing modules**

# %%
import numpy as np
import pandas as pd

from sklearn.metrics import pairwise_distances

import joblib
import warnings
warnings.filterwarnings('ignore')

# %% [markdown]
# **Importing data**

# %%
data = pd.read_csv('radio_songs.csv')
data.set_index('user', inplace = True)

# %% [markdown]
#  ### 1\. Collaborative Filtering
# 
#  Use this user-item matrix to:
# 
#  **A. Recommend 10 songs to users who have listened to 'u2' and 'pink floyd'. Use item-item collaborative filtering to find songs that are similar using spatial distance with cosine. Since this measures the distance you need to subtract from 1 to get similarity as shown below.**
# 
# %% [markdown]
#  Creating a column for users who've listened to both Pink Floyd and U2

# %%
data['u2 and pink floyd'] = data['u2'] & data['pink floyd']
print(f'Number of users who have listened to both U2 and Pink Floyd = {sum(data["u2 and pink floyd"])}')

# %% [markdown]
#  There are no users who have listened to both Pink Floyd and U2. We'll check for users who've listened to either U2 **or** Pink Floyd.

# %%
data['u2 or pink floyd'] = data['u2'] | data['pink floyd']
print(f'Number of users who have listened to U2 or Pink Floyd = {sum(data["u2 or pink floyd"])}')

# %% [markdown]
#  Taking transpose and calculate pairwise cosine distance b/w each band

# %%
data_T = data.T
item_cosine_matrix = pd.DataFrame(1 - pairwise_distances(data_T , metric='cosine'), index=data_T.index, columns=data_T.index)


# %%
print('10 recommendations for listeners of U2 or Pink Floyd:\n')
item_cosine_matrix.drop(index=['u2', 'pink floyd', 'u2 or pink floyd'])['u2 or pink floyd'].nlargest(10)

item_cosine_matrix.drop(index=['u2 and pink floyd', 'u2 or pink floyd'], columns=['u2 and pink floyd', 'u2 or pink floyd'], inplace=True)
data.drop(columns=['u2 and pink floyd', 'u2 or pink floyd'], inplace=True)
data_T.drop(index=['u2 and pink floyd', 'u2 or pink floyd'], inplace=True)

# %% [markdown]
#  **B\. Find user most similar to user 1606. Use user-user collaborative filtering with cosine similarity. List the recommended songs for user 1606 (Hint: find the songs listened to by the most similar user).**

# %%
user_cosine_matrix = pd.DataFrame(1 - pairwise_distances(data, metric='cosine'), index=data.index, columns=data.index)

sim_user_1606 = user_cosine_matrix.drop(index=[1606])[1606].nlargest(1).index[0]

print('Most similar user to user 1606:\n')
sim_user_1606

rec_1606 = pd.DataFrame(data_T[sim_user_1606][data_T[sim_user_1606] == 1].index, columns=['Recommended'])

print('Recommended bands for user 1606:\n')
rec_1606

# %% [markdown]
#  **C\. How many of the recommended songs has already been listened to by user 1606?**

# %%
print('Recommended bands already listened to by user 1606:\n')
[x for x in data_T.index[data_T[1606]==1] if x in list(rec_1606['Recommended'])]

# %% [markdown]
#  **D\. Use a combination of user-item approach to build a recommendation score for each song for each user using the following steps for each user-**
# 
#  - 1\. For each song for the user row, get the top 10 similar songs and their similarity score.
# 
#  - 2\. For each of the top 10 similar songs, get a list of the user purchases
# 
#  - 3\. Calculate a recommendation score as follows:
#  $\sum(purchaseHistory.similarityScore)/\sum similarityScore$

# %%
try:
    rec_scores = joblib.load('rec_scores.pkl')
except:
    rec_scores = pd.DataFrame(index=data.index, columns=data_T.index)
    for i in range(rec_scores.shape[0]):
        for j in range(rec_scores.shape[1]): 
            user = rec_scores.index[i] 
            band = rec_scores.columns[j]
            if data.iloc[i, j] == 1: 
                rec_scores.iloc[i, j] = 0 
            else: 
                sim_bands = item_cosine_matrix.drop(index=[band])[band].nlargest(10)
                history = data.loc[user, sim_bands.index]
                rec_scores.iloc[i, j] = sum(history*sim_bands)/sum(sim_bands)
    rec_scores.fillna(0, inplace=True)
    joblib.dump(rec_scores, 'rec_scores.pkl')

rec_scores

# %% [markdown]
#  - 4\. What are the top 5 song recommendations for user 1606?

# %%
print('Recommended bands for user 1606:\n')
pd.DataFrame(rec_scores.loc[1606, :].nlargest(5).index, columns=['Recommended'])

# %% [markdown]
# ### 2\. Conceptual questions:
# 
# **1. Name 2 other similarity measures that you can use instead of cosine similarity above.**  
# **Jaccard similarity** and **\(1 - Euclidean distance\)** could have been used instead of cosine similarity.
# 
# **2. What is needed to build a Content-Based Recommender system?**  
# Content-based recommender system circumvents the cold start problem encountered in traditional recommenders, but it needs the items broken down and scored by as many attributes as possible to provide good recommendations by matching users to attributes. Model based approaches can work, but the interpretability in recommendations is lost, and it's not ideal.
# 
# **3. Name 2 methods to evaluate your recommender system.**  
# - A traditional method to evaluate a recommendation system is to check **precision and recall @ k**, which means, of the k recommendations made, how many were correct, and how many of the correct recommendations were captured in k respectively
# - If the order of recommendations is important in our recommender system, **Normalized Discounted Cumulative Gain \(nDCG\)** can be used for evaluation.

