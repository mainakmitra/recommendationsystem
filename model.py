#!/usr/bin/python

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

# This method is the main method of the model
# It takes input as user id
# With that userid it first recommends top 20 products
# Then it filters top 5 product based on the sentiment analysis

def getModeloutput(userid):
    #print("hello " + userid)
    # load the recommendation model
    reco = pickle.load(open('pickle/final_Recommendation_UU.pkl', 'rb'))
    user_items = reco.loc[userid].sort_values(ascending=False)[0:20]
    # Load the dataset
    df = pd.read_csv('dataset/sample30.csv')
    item_mapping = pd.merge(user_items, df,left_on='name',right_on='name', how = 'inner')
    x = item_mapping.reviews_text
    # load the TFIDF vectorizer 
    tfidf_vec = pickle.load(open('pickle/final_TF_IDF_vec.pkl', 'rb'))
    x_transformed = tfidf_vec.transform(x)
    # load the Sentiment analysis model
    lg_model = pickle.load(open('pickle/final_LG_SentimentAnalysis.pkl', 'rb'))
    y_pred= lg_model.predict(x_transformed)
    # get the top5 products
    df_concat = pd.concat([item_mapping,pd.Series(y_pred).rename('prediction')],axis = 1)
    new_df = pd.crosstab(df_concat.name,df_concat.prediction)
    new_df = new_df.add_prefix('tot ').reset_index().rename_axis(None, axis=1)
    new_df['positive_percent'] = new_df['tot 1']/(new_df['tot 0'] + new_df['tot 1'])*100
    final_df = new_df.sort_values(by=['positive_percent'], ascending=False)
    return final_df.head(5)




