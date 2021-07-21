#!/usr/bin/env python
# coding: utf-8

# https://pythonforundergradengineers.com/streamlit-app-with-bokeh.html

#link
#https://www.analyticsvidhya.com/blog/2020/12/streamlit-web-api-for-nlp-tweet-sentiment-analysis/
#https://www.youtube.com/watch?v=SIu2VL-RAXc&list=PLJ39kWiJXSixyRMcn3lrbv8xI8ZZoYNZU&index=6
# https://www.youtube.com/watch?v=bEOiYF1a6Ak&list=PLJ39kWiJXSixyRMcn3lrbv8xI8ZZoYNZU&index=9

from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.feature_extraction.text import TfidfVectorizer

import streamlit as st

from joblib import load

#load modal
model_age = load("lr_age_model")
model_status = load("nb_status_model")
vectorizer = CountVectorizer()


# load EDA pkgs
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import seaborn as sns
sns.set_style('whitegrid')
import numpy as np

#Wordcloud
from wordcloud import WordCloud
from PIL import Image

# frequent
import nltk
from nltk import FreqDist

# display data
df = pd.read_csv("data/clean_data.csv")
#df = pd.read_csv("eda_data.csv")
df = df.sample(frac=0.2)

#numeric_col = df.select_dtypes(['int32', 'int64', 'float32', 'float64']).columns.tolist()



#method 1
#st.dataframe(df)
#import re
# function
@st.cache 
#age
def predict_text_age(text):
    results = model_age.predict([text])
    return results

def prob_text_age(text):
    results = model_age.predict_proba([text])
    return results
#marital status
def predict_text_status(text):
    results = model_status.predict([text])
    return results

def prob_text_status(text):
    results = model_status.predict_proba([text])
    return results

# wordcloud
def plot_wordcloud(corpus, max_words=150, max_font_size=35):
            wordcloud = WordCloud(collocations=True,
                                  background_color='black', 
                                  max_words=150,
                                  max_font_size=35, 
                                  )
            wordcloud.generate(str(corpus))
            fig, ax = plt.subplots(figsize=(10,10))
            plt.axis('off')
            plt.tight_layout()
            plt.imshow(wordcloud, cmap=None)
            #plt.imshow(wordcloud, interpolation = "bilinear")
            st.pyplot(fig)
            
def frequent(text, number = 30, figsize=(10,7)):
    tokens = nltk.tokenize.word_tokenize(','.join(map(str, text)))
    freq = FreqDist(tokens)
    #display(freq.most_common(number))
    most_common = pd.DataFrame(freq.most_common(number),
                           columns=['word','count']).sort_values('count',
                                                                 ascending=True)
    #plot
    fig, ax = plt.subplots(figsize=figsize)
    most_common.set_index('word').tail(25).plot(kind='barh',ax=ax)
    ax.set(xlabel=None, ylabel=None, title="Most frequent words")
    ax.grid(axis="x")
    st.pyplot(fig)
    
    


def main():
    
    st.title("Text Analysis")
    menu = ["Home", "EDA", "About"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    # Home
    if choice == "Home":
        #st.subheader("Home-Text")
        #st.success("Hello")
        
        # search to use later
        #search = st.text_input("Search")
        #with st.beta_expander("results"):
            #retrieved_df = df[df["clean"].str.contains(search)]
            #st.dataframe(retrieved_df[["Agegroup", "status", "clean"]])
                               
                               
        
        with st.form(key = "form"):
            raw_text = st.text_area("write...")
            column1, column2 = st.beta_columns(2)
            with column1:
                #st.success("Text")
                #st.write(raw_text)
                submit = st.form_submit_button(label = "Age")
            with column2:
                submit1 = st.form_submit_button(label = "Status")
            
            
        if submit:
            column1, column2 = st.beta_columns(2)
            
            prediction = predict_text_age(raw_text)
            prob = prob_text_age(raw_text)
            
            with column1:
                #st.success("Text")
                #st.write(raw_text)
               
                st.success("Predict")
                result = ['under 35' if prediction == 0 else 'over 35' for prediction in prediction]
                st.write(result[0])
                
            with column2:
                st.success("Prediction Prob")
                st.write(prob)
                
                
              
            
        if submit1:
            
            #selectbox_1 = st.checkbox("Age")
            column1, column2 = st.beta_columns(2)
            # apply function
            prediction = predict_text_status(raw_text)
            prob = prob_text_status(raw_text)

            with column1:
                st.success("Text")
                st.write(raw_text)

                st.success("Predict")
                result = ['single' if prediction == 0 else 'married' for prediction in prediction]
                st.write(result[0])


            with column2:
                st.success("Prediction Prob")
                st.write(prob)

        st.success("WordCloud And Frequent")
        
        
        
        
        

        #st.sidebar.subheader("Create plot")
        
        
        
        # add select widget 
        data = df[["Agegroup", "status"]]
        selectbox_1 = st.sidebar.selectbox(label = "Age & Marital Status", options = data.columns)
        
        #Age
        if selectbox_1 == "Agegroup":
        
                # wordcloud 
            column1, column2 = st.beta_columns(2) 
            with column1:

                with st.beta_expander("WordCloud"):
                    for i in df["Agegroup"].unique():
                        st.write("******** {} ********".format(i))
                        plot_wordcloud(corpus=df[df["Agegroup"]==i]["clean"],
                                       max_words=150, max_font_size=35)

            with column2:        
                with st.beta_expander("Frequent"):
                    for i in df["Agegroup"].unique():
                        st.write("********* {}:  ************".format(i))
                        frequent(df[df["Agegroup"]==i]["clean"],  20)
                     
        #status                
        elif selectbox_1 == "status":
            
            column1, column2 = st.beta_columns(2) 
            with column1:

                with st.beta_expander("WordCloud"):
                    for i in df["status"].unique():
                        st.write("******** {} ********".format(i))
                        plot_wordcloud(corpus=df[df["status"]==i]["clean"],
                                       max_words=150, max_font_size=35)

            with column2:        
                with st.beta_expander("Frequent"):
                    for i in df["status"].unique():
                        st.write("********* {}:  ************".format(i))
                        frequent(df[df["status"]==i]["clean"],  20)
        st.write("select agegroup or marital status to see \n worldCould and frequent word")

    
    
    # EDA
    
    elif choice == "EDA":
        st.subheader("EDA")
        st.title("EDA")
        
       
        
        # create plot
        st.sidebar.subheader("Create plot")
        
        # add select widget
        selectbox_1 = st.sidebar.selectbox(label = "Feature", options = df.columns)
        fig, ax = plt.subplots()
        g = sns.countplot(x = df[selectbox_1], hue = "Agegroup", data= df)
        g.set_xticklabels(ax.get_xticklabels(),rotation = 45, fontsize = 12,  ha="right")
        st.pyplot(fig)
       
        # create hist
        #fig, ax = plt.subplots()
        #st.sidebar.subheader("hist")
        #selectbox_2 = st.sidebar.selectbox(label = "Y axis", options = df.columns)
        #selectbox_3 = st.sidebar.selectbox(label = "X axis", options = df.columns)
        #hist_slider = st.sidebar.slider(label ="Number of bins", min_value = 2, max_value =100, value = 20)
        
        #sns.distplot(df[selectbox_3], hist_slider)
        #st.pyplot(fig)
        
    # About  
    else:
        st.subheader("About")
    
    
    
    
if __name__ == '__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    

