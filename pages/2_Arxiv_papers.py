import numpy as np
import pandas as pd

import streamlit as st
from streamlit_chat import message


from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.document_loaders import ReadTheDocsLoader
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings
import pickle
from langchain.chains import ConversationChain
from langchain.llms import OpenAI,Cohere
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings,CohereEmbeddings
from langchain.chains import ChatVectorDBChain
import pickle
from langchain import OpenAI, VectorDBQA
from langchain.prompts.prompt import PromptTemplate

st.title("""Arxiv Papers """)

   
st.header("The Arxiv Papers")


import streamlit as st
from langchain import PromptTemplate
from langchain.llms import Cohere

template = """
    Below is an email that may be poorly worded.
    Your goal is to:
    - Properly format the email
    - Convert the input text to a specified tone
    - Convert the input text to a specified dialect
    Here are some examples different Tones:
    - Formal: We went to Barcelona for the weekend. We have a lot of things to tell you.
    - Informal: Went to Barcelona for the weekend. Lots to tell you.  
    Here are some examples of words in different dialects:
    - American: French Fries, cotton candy, apartment, garbage, cookie, green thumb, parking lot, pants, windshield
    - British: chips, candyfloss, flag, rubbish, biscuit, green fingers, car park, trousers, windscreen
    Example Sentences from each dialect:
    - American: I headed straight for the produce section to grab some fresh vegetables, like bell peppers and zucchini. After that, I made my way to the meat department to pick up some chicken breasts.
    - British: Well, I popped down to the local shop just the other day to pick up a few bits and bobs. As I was perusing the aisles, I noticed that they were fresh out of biscuits, which was a bit of a disappointment, as I do love a good cuppa with a biscuit or two.
    Please start the email with a warm introduction. Add the introduction if you need to.
    
    Below is the email, tone, and dialect:
    TONE: {tone}
    DIALECT: {dialect}
    EMAIL: {email}
    
    YOUR {dialect} RESPONSE:
"""

prompt = PromptTemplate(
    input_variables=["tone", "dialect", "email"],
    template=template,
)

def copy_list(a):
    b = []
    for i in range(0,4):
        b.append(a[i].page_content)
    return b
def load_vectorstore():
    '''load embeddings and vectorstore'''
           
    embeddings = CohereEmbeddings(cohere_api_key= "vGCEakgncpouo9Nz0rsJ0Bq7XRvwNgTCZMKSohlg")
       
    return FAISS.load_local('arxiv_embeddings', embeddings)

docsearch = load_vectorstore()
def load_data(file_name):
    """
    Load a CSV file as a Pandas DataFrame from the data folder in a GitHub repository.
    """
    url = "https://raw.githubusercontent.com/PrateekKumar2109/Mulitpage-Chatbot/main/Data/Arxiv_data.csv.csv"
    
    df = pd.read_csv(url)
    return df

df=load_data('arxiv')

st.header("Globalize Text")



st.markdown("## Enter your Idea to learn")



col1, col2 = st.columns(2)
with col1:
    option_tone = st.selectbox(
        'Which tone would you like your email to have?',
        ('Formal', 'Informal'))
    
with col2:
    option_dialect = st.selectbox(
        'Which English Dialect would you like?',
        ('American', 'British'))

def get_text():
    input_text = st.text_area(label="Topic", label_visibility='collapsed', placeholder="Your Interest...", key="query")
    return input_text

query= get_text()

if len(query.split(" ")) > 700:
    st.write("Please enter a shorter query. The maximum length is 700 words.")
    st.stop()


#st.button("*See An Example*", type='secondary', help="Click to see an example of the email you will be converting.", on_click=update_text_with_example)

#st.markdown("### Your Converted Email:")

if email_input:
    
    #llm = load_LLM(openai_api_key=openai_api_key)
    list_queries=docsearch.similarity_search(query)
    x_page_content=copy_list(list_queries)
    result = df[df['text_full'].isin(x_page_content)][['title', 'year','authors','abstract']]
    
    col1, col2,col3, col4 = st.columns(4)

    with col1:
         st.write(result.iloc[0])
         st.markdown("Often professionals would like to improve their emails, but don't have the skills to do so. \n\n This tool \
                will help you improve your email skills by converting your emails into a more professional format. This tool \
                is powered by [LangChain](https://langchain.com/) and [OpenAI](https://openai.com) and made by \
                [@GregKamradt](https://twitter.com/GregKamradt). \n\n View Source Code on [Github](https://github.com/gkamradt/globalize-text-streamlit/blob/main/main.py)")

    with col2:
         st.write(result.iloc[1])
    
    with col3:
         st.write(result.iloc[2])
    with col4:
         st.write(result.iloc[3])
    #prompt_with_email = prompt.format(tone=option_tone, dialect=option_dialect, email=email_input)

    #formatted_email = llm(prompt_with_email)

    
