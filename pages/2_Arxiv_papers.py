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

st.title("""Arxiv Papers 📝 """)

   



import streamlit as st
from langchain import PromptTemplate
from langchain.llms import Cohere

template = """
    Below is  the abstract of a research papers. You need to summarize the abstract and 
    write the summary in {tone} format . Write  the summary  as {expertise} years old considering the following abstract 
    ABSTRACT: {abstract}
    Tone:{tone}
    YOUR Summary :
"""

prompt = PromptTemplate(
    input_variables=["tone", "expertise", "abstract"],
    template=template,
)

#qa=VectorDBQA.from_chain_type(llm=Cohere(model="command-xlarge-nightly", cohere_api_key="vGCEakgncpouo9Nz0rsJ0Bq7XRvwNgTCZMKSohlg",
#                                         temperature=0.7),k=1,vectorstore=docsearch, return_source_documents=False)
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


#st.markdown("## Enter your Idea to learn")



col1, col2,col3 = st.columns(3)
with col1:
    option_tone = st.selectbox(
        'Which format you want to have your summarization?',
        ('Bullet point', 'Paragraph','Linkedin post'))
    
with col2:
    option_expert = st.selectbox(
        'Which type of Summarization would you like?',
        ('Basic for  6', 'Medium for 18', 'Expert for 30'))

with col3:
    option_abstract = st.selectbox(
        'Which paper would you like to summarize?',
        ('1','2','3','4'))
def get_text():
    input_text = st.text_area(label="Topic", label_visibility='collapsed', placeholder="Your Interest...", key="query")
    return input_text

query= get_text()

if len(query.split(" ")) > 700:
    st.write("Please enter a shorter query. The maximum length is 700 words.")
    st.stop()


#st.button("*See An Example*", type='secondary', help="Click to see an example of the email you will be converting.", on_click=update_text_with_example)

#st.markdown("### Your Converted Email:")
llm=Cohere(model="command-xlarge-nightly", cohere_api_key="vGCEakgncpouo9Nz0rsJ0Bq7XRvwNgTCZMKSohlg",
                                         temperature=0.7)
if query:
    
    #llm = load_LLM(openai_api_key=openai_api_key)
    list_queries=docsearch.similarity_search(query)
    x_page_content=copy_list(list_queries)
    result = df[df['text_full'].isin(x_page_content)][['title', 'year','authors','abstract']]
    st.dataframe(result,900,200)
    #col1, col2,col3, col4 = st.columns(4)
    abstract_input=x_page_content[int(option_abstract)-1]
    
    #with col1:
         #st.write(result.iloc[0])
        
    #with col2:
    #     st.write(result.iloc[1])
    
    #with col3:
     #    st.write(result.iloc[2])
    #with col4:
     #    st.write(result.iloc[3])
    
    prompt_with_email = prompt.format(tone=option_tone, expertise=option_expert, abstract=abstract_input)

    formatted_email = llm(prompt_with_email)
    st.markdown("### Your Converted Abstract📃:")
    st.write(formatted_email)
    
