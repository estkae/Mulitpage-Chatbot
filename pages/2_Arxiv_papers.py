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
#st.title("""PaperPro: Your Assistant 📝 """)
   



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

template_1 = """
    You are given two abstract A and abstract B. If Abstract A and Absctract B are same reply that abstracts are same.
    If the abstract A and abstract B are different then compare and highlight differences step by step in terms of content and style which 
    are given below: 
    
    ABSTRACT A: {abstract_a}
    
    ABSTRACT B: {abstract_b}
    YOUR Comparisons Step by step:\n
"""
prompt_1 = PromptTemplate(
    input_variables=["abstract_a", "abstract_b"],
    template=template_1,
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


#st.sidebar.header('Version:3.0.0')
st.sidebar.header('Arxiv Papers')

st.sidebar.write("Arxiv Papers is an open-source data created by the University of Cornell 🎓 that provides access to a wide range of research papers 📝. Our platform specializes in artificial intelligence ,  natural language processing , prompt engineering, and energy-related papers ⚡.")
st.sidebar.header('How to Use')
st.sidebar.write("Our app is designed to help you summarize and compare research papers 📝. To get started, simply enter a query related to the topic you’re interested in. Our app will then select the most relevant papers related to your query. You can then compare and contrast the papers to get a better understanding of the topic. Our app is designed to be user-friendly and easy to use. We hope you find it helpful and informative! 📚🤓")
col1, col2,col3,col4 = st.columns(4)
with col1:
    option_tone = st.selectbox(
        'Which format you want to have your summarization?',
        ('Bullet point', 'Paragraph','Linkedin post'))
    
with col2:
    option_expert = st.selectbox(
        'What is the level of Expertise you like?',
        ('Basic for  6', 'Medium for 18', 'Expert for 30'))

with col3:
    option_abstract = st.selectbox(
        'Which paper would you like to summarize?',
        ('1','2','3','4'))
with col4:
    option_compare = st.selectbox(
        'Which paper would you want to compare with?',
        ('None','1','2','3','4'))
def get_text():
    input_text = st.text_input(label="Topic", label_visibility='collapsed', placeholder="Your Interest...", key="query")
    return input_text

query= get_text()

if len(query.split(" ")) > 700:
    st.write("Please enter a shorter query. The maximum length is 700 words.")
    st.stop()


#st.button("*See An Example*", type='secondary', help="Click to see an example of the email you will be converting.", on_click=update_text_with_example)

#st.markdown("### Your Converted Email:")
llm=Cohere(model="command-xlarge-nightly", cohere_api_key="vGCEakgncpouo9Nz0rsJ0Bq7XRvwNgTCZMKSohlg",
                                         temperature=0.7)
llm_1=Cohere(model="summarize-xlarge", cohere_api_key="vGCEakgncpouo9Nz0rsJ0Bq7XRvwNgTCZMKSohlg",temperature=0.9)
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
    st.markdown("### Your Summarized Abstract📃:")
    st.write(formatted_email)
    if option_compare=='None':
       st.markdown("### Comaparison between Abstracts📃:")
    
    else:
       abstract_a_t=x_page_content[int(option_abstract)-1]
       abstract_b_t=x_page_content[int(option_compare)-1]
       prompt_with_comp = prompt_1.format(abstract_a=abstract_a_t, abstract_b=abstract_b_t)

       formatted_comp = llm_1(prompt_with_comp)
       st.markdown("### Comaparison between Abstracts📃:")
       st.write(formatted_comp)
