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
import os
from typing import Optional, Tuple

import pickle
from query_data import get_chain
from threading import Lock
import docsearch as ds

with open("vectorstore.pkl", "rb") as f:
    vectorstore = pickle.load(f)

#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
def set_openai_api_key(api_key: str):
    """Set the api key and return chain.
    If no api_key, then None is returned.
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        chain = get_chain(vectorstore)
        os.environ["OPENAI_API_KEY"] = ""
        return chain

class ChatWrapper:

    def __init__(self):
        self.lock = Lock()
    def __call__(
            self, api_key: str, inp: str, history: Optional[Tuple[str, str]], chain
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        try:
            history = history or []
            # If chain is None, that is because no API key was provided.
            if chain is None:
                history.append((inp, "Please paste your OpenAI key to use"))
                return history, history
            # Set OpenAI key
            import openai
            openai.api_key = api_key
            # Run chain and append input.
            output = chain({"question": inp, "chat_history": history})["answer"]
            history.append((inp, output))
        except Exception as e:
            raise e
        finally:
            self.lock.release()
        return history, history

chat = ChatWrapper()

st.set_page_config(
     page_title='Oil Buddy 🤖',
     initial_sidebar_state="expanded"
     #layout="wide",
)


#default embeddings
#doc1 = load_vectorstore()
#doc2 = load_vectorstore2()
#docsearch=doc1.merge_from(doc2)
#st.set_page_config(page_title="Chatbot", page_icon=":shark:")
def load_vectorstore():
    '''load embeddings and vectorstore'''

    embeddings = CohereEmbeddings(cohere_api_key= "vGCEakgncpouo9Nz0rsJ0Bq7XRvwNgTCZMKSohlg")

    return FAISS.load_local('tot_embeddings', embeddings)
    #return FAISS.load_local('resr_manang_embeddings', embeddings)

st.header("Oil Buddy🤖 Your Assistant ",)
st.sidebar.header('Sources and Citations')

st.sidebar.write("Oil Buddy  has been using open source  Oil and Gas Engineering Materials for educational purposes only.\nIts primary sources are\nPetroleum Engineering material prepared for GATE by courtsey of Mr. Akshay Shekhawat,Inhouse experts data of various disciplines as well as reputable websites such as Wikipedia, PetroWiki and  You Tube Videos related to Petroleum Engineering\n")
st.sidebar.header('References')
st.sidebar.write("Check Out References for more detailed info ℹ️ :\n  [Wikipedia](https://www.wikipedia.org/)\n [PetroWiki](https://petrowiki.spe.org/PetroWiki)\n [Oil and Gas](https://www.oil-gasportal.com/)")
st.sidebar.header('A Friendly Reminder')
st.sidebar.write("Hey there! Just a quick note to let you know that the information provided by this chatbot is for general informational purposes only.\n.So, please take the results with a grain of salt and don't hesitate to double-check the information if you're not sure. We're here to help you, but we're not perfect. 😊")
import streamlit as st 


expander = st.expander("Know about Me")

expander.write("""
     :black[I am an AI assistant for Oil and Gas Engineers based on LLMs(Large Language Models).Presently I know about  Basics of Reservoir & Production Engineering. Consider the generated response as starting point to assist in our work.] 
     
 """)
if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


placeholder2 = st.empty()
placeholder = st.empty()

def get_text_sk():

    input_text = placeholder2.text_input("Paste your OpenAI API key (sk-...) 👇", "", key=None, type="password")
    return input_text


def get_text():

    input_text = placeholder.text_input("Enter what you want to know 👇", "", key=None)
    return input_text


qa_chain=set_openai_api_key(get_text_sk())
st.text(qa_chain)
user_input = get_text()
docsearch = load_vectorstore()

placeholder11="Ask questions about the most recent state of the union"
placeholder12="Did he mention Stephen Breyer?"
placeholder13="What was his stance on Ukraine"
if st.button(placeholder11, key=None):
    message = placeholder11
    st.text(placeholder11)
if st.button(placeholder12, key=None):
    message = placeholder12
    st.text(placeholder12)
if st.button(placeholder13, key=None):
    message = placeholder13
    st.text(placeholder13)

#submit = st.button("Submit Your Query", key=None)
#submit.click(chat, inputs=[open_ai_key, message, state, agent_state], outputs=[chatbot, state])
#message.submit(chat, inputs=[open_ai_key, message, state, agent_state], outputs=[chatbot, state])

if st.button("Submit Your Query"):
    # check 
    docs = docsearch.similarity_search(user_input)
    # if checkbox is checked, print docs

    print(len(docs))
    #if user_input:
    chat_history = []
    print("Human:")
    question = "Ask questions about the most recent state of the union"
    result = qa_chain({"question": question, "chat_history": chat_history})
    chat_history.append((question, result["answer"]))
    print("AI:")
    print(result["answer"])

    #output = open_ai_key({"question": user_input, "chat_history": chat_history})
    #output = qa.run(user_input)
    
    st.session_state.past.append(user_input)
    #st.session_state.generated.append(output["source_documents"][0])
    #st.session_state.generated.append([output["answer"],output["source_documents"]])
    st.session_state.generated.append(result["answer"])
    #st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["past"][i], is_user=True,avatar_style="thumbs",seed='Aneka',key=str(i) + "_user")

        message(st.session_state["generated"][i],avatar_style="fun-emoji", key=str(i))
        #message(st.session_state["generated"][i+1], key=str(i+1))
