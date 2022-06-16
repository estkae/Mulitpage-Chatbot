import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

st.title("""Well wise Performance """)
st.header("Upload the Matser Production  data file here ")
st.markdown(" The file format is  standard Excel File")

data_uploader = st.file_uploader("upload file", type={"csv", "txt",'xlsx'})
if data_uploader is not None:
    try:
          data_df=pd.read_csv(data_uploader)
          data_df=data_df[['Platform','Well No','Date','Days','YEAR','Ql, blpd', 'Qo, bopd', 'Qw, bopd','RecOil, bbls   ',
                  'Qg (Assoc. Gas), m3/d','Moil, MMt', 'RecGas, m3']]  
    except:      
          data_df=pd.read_excel(data_uploader,sheet_name=10)

          data_df=data_df[['Platform','Well No','Date','Days','YEAR','Ql, blpd', 'Qo, bopd', 'Qw, bopd','RecOil, bbls   ',
                  'Qg (Assoc. Gas), m3/d','Moil, MMt', 'RecGas, m3']]
    
    
st.header("The Master Production Data ")
st.sidebar.header("User input parameter")

platfor=st.sidebar.selectbox('Select the platform ',options=data_df['Platform'].unique(),default=data_df['Platform'].unique()[-1])
df=data_df.copy()
df=df[df['Platform']==platfor]
well_name=st.sidebar.selectbox('Select the Well  of the platform ',options=df['Well No'].unique(),default=df['Well No'].unique()[0])



