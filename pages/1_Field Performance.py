import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
st.title("""Field Performance Dashoboard""")

st.header("Upload the Matser Production  data file here ")
st.markdown(" The file format is  standard Excel File")

data_uploader = st.file_uploader("upload file", type={"csv", "txt",'xlsx'})
if data_uploader is not None:
    data_df=pd.read_excel(data_uploader,sheet_name=10)

    data_df=data_df[['Platform','Well No','Date','Days','YEAR','Ql, blpd', 'Qo, bopd', 'Qw, bopd','RecOil, bbls   ',
       'Qg (Assoc. Gas), m3/d','Moil, MMt', 'RecGas, m3']]
    
st.header("The Matser Production Data ")
st.sidebar.header("User input parameter")


from datetime import datetime
#start_time = st.sidebar.slider(
#     "When do you want plot to start?",
#     value=datetime(2016, 1),
#     format="Mth-yy")
st.dataframe(data_df.head())
df=data_df.copy()
platform=df['Platform'].unique()
year_list=df['YEAR'].unique()
df_platform=df.groupby(['Platform','Date']).sum()

data_frame_list=[]
for i in range(len(platform)):
    temp_df=df_platform.loc[platform[i]]
    data_frame_list.append(temp_df)
 
def data_frame_for_plot(data_frame_list_d):
  
   field_data_plot=pd.concat( data_frame_list_d)
   field_data_plot=field_data_plot.groupby('Date').sum()
   field_data_plot=field_data_plot.reset_index()
   field_data_plot['W/C']=field_data_plot['Qw, bopd']*100/field_data_plot['Ql, blpd']
   field_data_plot['GOR']=field_data_plot['Qg (Assoc. Gas), m3/d']*6.28/field_data_plot['Qo, bopd']
   return field_data_plot

def field_perf_plot(field_data_plot):
   field_data_plot['Date']=pd.to_datetime(field_data_plot['Date'])
   field_data_plot['Date']=field_data_plot['Date'].dt.strftime("%b-%y")
   fig=plt.figure(figsize=(18,10.5),dpi=70)
   ax = fig.add_subplot(211)

   ax.set_title('  Ratna & R-Series Field Performance plot ',fontsize=24)
   ax.plot(field_data_plot['Date'],field_data_plot['Ql, blpd'],color='brown',lw=2.5,label='Liquid Rate')
   ax.plot(field_data_plot['Date'],field_data_plot['Qo, bopd'],color='green',marker='o',lw=2.5,label='Oil Rate')
   ax.legend(loc=1,fontsize='x-large')
   ax.set_ylim([0, (int(field_data_plot['Ql, blpd'].values.max())+10000)])

 
   ax.set_xlabel("Date",fontsize=22,labelpad=10)
   ax.tick_params( axis='y',labelsize=13,direction='out', length=6, width=2, colors='black',
               grid_color='r', grid_alpha=0.5)
   ax.set_xticklabels(field_data_plot['Date'],fontsize=14,rotation=45)
   ax.set_ylabel("Ql & Qo in bpd",color="green",fontsize=22)
   ax2=ax.twinx()
   ax2.plot(field_data_plot['Date'], field_data_plot['W/C'],color="blue",marker="o",lw=2.5,label='Water Cut')
   ax2.set_yticks(np.round(np.linspace(0, 100, 16), 0))
   ax2.set_ylim([0, 100])
   ax2.legend(loc='upper right', fontsize='x-large',bbox_to_anchor=(0.99, 0.85))
   ax2.tick_params( axis='y',labelsize=13,direction='out', length=6, width=2, colors='black',
               grid_color='r', grid_alpha=0.5)
   ax2.set_ylabel("Water Cut ",color="blue",fontsize=22)
   ax.xaxis.grid(color='black', linestyle='--', linewidth=0.5)
   ax.yaxis.grid(color='black', linestyle='--', linewidth=0.5)

   ax3 = fig.add_subplot(212)
   #ax.set_title(platform[k]+'  Performance plot Allocation',fontsize=20)
   ax3.plot(field_data_plot['Date'],field_data_plot['Qg (Assoc. Gas), m3/d'],color='brown',lw=2.5,label='Gas Rate in m3/d')

   ax3.legend(loc=1,fontsize='x-large')
   ax3.set_ylim([0, (int(field_data_plot['Qg (Assoc. Gas), m3/d'].values.max())+20000)])
   #ax3.set_xlim(['Aug-18', 'Jun-22'])
   ax3.set_xlabel("Date",fontsize=22,labelpad=10)
   ax3.tick_params( axis='y',labelsize=13,direction='out', length=6, width=2, colors='black',
               grid_color='r', grid_alpha=0.5)
   ax3.set_xticklabels(field_data_plot['Date'],fontsize=14,rotation=45)
   ax3.set_ylabel("Gas Rate in m3/d",color="brown",fontsize=22)
   ax4=ax3.twinx()
   ax4.plot(field_data_plot['Date'], field_data_plot['GOR'],color="orange",marker="o",lw=2.5,label='GOR (v/v)')
   ax4.set_ylim([0, (int(field_data_plot['GOR'].values.max())+200)])
   ax4.legend(loc='upper right', fontsize='x-large',bbox_to_anchor=(0.99, 0.9))
   ax4.tick_params( axis='y',labelsize=13,direction='out', length=6, width=2, colors='black',
               grid_color='r', grid_alpha=0.5)
   ax4.set_ylabel("GOR (v/v)",color="orange",fontsize=22)
   ax3.xaxis.grid(color='black', linestyle='--', linewidth=0.5)
   ax3.yaxis.grid(color='black', linestyle='--', linewidth=0.5)
   #plt.show()
   #plt.savefig("Performance plot Allocation. pdf", format="pdf", bbox_inches="tight")

   return fig
df_field_dta_plot=data_frame_for_plot(data_frame_list)
fig1=field_perf_plot(df_field_dta_plot)
st.text('Pressure & Temperature Plot')
st.pyplot(fig1,width=20)
