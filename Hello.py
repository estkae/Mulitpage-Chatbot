import streamlit as st
st.set_page_config(
    page_title="Hello",
    page_icon="👋"
)
st.title(" Web App Based Dashboard")
st.header(" Welcome 👋")

st.markdown(
    """
    We are here to make Oil and Gas Engineer life easy   
""")

activities=['Login','About']
choice=st.sidebar.selectbox("Select Activity", activities)
if choice=='Login':
    st.subheader('Login')
    st.markdown(
    """
    Enter you name is Username   
    """)
    username=st.text_input("Enter Username")
    password=st.text_input("Enter Password", type='password')
    if st.button("Submit"):
        if password=='12345':
            st.balloons()
            st.write("Hello {}".format(username))
        else:
            st.warning('Wrong Password')
    
elif choice=='About':   
    st.write(
        """With the Release Streamlit Version 1.10.0 it is now possible to make a Multi-Page application 
     eliminating need of third party plugins. In this Web application we are working to make a dashboard of field performance 
   .We are Energy professionals &  our aim is to reduce the complexitiy of O & G Industy. 
             """)
expander = st.expander("Domain Knowledge of Oil & Gas ")
expander.write("""
     In the fields Performance we monitor the Rates of Liquid, Oil,Gas,Water production of the wells.
     We use the data for optimizing the fields production & hence increasing the profit for the producer.
     We need to observe the water cut & GOR(Gas Oil Ratio) data at the platform level,well level & at the field level.
 """)
