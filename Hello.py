import streamlit as st
st.set_page_config(
    page_title="Hello",
    page_icon="🤖"
)
st.title(" Oil Buddy 🤖: Your Assistant")
st.header(" Welcome 👋")

st.markdown(
    """
    We are here to make Energy Professionals life easy   
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
    st.markdown("Often professionals would like to know about Oil and Gas. \n\n This tool \
                will help you improve your technical skills by advising for Oil and Gas. This tool \
                is powered by [LangChain](https://langchain.com/) and [OpenAI](https://openai.com) and made by \
                [@GregKamradt](https://twitter.com/GregKamradt).")

expander = st.expander("Domain Knowledge and Vision ")
expander.write("""
     In the fields Performance we monitor the Rates of Liquid, Oil,Gas,Water production of the wells.
     We use the data for optimizing the fields production & hence increasing the profit for the producer.
     We need to observe the water cut & GOR(Gas Oil Ratio) data at the platform level,well level & at the field level.
 """)
