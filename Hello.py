import streamlit as st
st.set_page_config(
    page_title="Oil Buddy",
    page_icon="🤖"
)
st.title(" Oil Buddy 🤖: Your Assistant")
st.write(" Version 3.0.0")

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
                will help you improve your technical skills by advising for Oil and Gas. This tool \n
                is powered by [LangChain](https://langchain.com/) and [OpenAI](https://openai.com).\n
                ")

expander = st.expander("Domain Knowledge and Vision ")
expander.write("""
     Domain knowledge is important in the oil and gas industry because it helps employees understand the complexities of the industry
     and the processes involved in producing and distributing oil and gas
    Our vision is to revolutionize the energy industry by creating an app that leverages the power of AI\n
                and large language models to provide innovative solutions.
 """)
