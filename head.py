import streamlit as st
import sys
from landmarks import *
sys.path.insert(1, 'D:/Streamlit/Gifs')


st.title("Sign Language / Action Recognition")





    
st.write("  ")
st.image("Gifs\Hello.gif")
st.write("""There have been several advancements in technology and a lot of research has been done to help the people who are deaf and dumb. Aiding the cause, Deep learning, and computer vision can be used too to make an impact on this cause.

This can be very helpful for the deaf and dumb people in communicating with others as knowing sign language is not something that is common to all, moreover, this can be extended to creating automatic editors, where the person can easily write by just their hand gestures.

""")

st.subheader("Check your background testing")
col1,col2 = st.columns(2)
with col1:
    
    st.image('Gifs\correct.png')
    st.markdown("## Correct background")
with col2:
    
    st.image("Gifs\incorrect.png")
    st.markdown("## Incorrect background")
    
    
st.subheader("Learn Some Indian sign language words.")
col1,col2 = st.columns(2)
with col1:
    st.image("Gifs\Thanks.gif")
    st.caption("Thanks")
    
    st.image("Gifs\Hello.gif")
    st.caption("Hello")
    
    st.image("Gifs\Man.gif")
    st.caption("man")
    
    st.image("Gifs\sign.gif")
    st.caption("man")
    
    st.image("Gifs\sorry.gif")
    st.caption("Sorry")


with col2:
    st.image("Gifs\Indian.gif")
    st.caption("Indian")
    
    st.image("Gifs\Deaf.gif")
    st.caption("Deaf")
    
    st.image("Gifs\Women.gif")
    st.caption("Women")
    
    st.image("Gifs\Language.gif")
    st.caption("man")
    
    st.image("Gifs\Teacher.gif")
    st.caption("Teacher")
    


