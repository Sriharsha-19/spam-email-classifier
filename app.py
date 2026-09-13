import streamlit as st
import pickle

# Page settings
st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="centered"
)

# Load trained model
with open("spam_model.pkl", "rb") as file:
    model = pickle.load(file)

# Title
st.title("📧 Spam Email Classifier")

st.write(
    "Enter an email or message below and the machine-learning model "
    "will predict whether it is Spam or Ham."
)

st.divider()

# Message input
message = st.text_area(
    "📝 Enter your message",
    placeholder="Example: Congratulations! You have won a free prize...",
    height=150
)

# Check button
if st.button("🔍 Check Message", use_container_width=True):

    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")

    else:
        prediction = model.predict([message])[0]

        st.divider()

        if prediction == "spam":
            st.error("🚨 SPAM MESSAGE")
            st.write(
                "The model predicts that this message is likely to be spam."
            )

        else:
            st.success("✅ HAM / NORMAL MESSAGE")
            st.write(
                "The model predicts that this message is a normal message."
            )

st.divider()

st.caption(
    "Machine Learning Project • TF-IDF + Naive Bayes"
)