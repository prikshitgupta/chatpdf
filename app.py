import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")

    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain



def handle_userinput(user_question):
    # Initialize chat_history if it's not already set
    if 'chat_history' not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = []

    # Instructions for the GPT model's context. These are NOT to be displayed to the user.
    specific_instructions = (
        "You don't know anything other than the data given to you."
        "Just reply You can about your receipts and expenses"
        "You will be provided with a PDF of receipts. It will have data from the user of their business expense receipts "
        "where the image of the receipt is in this 'image' json key. When the user asks for an image or images, just reply with the URL in 'image'json key."
        "i will presesent image locally with converting urls to images in swiftui app."
        "The User's Business name is 'businessName'."
    )
    
    # Merge the specific instructions with the user's question for the GPT model's prompt.
    full_prompt = f"{specific_instructions} {user_question}"

    # Generate a response from the GPT model using the full prompt.
    response = st.session_state.conversation(full_prompt)

    # Extract the AI's answer from the response.
    ai_answer = response.get('answer', "I'm not sure how to respond to that.")

    # Update chat history for display
    st.session_state.chat_history.append({'content': user_question, 'is_user': True})
    st.session_state.chat_history.append({'content': ai_answer, 'is_user': False})

    # Display the chat history in the Streamlit UI.
    for message in st.session_state.chat_history:
        if message['is_user']:
            # Display the user's message
            st.write(user_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)
        else:
            # Display the bot's (AI's) message
            st.write(bot_template.replace("{{MSG}}", message['content']), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
