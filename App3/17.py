# Title: Use 4 Input Variables in the Template
#
# Description:
# This script expands on the previous lesson by demonstrating how to use MULTIPLE variables
# in a LangChain PromptTemplate.
#
# In real applications, a prompt is rarely just one variable ("question"). It often combines
# user data (name), context (topic), specific constraints (instructions), and the query itself.
# This script shows how to structure a complex prompt with 4 distinct dynamic inputs.
#
# Installation:
# pip install streamlit==1.33.0 langchain==0.2.16 langchain-community==0.2.17
# pip install streamlit langchain-community
# How to run:
# streamlit run 17.py

import streamlit as st
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

st.title("Local LLM with Langchain!")

# --- UI Section: Collect Multiple Inputs ---
# We create separate input fields for each piece of data we need.
user_name = st.text_input(label="Your Name:")
topic = st.text_input(label="Topic:")
question = st.text_input(label="Question:")
instructions = st.text_area(label="Instructions:")  # text_area for longer input
button = st.button("Okay")

if button:
    # Ensure all fields are filled before proceeding
    if user_name and topic and question and instructions:
        # Initialize the local LLM
        llm = Ollama(model='llama2') # Specify your model here

        # --- Define Complex Prompt Template ---
        # Note how we use four different placeholders: {name}, {topic}, {question}, {instructions}.
        # This allows us to inject data into specific parts of the prompt sentence structure.
        template = """You are a helpful assistant. {name} has asked a question about {topic}.

Question: {question}

Additional Instructions: {instructions}

Please provide a detailed and thoughtful response."""

        # When creating the PromptTemplate object, we list ALL variables in the input_variables list.
        prompt_template = PromptTemplate(
            template=template,
            input_variables=["name", "topic", "question", "instructions"]
        )

        # Create the chain using RunnableSequence (prompt | llm)
        chain = prompt_template | llm

        # --- Invoke the Chain with a Dictionary ---
        # Pass a dictionary mapping each template variable to its value.
        response = chain.invoke({
            "name": user_name,
            "topic": topic,
            "question": question,
            "instructions": instructions,
        })

        # Display the response
        st.markdown(response)
