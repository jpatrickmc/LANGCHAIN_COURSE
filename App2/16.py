# Title: Use PromptTemplate and LLMChain
#
# Description:
# This script introduces two core concepts of LangChain:
# 1. PromptTemplate: A way to create dynamic, reusable prompts where you can insert variables (like the user's question).
# 2. LLMChain: A simple "Chain" that links a PromptTemplate and an LLM together.
#
# Instead of just sending the raw user input to the model, we wrap it in a structure
# (e.g., "You are a helpful assistant...") to control the AI's behavior and formatting.
#
# Installation:
# pip install streamlit==1.33.0 langchain==0.2.16 langchain-community==0.2.17
# pip install streamlit langchain-community
#
# How to run:
# streamlit run 16.py

import streamlit as st
from langchain_community.llms import Ollama  # Wrapper for the local model
from langchain.prompts import PromptTemplate  # For creating flexible prompt strings

st.title("Local LLM with Langchain!")

# Input for the prompt
prompt = st.text_area(label="Write your prompt.")
button = st.button("Okay")

if button:
    if prompt:
        # Initialize the local LLM
        llm = Ollama(model='llama2')  # Specify your model here

        # --- Define the Prompt Template ---
        # A template is a string with placeholders (variables inside curly braces {}).
        # This allows us to "frame" the user's question.
        # Here, we force the AI to answer as a "list with short items" regardless of what the user asks.
        template = """You are a helpful assistant. You have been asked the following question:

Question: {question}

Please provide a detailed and thoughtful response as a list with short items.
"""
        # Create the PromptTemplate object.
        # input_variables=["question"] tells LangChain that we will provide a value for {question} later.
        prompt_template = PromptTemplate(template=template, input_variables=["question"])

        # --- Create the chain using RunnableSequence ---
        # Using the pipe operator (|) to chain the prompt template and the LLM.
        # When invoked, it will:
        # 1. Take our input dict and fill the {question} slot in the template.
        # 2. Send the formatted string to the LLM.
        chain = prompt_template | llm

        # Generate a response by invoking the chain with the user's input.
        response = chain.invoke({"question": prompt})

        # Display the response
        st.markdown(response)
