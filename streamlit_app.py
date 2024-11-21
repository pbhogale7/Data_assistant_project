import os 
import streamlit as st
import pandas as pd

from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv

# Initialize all session state variables
if 'clicked' not in st.session_state:
    st.session_state.clicked = {1: False}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'llm' not in st.session_state:
    st.session_state.llm = None
if 'pandas_agent' not in st.session_state:
    st.session_state.pandas_agent = None

# Cache the LLM initialization
@st.cache_resource
def get_llm():
    """Initialize and return the OpenAI LLM instance"""
    return OpenAI(temperature=0)

# Cache the pandas agent creation
@st.cache_resource
def get_pandas_agent(_llm, df):
    """Create and return the pandas dataframe agent"""
    return create_pandas_dataframe_agent(_llm, df, verbose=True, allow_dangerous_code=True)

# Cache categorical data analysis
@st.cache_data(show_spinner="Analyzing categorical data...")
def analyze_categorical_data(_pandas_agent, column_name, df):
    st.write(f"**Analysis of {column_name}**")
    
    if column_name not in df.columns:
        st.error(f"Column '{column_name}' not found in the dataset")
        return
    
    try:
        if isinstance(df[column_name], pd.DataFrame):
            value_counts = df[column_name].iloc[:, 0].value_counts()
        else:
            value_counts = df[column_name].value_counts()
            
        st.write("Category Distribution:")
        st.write(value_counts)
        
        st.write("Distribution Visualization:")
        chart_data = pd.DataFrame({
            'Category': value_counts.index,
            'Count': value_counts.values
        })
        st.bar_chart(chart_data.set_index('Category'))
        
        distribution = _pandas_agent.run(f"""Analyze the distribution of {column_name} and provide insights. 
        Include:
        1. Most common categories
        2. Least common categories
        3. Any interesting patterns
        4. Potential business implications
        """)
        st.write("**Key Insights:**")
        st.write(distribution)
        
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        st.write("Raw data for debugging:")
        st.write(df[column_name].head())

# Cache question processing
@st.cache_data(show_spinner="Processing question...", ttl="10m")
def process_question(_pandas_agent, question, df):
    """Process a data-related question and determine if visualization is needed"""
    try:
        visualization_keywords = ["show", "plot", "display", "visualize", "graph", "chart", "distribution"]
        
        # Check if the question involves visualization
        needs_viz = any(keyword in question.lower() for keyword in visualization_keywords)
        
        if needs_viz:
            # Try to identify which columns to visualize
            columns_mentioned = [col for col in df.columns if col.lower() in question.lower()]
            
            if columns_mentioned:
                for col in columns_mentioned:
                    if df[col].dtype in ['object', 'category'] or df[col].dtype == 'bool':
                        analyze_categorical_data(_pandas_agent, col, df)
                    else:
                        st.line_chart(df, y=[col])
                        statistics = _pandas_agent.run(f"Analyze {col} and provide key statistical insights")
                        return statistics
        
        # Get the answer from the agent
        response = _pandas_agent.run(question)
        return response
    except Exception as e:
        return f"Error processing question: {str(e)}"

# Cache initial analysis
@st.cache_data(show_spinner="Performing initial analysis...")
def initial_analysis(_pandas_agent, df):
    """Perform initial EDA and store results in session state"""
    if not st.session_state.analysis_complete:
        st.write("**Data Overview**")
        st.write("The first rows of your dataset look like this:")
        st.write(df.head())
        
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        st.write("\n**Categorical Columns Available:**")
        st.write(list(categorical_columns))
        
        st.write("**Data Quality Assessment**")
        columns_df = _pandas_agent.run("""For each column, provide:
        1. The data type
        2. Whether it's categorical or numerical
        3. A brief description of what the column represents
        """)
        st.write(columns_df)
        
        missing_values = _pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
        st.write(missing_values)
        
        duplicates = _pandas_agent.run("Are there any duplicate values and if so where?")
        st.write(duplicates)
        
        st.session_state.analysis_complete = True
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "I've completed the initial analysis. What would you like to know about your data? You can ask me anything about the patterns, relationships, or specific aspects of your dataset."
        })

def clicked(button):
    st.session_state.clicked[button] = True

# OpenAIKey
os.environ['OPENAI_API_KEY']=st.secrets["openai_apikey"]
load_dotenv(find_dotenv())

# Title
st.title('AI Assistant for Data Science 🤖')

# Welcoming message
st.write("Hello, 👋 I am your AI Assistant and I am here to help you with your data science projects.")

# Explanation sidebar
with st.sidebar:
    st.write('*Your Data Science Adventure Begins with an CSV File.*')
    st.caption('''**You may already know that every exciting data science journey starts with a dataset.
    That's why I'd love for you to upload a CSV file.
    Once we have your data in hand, we'll dive into understanding it and have some fun exploring it.
    Then, we'll work together to shape your business challenge into a data science framework.**
    ''')
    st.divider()
    st.caption("<p style='text-align:center'> made with ❤️ by SU iBot</p>", unsafe_allow_html=True)

# Main app flow
st.button("Let's get started", on_click=clicked, args=[1])
if st.session_state.clicked[1]:
    user_csv = st.file_uploader("Upload your file here", type="csv")
    
    if user_csv is not None:
        # Only load the data if it's not already loaded or if it's a new file
        if st.session_state.df is None:
            user_csv.seek(0)
            st.session_state.df = pd.read_csv(user_csv, low_memory=False)
            
            # Initialize LLM and pandas agent if not already done
            if st.session_state.llm is None:
                st.session_state.llm = get_llm()
            if st.session_state.pandas_agent is None:
                st.session_state.pandas_agent = get_pandas_agent(st.session_state.llm, st.session_state.df)
            
            # Perform initial analysis only once
            if not st.session_state.analysis_complete:
                initial_analysis(st.session_state.pandas_agent, st.session_state.df)

        # Chat interface
        st.write("---")
        st.subheader("Ask me anything about your data 💭")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        # User input
        user_question = st.chat_input("Type your question here...")
        if user_question:
            # Add user message to chat history
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_question
            })
            
            # Process the question and get response
            with st.chat_message("assistant"):
                response = process_question(st.session_state.pandas_agent, user_question, st.session_state.df)
                st.write(response)
                
                # Add assistant's response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                
                # Proactively ask if user wants to know more
                st.write("Is there anything else you'd like to know about your data?")