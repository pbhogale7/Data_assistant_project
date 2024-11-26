import os 
import streamlit as st
import pandas as pd
import plotly.express as px
from langchain_openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from dotenv import load_dotenv, find_dotenv

# Cache the LLM initialization
@st.cache_resource
def get_llm():
    """Initialize and return the OpenAI LLM instance"""
    try:
        return OpenAI(temperature=0)
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

# Cache the pandas agent creation
@st.cache_resource
def get_pandas_agent(_llm, df):
    """Create and return the pandas dataframe agent"""
    try:
        return create_pandas_dataframe_agent(_llm, df, verbose=True)
    except Exception as e:
        st.error(f"Error creating pandas agent: {str(e)}")
        return None

def create_visualization(df, column_name, viz_type, title=None):
    """Create different types of visualizations based on the specified type"""
    try:
        if viz_type == "bar":
            fig = px.bar(df, x=column_name, title=title)
        elif viz_type == "line":
            fig = px.line(df, y=column_name, title=title)
        elif viz_type == "scatter":
            fig = px.scatter(df, x=df.index, y=column_name, title=title)
        elif viz_type == "histogram":
            fig = px.histogram(df, x=column_name, title=title)
        elif viz_type == "box":
            fig = px.box(df, y=column_name, title=title)
        elif viz_type == "pie":
            value_counts = df[column_name].value_counts()
            fig = px.pie(values=value_counts.values, names=value_counts.index, title=title)
        else:
            raise ValueError(f"Unsupported visualization type: {viz_type}")
        
        st.plotly_chart(fig)
        return True
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return False

# Cache categorical data analysis
@st.cache_data(show_spinner="Analyzing categorical data...")
def analyze_categorical_data(_pandas_agent, column_name, df):
    try:
        st.write(f"**Analysis of {column_name}**")
        
        if column_name not in df.columns:
            st.error(f"Column '{column_name}' not found in the dataset")
            return
        
        if isinstance(df[column_name], pd.DataFrame):
            value_counts = df[column_name].iloc[:, 0].value_counts()
        else:
            value_counts = df[column_name].value_counts()
        
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
        st.error(f"Error in analysis: {str(e)}")
        st.write("Raw data for debugging:")
        st.write(df[column_name].head())

# Cache question processing
@st.cache_data(show_spinner="Processing question...", ttl="10m")
def process_question(_pandas_agent, question, df):
    """Process a data-related question and determine appropriate visualization"""
    try:
        visualization_keywords = ["show", "plot", "display", "visualize", "graph", "chart", "distribution"]
        
        # Check if the question involves visualization
        needs_viz = any(keyword in question.lower() for keyword in visualization_keywords)
        
        if needs_viz:
            # Ask LLM to analyze the question and data to determine visualization needs
            viz_analysis = _pandas_agent.run(f"""
            Analyze this question: "{question}"
            For any columns that need visualization, provide a JSON-like response with:
            1. The column name
            2. The best visualization type (choose from: bar, line, scatter, histogram, box, pie)
            3. A brief explanation of why this visualization type is appropriate
            4. A title for the visualization
            
            Consider:
            - Data type of the column
            - The question's intent
            - Statistical properties of the data
            
            Format: {{"column": "column_name", "viz_type": "type", "reason": "explanation", "title": "chart title"}}
            If multiple visualizations are needed, provide multiple JSON objects.
            """)
            
            # Extract visualization recommendations from the LLM's response
            import re
            import json
            
            # Find all JSON-like objects in the response
            json_patterns = re.findall(r'\{[^}]+\}', viz_analysis)
            
            if json_patterns:
                for json_str in json_patterns:
                    try:
                        viz_info = json.loads(json_str)
                        if all(k in viz_info for k in ["column", "viz_type", "reason", "title"]):
                            st.write(f"**Visualization Insight:** {viz_info['reason']}")
                            create_visualization(
                                df, 
                                viz_info['column'], 
                                viz_info['viz_type'],
                                viz_info['title']
                            )
                    except json.JSONDecodeError:
                        continue
        
        # Get the answer from the agent
        response = _pandas_agent.run(question)
        return response
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return f"I encountered an error while processing your question: {str(e)}"

# Cache initial analysis
@st.cache_data(show_spinner="Performing initial analysis...")
def initial_analysis(_pandas_agent, df):
    """Perform initial EDA and store results in session state"""
    try:
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
    except Exception as e:
        st.error(f"Error in initial analysis: {str(e)}")

# Main application
try:
    # OpenAIKey
    os.environ['OPENAI_API_KEY'] = st.secrets["openai_apikey"]
    load_dotenv(find_dotenv())

    # Title
    st.title('SU iBot 🤖')

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

    # Initialize session state
    if 'clicked' not in st.session_state:
        st.session_state.clicked = {1: False}
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False

    def clicked(button):
        st.session_state.clicked[button] = True

    st.button("Let's get started", on_click=clicked, args=[1])
    
    if st.session_state.clicked[1]:
        user_csv = st.file_uploader("Upload your file here", type="csv")
        if user_csv is not None:
            try:
                user_csv.seek(0)
                df = pd.read_csv(user_csv, low_memory=False)

                # Create LLM and pandas agent using cached functions
                llm = get_llm()
                if llm is not None:
                    pandas_agent = get_pandas_agent(llm, df)
                    if pandas_agent is not None:
                        # Perform initial analysis
                        initial_analysis(pandas_agent, df)

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
                                response = process_question(pandas_agent, user_question, df)
                                st.write(response)
                                
                                # Add assistant's response to chat history
                                st.session_state.chat_history.append({
                                    "role": "assistant",
                                    "content": response
                                })
                                
                                # Proactively ask if user wants to know more
                                st.write("Is there anything else you'd like to know about your data?")
            except Exception as e:
                st.error(f"Error processing CSV file: {str(e)}")

except Exception as e:
    st.error(f"Application error: {str(e)}")