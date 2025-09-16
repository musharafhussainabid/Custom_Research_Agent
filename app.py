from dotenv import load_dotenv
import os
import streamlit as st
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain.chains import LLMMathChain

# 1. Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# 2. Initialize LLM with API key from .env
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GOOGLE_API_KEY
)

# 3. Research Agent Class
class ResearchAgent:
    def __init__(self, llm):
        self.llm = llm
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Wikipedia Tool
        wikipedia_tool = Tool(
            name="Wikipedia",
            func=self.search_wikipedia,
            description="Useful for searching Wikipedia articles."
        )

        # Calculator Tool
        llm_math = LLMMathChain.from_llm(llm=self.llm)
        calculator_tool = Tool(
            name="Calculator",
            func=llm_math.run,
            description="Useful for performing mathematical operations and calculations."
        )

        self.tools = [wikipedia_tool, calculator_tool]

        # Initialize agent
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True
        )

    def search_wikipedia(self, query):
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
        return wikipedia.run(query)

    def run(self, query):
        return self.agent.run(query)


# 4. Streamlit UI
st.set_page_config(page_title="Research Agent", page_icon="🔎", layout="wide")

st.title("🔎 Research Agent")
st.markdown("Ask me anything! I can search Wikipedia and perform calculations if needed.")

# Initialize the ResearchAgent once
if "agent" not in st.session_state:
    st.session_state.agent = ResearchAgent(llm)
    st.session_state.chat_history = []

# Chat input
query = st.chat_input("Type your question here...")

if query:
    # Append user query to chat
    st.session_state.chat_history.append(("user", query))

    # Run agent
    with st.spinner("Thinking..."):
        response = st.session_state.agent.run(query)

    # Append response
    st.session_state.chat_history.append(("agent", response))

# Display chat history
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        st.chat_message("assistant").write(msg)
