# %%
#%pip install -r requirements.txt

# %%
import os
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type
from bs4 import BeautifulSoup
import requests
import json
from langchain.schema import SystemMessage
from fastapi import FastAPI
import streamlit as st

load_dotenv()
brwoserless_api_key = os.getenv("BROWSERLESS_API_KEY")
serper_api_key = os.getenv("SERP_API_KEY")

# %%
# 1. Tool for search

def search(query):
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query
    })
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    print(response.text)
    return response.text
# Langchain also gives a Serper AI for google; Make sure to use Google Serper API and not Serp API there if you use Langchain for the above step

# %%
search("Who was the first president of Ghana")

# %%
# 2. a) Tool for scraping
def scrape_website(objective: str, url: str):
    # scrape website, and also will summarize the content based on objective if the content is too large
    # objective is the original objective & task that user give to the agent, url is the url of the website to be scraped

    print("Scraping website...")
    # Define the headers for the request
    headers = {
        'Cache-Control': 'no-cache',
        'Content-Type': 'application/json',
    }

    # Define the data to be sent in the request
    data = {
        "url": url
    }

    # Convert Python object to JSON string
    data_json = json.dumps(data)

    # Send the POST request
    post_url = f"https://chrome.browserless.io/content?token={brwoserless_api_key}"
    response = requests.post(post_url, headers=headers, data=data_json)

    # Check the response status code
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        text = soup.get_text()
        print("CONTENTTTTTT:", text)

# If length of text is greater than 10k words, we will return the summary else we can return the text as is
        if len(text) > 10000:
            output = summary(objective, text)
            return output
        else:
            return text
    else:
        print(f"HTTP request failed with status code {response.status_code}")



# Too many tokens from scraping output; We need to summarize the data and shrink - need to fit within limit token limit; Two options available
# A) we take each paragraph and feed it to an LLM to summarize each paragraph [1 API call per paragraph] and club them - PREFERABLE FOR 2 PAGES OF DATA OR SMALL DATA
# B) Alternatively, we can save the full text as vector embeddings and we will do a vector search for the most relevant content - PREFERABLE FOR LARGE DATASET

# b. We're using summary in the a. for summarizing text which is greater than 10k words
def summary(objective, content):
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
    #Splitter will convert large content into chunks of 10k -- We're using GPT3.5 which has a 16k token context length, so this should be fine
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500)
    docs = text_splitter.create_documents([content])
    #Prompt used for writing summary of the content
    map_prompt = """
    Write a summary of the following text for {objective}:
    "{text}"
    SUMMARY:
    """
    map_prompt_template = PromptTemplate(
        template=map_prompt, input_variables=["text", "objective"])
    # The following will automatically summarize each of the chunks we split earlier and it will be combined 
    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type='map_reduce',
        map_prompt=map_prompt_template,# prompt used to summarize each chunk/paragraph of text
        combine_prompt=map_prompt_template,#prompt used to summarize all chunks/paragraphs into one paragraph
        verbose=True
    )
    output = summary_chain.run(input_documents=docs, objective=objective)
    return output


# %%
class ScrapeWebsiteInput(BaseModel):
    """Inputs for scrape_website"""
    # Type casting to str in ojbective and url
    # Helps agent figure out what kinds of inputs to pick and pass onto the function where this class/constructor is called
    objective: str = Field(description="The objective & task that users give to the agent")
    url: str = Field(description="The url of the website to be scraped")


class ScrapeWebsiteTool(BaseTool):
    name = "scrape_website"
    description = "useful when you need to get data from a website url, passing both url and objective to the function; DO NOT make up any url, the url should only be from the search results"
    # Defining the schema of the input which is given the prior class ScrapeWebsiteInput which inherits from the base model
    # At the time of initialization itself, the following code will be run to extract the inputs required for Scraping website i.e. objective & URL
    args_schema: Type[BaseModel] = ScrapeWebsiteInput
    # If this class is called, _run will be run 
    def _run(self, objective: str, url: str):
        # Function to be run along with the fields for input to it --- extremely convoluted way to do this but it is for the agent to pick and share what kind of inputs to pass to the function
        return scrape_website(objective, url)

    def _arun(self, url: str):
        raise NotImplementedError("error here")


# 3. Create langchain agent with the tools above
tools = [
    # Tool converts a regular function into a a tool. search is defined in 1. and runs this url https://google.serper.dev/search to insert query - this is basically provided to agent
    #  where it will have to generate the query which will be run on search() basis teh description given below
    # Then a ScrapeWebsiteTOol() is run -- ScrapeWebsiteTOol() has more code as it requires two inputs instead of just 1 that is required for search()
    Tool(
        name="Search",
        func=search,
        description="useful for when you need to answer questions about current events, data. You should ask targeted questions"
    ),
    ScrapeWebsiteTool(),
]

system_message = SystemMessage(
    content="""You are a world class researcher, who can do detailed research on any topic and produce facts based results; 
            you do not make things up, you will try as hard as possible to gather facts & data to back up the research
            
            Please make sure you complete the objective above with the following rules:
            1/ You should do enough research to gather as much information as possible about the objective
            2/ If there are url of relevant links & articles, you will scrape it to gather more information
            3/ After scraping & search, you should think "are there any new things i should search & scraping based on the data I collected to increase research quality?" If answer is yes, continue; But don't do this more than 3 iteratins
            4/ You should not make things up, you should only write facts & data that you have gathered
            5/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            6/ In the final output, You should include all reference data & links to back up your research; You should include all reference data & links to back up your research
            """
            # 5 & 6 are repeated on purpose as just giving it once sometimes doesn't work
)

agent_kwargs = {
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
    "system_message": system_message,
}


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")
# Memory variable through which Agent will remember context i.e previous steps / recent chat history word by word till some limit and prior that it will remmember the summary 
# Exactly remmebers 1000 tokens and older than that it will remember summary
memory = ConversationSummaryBufferMemory(
    memory_key="memory", return_messages=True, llm=llm, max_token_limit=1000)

# This is the main running code for the agent
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.OPENAI_FUNCTIONS,#Better performance for extracting information and passing them as inputs to other functions
    verbose=True,# Can see what agent is thinking at every single step
    agent_kwargs=agent_kwargs,
    memory=memory,
)


# %%

# 4. Use streamlit to create a web app
#def main():
#    st.set_page_config(page_title="AI research agent", page_icon=":bird:")

#    st.header("AI research agent :bird:")
#      
#    query = st.text_input("Research goal")

#    if query:
#        st.write("Doing research for ", query)

#        result = agent({"input": query})

#        st.info(result['output'])


#if __name__ == '__main__':
#    main()



# %% [markdown]
# Install streamlit in conda environment using "pip install streamlit"
# Enter "streamlit run app.py" to start streamlit server

# %%


# %%
# 5. Set this as an API endpoint via FastAPI
# To run - comment out the portion for streamlit and uncomment the following section
# to run server enter the following in the command line -- "uvicorn app:app --host 0.0.0.0 --port 10000"
app = FastAPI()


class Query(BaseModel):
    query: str


@app.post("/")
def researchAgent(query: Query):
    query = query.query
    content = agent({"input": query})
    actual_content = content['output']
    return actual_content


