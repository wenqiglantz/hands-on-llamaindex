import logging, os
from dotenv import load_dotenv
from app.engine.constants import DATA_DIR, STORAGE_DIR, TITLES
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    SummaryIndex,
)
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.llms.openai import OpenAI
from llama_index.agent.openai import OpenAIAgent
import pandas as pd

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

model = os.getenv("MODEL", "gpt-3.5-turbo")
llm=OpenAI(model=model)

def generate_datasource():
    logger.info("Creating new index")
    # # load the documents and create the index
    # documents = get_documents()
    # index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    # # store it for later
    # index.storage_context.persist(STORAGE_DIR)
    # logger.info(f"Finished creating new index. Stored in {STORAGE_DIR}")

    # Build agents dictionary
    agents = {}

    for title in TITLES:

        # load the documents
        documents = {}
        for title in TITLES:
            documents[title] = SimpleDirectoryReader(input_files=[f"./data/{title}.pdf"]).load_data()
        print(f"loaded documents with {len(documents)} documents")
        
        # build vector index
        vector_index = VectorStoreIndex.from_documents(documents[title])

        # build summary index
        summary_index = SummaryIndex.from_documents(documents[title])

        # define query engines
        vector_query_engine = vector_index.as_query_engine(llm=llm)
        summary_query_engine = summary_index.as_query_engine(llm=llm)

        # define tools
        query_engine_tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name="vector_tool",
                    description=f"Useful for retrieving specific context related to {title}",
                ),
            ),
            QueryEngineTool(
                query_engine=summary_query_engine,
                metadata=ToolMetadata(
                    name="summary_tool",
                    description=f"Useful for summarization questions related to {title}",
                ),
            ),
        ]

        # build agent
        function_llm = OpenAI(model="gpt-3.5-turbo-0613")
        agent = OpenAIAgent.from_tools(
            query_engine_tools,
            llm=function_llm,
            verbose=False,
        )

        agents[title] = agent
        return agents



if __name__ == "__main__":
    generate_datasource()
