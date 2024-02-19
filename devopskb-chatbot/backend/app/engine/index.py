import logging
import os

from app.engine.constants import STORAGE_DIR, TITLES
from app.engine.generate import generate_datasource
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
)
from llama_index.core.schema import IndexNode
from llama_index.core.retrievers import RecursiveRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import get_response_synthesizer
from llama_index.llms.openai import OpenAI

def get_index_and_query_engine():
    # service_context = create_service_context()
    # # check if storage already exists
    # if not os.path.exists(STORAGE_DIR):
    #     raise Exception(
    #         "StorageContext is empty - call 'python app/engine/generate.py' to generate the storage first"
    #     )
    # logger = logging.getLogger("uvicorn")
    # # load the existing index
    # logger.info(f"Loading index from {STORAGE_DIR}...")
    # storage_context = StorageContext.from_defaults(persist_dir=STORAGE_DIR)
    # index = load_index_from_storage(storage_context, service_context=service_context)
    # logger.info(f"Finished loading index from {STORAGE_DIR}")
    # return index

    model = os.getenv("MODEL", "gpt-3.5-turbo")
    llm=OpenAI(model=model)

    agents = generate_datasource()

    # define index nodes that link to the document agents
    nodes = []
    for title in TITLES:
        doc_summary = (
            f"This content contains details about {title}. "
            f"Use this index if you need to lookup specific facts about {title}.\n"
            "Do not use this index if you want to query multiple documents."
        )
        node = IndexNode(text=doc_summary, index_id=title)
        nodes.append(node)

    # define retriever
    vector_index = VectorStoreIndex(nodes)
    vector_retriever = vector_index.as_retriever(similarity_top_k=1)

    # define recursive retriever
    # note: can pass `agents` dict as `query_engine_dict` since every agent can be used as a query engine
    recursive_retriever = RecursiveRetriever(
        "vector",
        retriever_dict={"vector": vector_retriever},
        query_engine_dict=agents,
        verbose=False,
    )

    response_synthesizer = get_response_synthesizer(response_mode="compact")

    # define query engine
    recursive_query_engine = RetrieverQueryEngine.from_args(
        recursive_retriever,
        response_synthesizer=response_synthesizer,
        llm=llm,
    )

    return recursive_query_engine
