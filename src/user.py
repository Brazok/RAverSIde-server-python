from llama_index.embeddings.jinaai import JinaEmbedding
from typing import Any
from src.Utils import get_new_api_key
from src.Utils import MixtralLLM
from src.Utils import retrieve_number
from src.Utils import increment_counter_if_cwe_found
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.legacy import PromptTemplate
from llama_index.core.retrievers  import VectorIndexRetriever
from llama_index.postprocessor.jinaai_rerank import JinaRerank

from src.feature import getPromptFromFile


def anthony(code, question) -> Any:
    jinaai_api_key = get_new_api_key()
    jina_embedding_model = JinaEmbedding(
        api_key=jinaai_api_key,
        model="jina-embeddings-v2-base-code",
    )

    mixtral_llm = MixtralLLM()

    Settings.llm = mixtral_llm
    Settings.embed_model = jina_embedding_model
    Settings.num_output = 512
    Settings.context_window = 4096

    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir="persist_dir")
    # load index
    index = load_index_from_storage(storage_context)


    qa_prompt_tmpl = getPromptFromFile("chatbot")
    qa_prompt_tmpl = qa_prompt_tmpl.replace("{question}", question)

    qa_prompt = PromptTemplate(qa_prompt_tmpl)



    # configure retriever
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=4,
    )


    jina_rerank = JinaRerank(api_key=jinaai_api_key, model="jina-reranker-v1-base-en", top_n=4)

    query_engine = index.as_query_engine(similarity_top_k=4, text_qa_template=qa_prompt, response_mode="compact", node_postprocessors=[jina_rerank], streaming=True)

    content = 0
    print("Querying")
    print(code)
    result = query_engine.query(code)


    result.print_response_stream()

    return "coucou"

    




