import itertools

import datetime
import re
from typing import Any
import os
import requests
import json

from typing import List

from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import VectorStoreIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings
from llama_index.core.schema import Document
from llama_index.core.readers import StringIterableReader
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)



def repair_json(s):
    depth = 0
    json_start_index = -1
    for i, char in enumerate(s):
        if char == '{':
            if depth == 0:
                json_start_index = i
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                # Tente de décoder à chaque fermeture du dernier objet
                try:
                    json_object = json.loads(s[json_start_index:i+1])
                    return json_object
                except json.JSONDecodeError as e:
                    print("Erreur de décodage JSON:", e)
                    pass

    if s.count('"') % 2 != 0:
        s += '"'

    # Ajoute les accolades fermantes manquantes si le JSON est incomplet
    if depth > 0 and json_start_index != -1:
        s = s[json_start_index:] + '}' * depth

    # Tentative de correction du JSON
    max_attempts = 10  # Limite le nombre de tentatives pour éviter les boucles infinies
    attempt = 0
    while attempt < max_attempts:
        # Ajoute un guillemet double à la fin de la chaîne si nécessaire


        try:
            json_object = json.loads(s)
            return json_object
        except json.JSONDecodeError as e:
            print(s)
            error_message = str(e)
            print("Erreur de décodage JSON:", error_message)
            if "Unterminated string" in error_message:
                error_position = int(error_message.split('char ')[1][:-1])
                s = s[:error_position] + '"' + s[error_position:]
            elif "Expecting ',' delimiter" in error_message:
                error_position = int(error_message.split('char ')[1][:-1])
                s = s[:error_position] + ',' + s[error_position:]
            elif "Expecting ':' delimiter" in error_message:
                error_position = int(error_message.split('char ')[1][:-1])
                s = s[:error_position] + ':' + s[error_position:]
            elif "Expecting property name enclosed in double quotes" in error_message:
                error_position = int(error_message.split('char ')[1][:-1])
                s = s[:error_position] + '"' + s[error_position:]
            elif "Expecting value" in error_message:
                error_position = int(error_message.split('char ')[1][:-1])
                s = s[:error_position] + 'null' + s[error_position:]
            elif "Extra data" in error_message:
                error_position = int(error_message.split('char ')[1][:-1])
                s = s[:error_position]  # Supprime les données supplémentaires
            else:
                print("Erreur de décodage JSON non gérable:", e)
                return None
            attempt += 1
    print("Impossible de réparer le JSON après", max_attempts, "tentatives.")
    return None

def get_new_api_key():
    headers = {
        'authority': 'embeddings-dashboard-api.jina.ai',
        'accept': '*/*',
        'accept-language': 'en-US,en;q=0.9',
        'content-length': '0',
        'dnt': '1',
        'origin': 'https://jina.ai',
        'referer': 'https://jina.ai/',
        'sec-ch-ua': '"Not_A Brand";v="8", "Chromium";v="120", "Opera";v="106"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Linux"',
        'sec-fetch-dest': 'empty',
        'sec-fetch-mode': 'cors',
        'sec-fetch-site': 'same-site',
        'user-agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36 OPR/106.0.0.0',
    }
    response = requests.post('https://embeddings-dashboard-api.jina.ai/api/v1/api_key', headers=headers)
    if response.status_code == 200:
        return response.json().get('api_key')
    else:
        raise Exception("Failed to get new API key")

# Liste de vos clés API
hf_inference_api_keys = [ 
    
    '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'
]

# Créer un itérateur cyclique pour les clés API
api_key_iterator = itertools.cycle(hf_inference_api_keys)

class MixtralLLM(CustomLLM):
    context_window: int = 4096
    num_output: int = 512
    model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1" 
    api_key = next(api_key_iterator)
    Settings.callback_manager = CallbackManager([])

    @property
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.num_output,
            model_name=self.model_name,
        )

    def do_hf_call(self, prompt: str) -> str:
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512
            }
        }

        response = requests.post(
            'https://api-inference.huggingface.co/models/' + self.model_name,
            headers={
                'authorization': f'Bearer {self.api_key}',
                # 'authorization': f'Bearer {apiKey}',
                'content-type': 'application/json',
            },
            json=data,
            stream=True
        )
        if response.status_code != 200 or not response.json() or 'error' in response.json():
            print(f"Error: {response}")
            return "Unable to answer for technical reasons."
        full_txt = response.json()[0]['generated_text']
        offset = full_txt.find("---------------------")
        ss = full_txt[offset:]
        offset = ss.find("Answer:")
        return ss[offset+7:].strip()

    @llm_completion_callback()
    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        response = self.do_hf_call(prompt)
        return CompletionResponse(text=response)

    @llm_completion_callback()
    def stream_complete(
            self, prompt: str, **kwargs: Any
    ) -> CompletionResponseGen:
        response = ""
        for token in self.do_hf_call(prompt):
            response += token
            yield CompletionResponse(text=response, delta=token)


def load_file(file_path: str) -> List[Document]:
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.readlines()

    # Ajouter la ligne au début et à la fin
    if not content[0].startswith("*** START OF THE DATA FILE"):
        content.insert(0, "*** START OF THE DATA FILE\n")  # Ajouter au début

    if not content[-1].startswith("*** END OF THE DATA FILE"):
        content.append("\n*** END OF THE DATA FILE")  # Ajouter à la fin

    # Réécrire le fichier avec les modifications
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(content)

    ret: List[str] = []
    buff: str = ""
    reject: bool = True
    with open(file_path, 'r', encoding='utf-8') as file:
        for raw_line in file:
            line = raw_line
            stripped_line = line.strip()
            if reject:
                if stripped_line.startswith("*** START OF THE DATA FILE"):
                    reject = False
                    continue
            else:
                if stripped_line.startswith("*** END OF THE DATA FILE"):
                    reject = True
                    continue
                if stripped_line:
                    if stripped_line.startswith('=') and stripped_line.endswith('='):
                        ret.append(buff)
                        buff = ""
                        buff += stripped_line[1:len(stripped_line) - 1] + "\n\n"
                    else:
                        buff += line.replace('\r', '')
    if buff.strip():
        ret.append(buff)
    return StringIterableReader().load_data(ret)


def getPromptFromFile(file):
    with open(f"prompt/{file}.txt", 'r') as f:
        return f.read()


# def format_rename_data(rename_data):
#     formatted_data = ""
#     for item in rename_data:
#         formatted_data += f"Type: {item['item_type']}, Nom actuel: {item['old_name']}\n"
#     return formatted_data

def askToLlama3(question):
    response = requests.post('https://fumes-api.onrender.com/llama3',
                             json={
                                 'prompt': f"""{{
     "systemPrompt": "You are a cyber software annalist and you are talking to a software engineer",
     "user": "{question}"
}}""",
                                 "temperature": 0.75,
                                 "topP": 0.9,
                                 "maxTokens": 600

                             }, stream=False)
    print(response.text)

    json_data = repair_json(response.text)
    if json_data:
        print("JSON réparé :", json_data)
        return json_data
    else:
        print("Impossible de réparer le JSON.")


def do_hf_call(prompt: str) -> str:
    data = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 512
        }
    }
    response = requests.post(
        'https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1',
        headers={
            'authorization': f'Bearer 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',
            # 'authorization': f'Bearer {apiKey}',
            'content-type': 'application/json',
        },
        json=data,
        stream=True
    )

    # print("response", response.json())
    if response.status_code != 200 or not response.json() or 'error' in response.json():
        print(f"Error: {response}")
        return "Unable to answer for technical reasons."
    print("\n--- response.json() ---\n", response.json())
    full_txt = response.json()[0]['generated_text']
    print("\n--- full_txt ---\n", full_txt)
    json_response = full_txt.split("Answer:")[1].strip()
    print("\n--- json_response ---\n", json_response)

    json_response = repair_json(json_response)
    if json_response:
        # print("\n--- JSON réparé ---\n", json_response)
        return json_response
    else:
        print("Impossible de réparer le JSON.")
        return "Unable to repair JSON."


def chatbot(code, question, apiKey) -> Any:
    # jinaai_api_key = get_new_api_key()
    jina_embedding_model = JinaEmbedding(
        api_key=apiKey,
        model="jina-embeddings-v2-base-code",
    )

    mixtral_llm = MixtralLLM()

    Settings.llm = mixtral_llm
    Settings.embed_model = jina_embedding_model
    Settings.num_output = 512
    Settings.context_window = 4096

    code_escaped = code.replace("{", "{{").replace("}", "}}")
    # rebuild storage context
    if os.path.exists("historique"):
        storage_context = StorageContext.from_defaults(persist_dir="historique")
        # load index
        index_historique = load_index_from_storage(storage_context)
        # configure retriever
        retriever = VectorIndexRetriever(
            index=index_historique,
            similarity_top_k=5,
        )

        qa_prompt_tmpl = getPromptFromFile("chatbot")
        qa_prompt_tmpl = qa_prompt_tmpl.replace("{question}", question)

        MyRetrieveText = retriever.retrieve(qa_prompt_tmpl)
        concat = ""
        for i, rt in enumerate(MyRetrieveText):
            print(rt.text)
            concat += rt.text + "\n"

        qa_prompt_tmpl = qa_prompt_tmpl.replace("{historique}", concat)

        qa_prompt_tmpl = qa_prompt_tmpl.replace("{query_str}", str(code_escaped))

    else:
        qa_prompt_tmpl = getPromptFromFile("chatbot")
        qa_prompt_tmpl = qa_prompt_tmpl.replace("{question}", question)
        qa_prompt_tmpl = qa_prompt_tmpl.replace("{historique}", " ")
        qa_prompt_tmpl = qa_prompt_tmpl.replace("{query_str}", str(code_escaped))
        # load index
        # index_historique = load_index_from_storage(storage_context)

    print("Querying")
    result = do_hf_call(qa_prompt_tmpl)

    print(result)
    # stocker la question et le code ainsi que la réponse en base de données pour chaque conversation
    # Obtenez l'heure et la date actuelles
    now = datetime.datetime.now()

    # Formatez la date et l'heure sous forme de chaîne pour le nom de fichier
    # Vous pouvez modifier le format de date/heure selon vos besoins
    date_time_str = now.strftime("%Y%m%d_%H%M%S")  # Exemple : '20230405_143501' pour 5 avril 2023, 14:35:01

    # Créez le nom du fichier avec la date et l'heure
    if not os.path.exists("historiqueFile"):
        os.makedirs("historiqueFile")

    filename = "historiqueFile/historique" + date_time_str + ".txt"
    with open(filename, "a") as file:
        file.write(f"This is in the history of the conversation\n")
        file.write(f"Previous question: {question}\n")
        file.write(f"previous code: {str(code)}\n")
        file.write(f"Previous response: {result}\n\n")

    index_historique = VectorStoreIndex.from_documents([])
    for filename in os.listdir("historiqueFile"):
        pathname = os.path.join("historiqueFile", filename)
        docs = load_file(pathname)
        print(f"Processing with {len(docs)} documents")
        for doc in docs:
            print(f"Indexing")
            index_historique.insert(doc)
    index_historique.storage_context.persist(persist_dir="historique")

    print("Indexing done")
    print("=> ma rep : ", result)

    # TODO
    # result = str(result).split("Answer:")[1]
    # print(type(response))


    # response = result.replace("{{", "{").replace("}}", "}")  # Remplacer les doubles accolades par des simples
    # Utilisation

    # if is_json(result):
    #     print("C'est un JSON valide.")
    # else:
    #     print("Ce n'est pas un JSON valide.")
    #     result = repair_json(str(result))
    #     print("JSON réparé :", result)


    return str(result)



def is_json(myjson):
  try:
    json.loads(str(myjson))
  except ValueError as e:
    return False
  return True



def RenameFunctionAndVariables(data) -> Any:
    qa_prompt_tmpl = getPromptFromFile("first-turn")
    result = do_hf_call(qa_prompt_tmpl)
    print(result)
    qa_prompt_tmpl = getPromptFromFile("rename-function-multiturn")
    qa_prompt_tmpl = qa_prompt_tmpl.replace("{explication}", result)
    qa_prompt_tmpl = qa_prompt_tmpl.replace("{rename_list}", str(data["items"]))
    qa_prompt_tmpl = qa_prompt_tmpl.replace("{code_decompile}", str(data["code_c"]))

    print(qa_prompt_tmpl)

    result = do_hf_call(qa_prompt_tmpl)
    print(result)

    return result


import requests

def get_cvss_severity(id):
    print("\n--- id ---\n", id)
    url = f"https://services.nvd.nist.gov/rest/json/cves/2.0?cweId={id}&resultsPerPage=1&startIndex=0"

    response = requests.get(url).json()

    print(response)

    vulnerabilities = response.get("vulnerabilities", [{}])
    cve = vulnerabilities[0].get("cve", {})
    metrics = cve.get("metrics", {})

    cvss_metric_v31 = metrics.get("cvssMetricV31", [{}])
    cvss_data_v31 = cvss_metric_v31[0].get("cvssData", {})
    base_severity_v31 = cvss_data_v31.get("baseSeverity")

    if base_severity_v31 is None:
        cvss_metric_v2 = metrics.get("cvssMetricV2", [{}])
        cvss_data_v2 = cvss_metric_v2[0].get("cvssData", {})
        base_severity_v31 = cvss_metric_v2[0].get("baseSeverity")

    cvss_severity = base_severity_v31 or "unknown"

    return cvss_severity