from typing import Any
import requests
import itertools
import os

from llama_index.core import SimpleDirectoryReader, SummaryIndex
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import (
    CustomLLM,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core import Settings


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

def retrieve_number():
    cwe_numbers = []
    for filename in os.listdir("CWE-Test-Extended"):
        # Vérifier si le fichier est un fichier .c
        if filename.endswith(".c"):
            # Extraire le numéro de CWE du nom du fichier
            cwe_number = filename.split("-")[2].split(".")[0]
            # Ajouter le numéro de CWE à la liste
            cwe_numbers.append(int(cwe_number))
            cwe_numbers.sort()
    return cwe_numbers

def increment_counter_if_cwe_found(total, counter, cwe_number, text):

    # Construire la chaîne de recherche
    cwe_pattern = f"CWE-{cwe_number}"
    total[0] += 1
    # Vérifier si la chaîne de recherche est dans le texte
    if cwe_pattern in text:
        # Incrementer le compteur
        counter[0] += 1
        print(f"{cwe_pattern} OK")
        return 3
    else:
        print(f"{cwe_pattern} NOT OK")