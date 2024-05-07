import shutil
from flask import Flask
from flask import request
import chromadb

from src.utils import *

app = Flask(__name__)

apiKey = ""

@app.route('/')
def route_home():
    return 'Hello, World!'

@app.route('/analyze', methods=['POST'])
def analyse():
    try:
        post_data = request.json
        print(f"Received data: {post_data}")

        code_c_json = json.dumps(post_data['code_c'])

        response = ""

        if post_data['rag']:

            apiKey = post_data['apiKey']
            print("API Key : " + apiKey)

            client = chromadb.PersistentClient(path="chroma_db")

            # On crée la collection
            collection = client.get_collection(name="quickstart")

            embed_model = JinaEmbedding(
                api_key=get_new_api_key(),
                model="jina-embeddings-v2-base-code",
            )

            embeddings = embed_model.get_query_embedding(code_c_json)

            query = collection.query(
                query_embeddings=embeddings,
                n_results=5
            )
            query_parts = []
            for i, meta in enumerate(query['metadatas'][0]):
                cwe_id = meta['CWE-id']

                result_str = f"CWE-id: {query['metadatas'][0][i]['CWE-id']}, CWE-name: {query['metadatas'][0][i]['name']}, CWE-description: {query['metadatas'][0][i]['description']}"

                if query['metadatas'][0][i]['isVulnerable']:
                    result_str += f", Ce code est vulnérable à la CWE {cwe_id}\n"
                else:
                    result_str += f", Ce code n'est pas vulnérable à la CWE {cwe_id}\n"
                result_str += f"code: {query['metadatas'][0][i]['code']}, IsVulnerable: {query['metadatas'][0][i]['isVulnerable']}"
                # print("result" + str(i) + result_str)
                query_parts.append(result_str)

            concatenated_query = "\n".join(query_parts)
            context = json.dumps(concatenated_query, indent=2)

            # Replace placeholders in the analyze_prompt string
            analyze_prompt = getPromptFromFile("analyze")
            full_prompt = analyze_prompt.replace("{code}", code_c_json).replace("{information}", context)
            print("RAG is true")
            # print(full_prompt)

            # response = askToLlama3(full_prompt)
            response = do_hf_call(full_prompt)

        elif not post_data['rag']:

            analyze_prompt = getPromptFromFile("analyzeWithoutRag")
            full_prompt = analyze_prompt.replace("{code}", code_c_json)
            print("RAG is false")

            response = do_hf_call(full_prompt)

        else:
            return "Invalid rag parameter", 400

        print("\n--- Before CVSS ---\n",response)

        # for line, id, comment in response['comment']:
        #     cvss_severity = get_cvss_severity(id)
        #     if cvss_severity is None:
        #         cvss_severity = "unknown"
        #     # appende cvss_severity to actual element
        #     response['comment'] = response['comment'] + cvss_severity

        for i in range(len(response['comment'])):
            cvss_severity = get_cvss_severity(response['comment'][i][1])
            if cvss_severity is None:
                cvss_severity = "unknown"
            response['comment'][i] = response['comment'][i] + [cvss_severity]


        print("\n--- After CVSS ---\n", response)
        return response
    except Exception as e:
        return {"error": str(e)}, 500


@app.route('/chatbot', methods=['POST'])
def root_chatbot():
    try:
        data = request.json
        print(f"Data : \n{data}")
        apiKey = data["apiKey"]
        print("API Key : " + apiKey)

        # items = data["items"]
        if "code_c" in data:
            code_c_json = ' '.join(map(str, data["code_c"].values()))
        else:
            code_c_json = "No code in this request"

        question = data["question"]
        response = str(chatbot(code_c_json, question, apiKey))
        print(f"Chatbot : \n{response}\n\n")
        return response
    except Exception as e:
        return {"error": str(e)}


@app.route('/renameVariableAndFunction', methods=['POST'])
def route_rename_variable_and_function():
    try:
        print("RenameVariableAndFunction")
        print("Request : {}".format(request))
        data = request.json
        print(f"Data : \n{data}\n\n")
        return RenameFunctionAndVariables(data)
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/clear', methods=['POST'])
def clear():
    try:
        history_dir = "historique"
        history_file_dir = "historiqueFile"
        print("Clearing history")
        # Check if directories exist before trying to delete them
        if os.path.exists(history_dir):
            shutil.rmtree(history_dir)
        if os.path.exists(history_file_dir):
            shutil.rmtree(history_file_dir)

        return "History cleared", 200
    except Exception as e:
        return {"error": str(e)}, 500


if __name__ == '__main__':
    print(get_new_api_key())
    app.run()
