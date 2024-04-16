from time import sleep

from flask import Flask
from flask import request

from src.feature import *
from src.user import anthony

# TODO:
#
# ❌ CVSS ranking


app = Flask(__name__)


@app.route('/')
def route_home():
    return 'Hello, World!'


@app.route('/renameFunction', methods=['POST'])
def route_rename_function():
    rename_function_prompt = getPromptFromFile("rename-function")

    data = request.json
    print(f"Data : \n{data}\n\n")

    items = data["items"]
    code_c_json = ' '.join(map(str, data["code_c"].values()))

    formated_rename_data = format_rename_data(items)


    full_prompt = rename_function_prompt.replace("{rename_list}", formated_rename_data)
    full_prompt = full_prompt.replace("{code_decompile}", code_c_json)
    print(f"Prompt : \n{full_prompt}\n\n")

    return full_prompt


# @app.route('/renameVariable', methods=['POST'])
# def route_rename_variable():
#     data = request.json
#     code_c_json = data["code_c"]
#     rename_variable_prompt = getPromptFromFile("rename-variable")
#     formated_rename_data = format_rename_data(data).replace("{code_decompile}", code_c_json)
#     full_prompt = rename_variable_prompt.replace("{rename_list}", formated_rename_data)
#
#     print(f"Data : \n{data}\n\n")
#     print(f"Prompt : \n{full_prompt}\n\n")
#     # return ask_question(data)

@app.route('/chatbot', methods=['POST'])
def root_chatbot():
    chatbot_prompt = getPromptFromFile("chatbot")

    data = request.json
    print(f"Data : \n{data}\n\n")

    # items = data["items"]
    code_c_json = ' '.join(map(str, data["code_c"].values()))

    question = data["question"]

    # formated_rename_data = format_rename_data(items)

    full_prompt = chatbot_prompt.replace("{question}", question)
    full_prompt = full_prompt.replace("{code_decompile}", code_c_json)
    print(f"Prompt : \n{full_prompt}\n\n")

    return anthony(code_c_json, question)

@app.route('/renameVariableAndFunction', methods=['POST'])
def route_rename_variable_and_function():
    print("RenameVariableAndFunction")
    print("Request : {}".format(request))
    data = request.json
    print(f"Data : \n{data}\n\n")
    return {"rename": [["variable", "local_18", "TEST1"]]}


if __name__ == '__main__':
    app.run()
