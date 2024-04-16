def getPromptFromFile(file):
    with open(f"prompt/{file}.txt", 'r') as f:
        return f.read()


def format_rename_data(rename_data):
    formatted_data = ""
    for item in rename_data:
        formatted_data += f"Type: {item['item_type']}, Nom actuel: {item['old_name']}\n"
    return formatted_data