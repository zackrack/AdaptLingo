import re

def read_words_file(filename):
    with open(filename, 'r') as file:
        file_content = file.read().strip()
    
    # Split the file content on commas and/or newlines
    words_list = re.split(r'[,\n]+', file_content)
    
    # Strip extra spaces and surrounding quotes, and filter out empty strings
    words_list = [word.strip().strip('"').strip("'") for word in words_list if word.strip()]
    
    return words_list

def build_prompt(boost_words, user_input):
    system_message = f"""You are AdaptLingo, an English conversation partner who converses with English learners.
    The user will practice a conversation in English with you. You can also teach English to the user if they ask you
    questions about the language. 
    Answer the following user's questions in two sentences, and no more than two sentences. 
    The first sentence answers the user's quesiton or addresses the user prompt.
    If you think a question should be asked back to the user, which is not always true, the second sentence asks them back a question.
    If asking a question back to the user will throw the conversation off, do not ask back a question.
    Your sentences will be concise but conversational.
    Please use the following words to the best of your ability: {', '.join(boost_words)}.
    You do not need to use all of the words."""

    # Construct the final prompt
    return f"{system_message}\n\nUser: {user_input}\nAssistant:"
