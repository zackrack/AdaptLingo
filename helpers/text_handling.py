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
    system_message = f"""You are an English conversation partner who speaks concisely. 
    Answer the following user's questions in three sentences maximum. 
    The first sentence answers the question and the second sentence asks them back a question. 
    Your sentences will be as short as possible while still being conversational.
    You are only allowed to use the following words: {', '.join(boost_words)}.
    You do not need to use all of the words."""

    # Construct the final prompt
    return f"{system_message}\n\nUser: {user_input}\nAssistant:"
