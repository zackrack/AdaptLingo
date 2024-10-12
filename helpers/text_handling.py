def read_words_file(filename):
    # Open the file in read mode
    with open(filename, 'r') as file:
        # Read the file contents
        file_content = file.read()

    # Replace newlines, split by commas, and remove surrounding quotes
    words_list = file_content.replace('\n', '').split(',')
    words_list = [word.strip().strip('"').strip("'") for word in words_list]

    return words_list

def build_prompt(boost_words, user_input):
    system_message = f"""You are an English conversation partner who speaks concisely. 
    Answer the following user's questions in two sentences. 
    The first sentence answers the question and the second sentence asks them back a question. 
    Your sentences will be as short as possible while still being conversational.
    You are only allowed to use the following words: {', '.join(boost_words)}.
    You do not need to use all of the words."""

    # Construct the final prompt
    return f"{system_message}\n\nUser: {user_input}\nAssistant:"
