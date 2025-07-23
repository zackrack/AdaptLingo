import re

def extract_words_from_file(file_path, output_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Match only the English headwords (column after "No" and number)
    pattern = r'\d+\s+\n(.+?)\n'
    words = re.findall(pattern, text)

    # Clean up whitespace and join
    clean_words = [word.strip() for word in words]
    comma_separated = ', '.join(clean_words)

    # Write result to file
    with open(output_path, 'w', encoding='utf-8') as out_file:
        out_file.write(comma_separated)

# Example usage
input_file = 'your_text_file.txt'
output_file = 'advanced_words_eiken1.txt'
extract_words_from_file(input_file, output_file)
