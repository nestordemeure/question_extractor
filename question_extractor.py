import json
from pathlib import Path
from question_extractor import extract_questions_from_directory

# Define the input and output paths
input_directory = Path('./data/docs')
output_filepath = Path('./data/questions.json')

# replace the "API" with your own API key, you can provide multiply APIs in the list
API_KEYS = ["API"]

# Run the question extraction on the input directory
extracted_questions = extract_questions_from_directory(input_directory)
# Save the extracted questions as a JSON file
with open(output_filepath, 'w') as output_file:
    json.dump(extracted_questions, output_file, indent=4)
    print(f"Results have been saved to {output_filepath}.")