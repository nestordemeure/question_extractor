import json
from pathlib import Path
from question_extractor import extract_questions_from_directory

# Define the input and output paths
input_directory = Path('./data/docs')
output_filepath = Path('./data/questions.json')

# Before running the code, one must replace the "API_KEY" in question_extractor/__init__.py with his own API key

# Run the question extraction on the input directory
extracted_questions = extract_questions_from_directory(input_directory)
# Save the extracted questions as a JSON file
with open(output_filepath, 'w') as output_file:
    json.dump(extracted_questions, output_file, indent=4)
    print(f"Results have been saved to {output_filepath}.")