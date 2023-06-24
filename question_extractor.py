import os
import json
from pathlib import Path
from question_extractor import extract_questions_from_directory

os.environ['OPENAI_API_KEY'] = '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'

# Define the input and output paths
input_directory = Path('./data/input')
output_filepath = Path('./data/questions.json')

# Run the question extraction on the input directory
extracted_questions = extract_questions_from_directory(input_directory)

# Save the extracted questions as a JSON file
with open(output_filepath, 'w') as output_file:
    json.dump(extracted_questions, output_file, indent=4)
    print(f"Results have been saved to {output_filepath}.")
