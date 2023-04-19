import json
from pathlib import Path
from question_extractor import question_extractor

# where is the data located
input_folder = Path('./data/docs')
output_file = Path('./data/questions.json')

# runs the question extraction
questions = question_extractor(input_folder)

# saves the result as a json file
with open(output_file, 'w') as file:
    file.write(json.dumps(questions, indent=4))
    print(f"Results have been saved to {output_file}.")