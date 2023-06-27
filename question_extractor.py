import json
import time
from pathlib import Path
from question_extractor import extract_questions_from_directory

# Define the input and output paths
input_directory = Path('./data/docs')
output_filepath = Path('./data/questions.json')

start_time = time.monotonic()
# Run the question extraction on the input directory
extracted_questions = extract_questions_from_directory(input_directory)
# Save the extracted questions as a JSON file
with open(output_filepath, 'w') as output_file:
    json.dump(extracted_questions, output_file, indent=4)
    print(f"Results have been saved to {output_filepath}.")

end_time = time.monotonic()
print(f"Took {end_time-start_time}")