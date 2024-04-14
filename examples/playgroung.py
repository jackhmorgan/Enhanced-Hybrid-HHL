import json
import os
import numpy as np
script_dir = os.path.dirname(os.path.realpath(__file__))
file_name = 'torino_small_matrix_preprocessing.json'

# Define the file path
file_path = os.path.join(script_dir, file_name)
with open(file_path, 'r') as file:
    json_data = json.load(file)

fixed_depths = json_data['fixed_depth']
enhanced_fixed_depths = json_data['enhanced_fixed_depth']

print(np.average(fixed_depths))
print(np.average(enhanced_fixed_depths))
