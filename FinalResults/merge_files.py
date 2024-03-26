import json
import os
script_dir = os.path.dirname(os.path.realpath(__file__))
file_name = 'N2_aria.json'

file_path = os.path.join(script_dir, file_name)
with open(file_path, 'r') as file:
    full_data = json.load(file)


file_name = 'enhanced_N2_aria.json'
file_path = os.path.join(script_dir, file_name)
with open(file_path, 'r') as file:
    enhanced_data = json.load(file)

merged_data = full_data
merged_data['enhanced_results'] = enhanced_data['enhanced_results']
merged_data['enhanced_ids'] = enhanced_data['enhanced_ids']
merged_data['enhanced_depths'] = enhanced_data['enhanced_depths']
merged_data['enhanced_preprocessing'] = enhanced_data['hybrid_preprocessing_list']
merged_data['enhanced_preprocessing_depth'] = enhanced_data['hybrid_preprocessing_depth']

file_name = 'merged_N2_aria.json'
file_path = os.path.join(script_dir, file_name)
with open(file_path,'w') as file:
    json.dump(merged_data, file)
