import json
import os
from collections import defaultdict

def merge_json_files(file_list):
    merged_dict = defaultdict(list)
    num_problems = 0
    for file_name in file_list:
        with open(file_name, 'r') as file:
            data = json.load(file)
            for key, value in data.items():
                if isinstance(value, list):
                    if not key in merged_dict.keys():
                        merged_dict[key].extend(value)
                    elif (not value == []) and 'result' in key:
                        if key in merged_dict.keys():
                            for problem, result in enumerate(value):
                                if problem > num_problems:
                                    for outcome in result.keys():
                                        merged_dict[key][problem%num_problems][outcome] += result[outcome]
            num_problems+=9
    return dict(merged_dict)

file_name = 'simulator_to_deneb_N2_matrix_hhl'
script_dir = os.path.dirname(os.path.realpath(__file__))
# Define the file path
file_path = os.path.join(script_dir, file_name)
file_list = [file_path+str(iteration)+'.json' for iteration in range(5)]
merged_data = merge_json_files(file_list)

# Print or save the merged data
print(json.dumps(merged_data, indent=4))

# Optionally, write the merged data to a new JSON file
with open(file_path+'_merged2.json', 'w') as output_file:
    json.dump(merged_data, output_file, indent=4)