import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Get the directory of the current script
script_dir = os.path.dirname(os.path.realpath(__file__))

# Define the filename
file_name = "example_figure100ideal.json"

# Define the file path
file_path = os.path.join(script_dir, file_name)

# Load the data from the JSON file
with open(file_path, "r") as json_file:
    data = json.load(json_file)

# Extract lists from the loaded data
lam_list = data["lam"]

for key in data.keys():
    if key == 'lam':
        continue
    elif key == 'Can.':
        label='Can.'
    elif key == 'Hybrid':
        label='Hybrid'
    elif key == 'Enhanced':
        label='Enhanced'
    else:
        continue
    plt.plot(lam_list, data[key], label=label)
    print(np.mean(data[key]))

# Add labels and legend
plt.xlabel("$\lambda$")
plt.ylabel("Error")
plt.title("Error with fixed preprocessing")
plt.legend()

# Show the plot
plt.show()