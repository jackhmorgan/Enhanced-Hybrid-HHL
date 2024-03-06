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
#cann_results = data["cann_results"]
#hybrid_results = data["hybrid_results"]
#enhanced_results = data["enhanced_results"]
#ideal_enhanced_results = data["ideal_enhanced_results"]
for key in data.keys():
    if key == 'lam':
        continue
    elif key == 'cann_results':
        label='Can.'
    elif key == 'hybrid_results':
        label='Hybrid'
    elif key == 'enhanced_results':
        label='Enhanced'
    else:
        continue
    plt.plot(lam_list, data[key], label=label)
    print(np.mean(data[key]))
# Create a line plot
#plt.plot(lam_list, cann_results, label="Cann.")
#plt.plot(lam_list, hybrid_results, label="Hybrid")
#plt.plot(lam_list, enhanced_results, label="Enhanced")
#plt.plot(lam_list, ideal_enhanced_results, label="Ideal Enhanced")

# Add labels and legend
plt.xlabel("$\lambda$")
plt.ylabel("Error")
plt.title("Error with fixed preprocessing")
plt.legend()

# Show the plot
plt.show()