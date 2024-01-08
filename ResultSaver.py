from pathlib import Path
from datetime import datetime as dt
import json

class ResultSaver:
    
    def __init__(self,
                 filename, 
                 keys,
                description = "No description",
                ):
        self.filename = filename
        if Path(self.filename).is_file()==False:
            date = dt.now().strftime("%m/%d/%Y")
            start_dict = {date : description}
            for key in keys:
                start_dict[key] = {}
            with open(filename,'w') as start:
                json.dump(start_dict, start)
        else:
            with open(filename,'r+') as file:
                file_data = json.load(file)
    
    def save_result(self, key, name, data):
        name = str(name)
        key = str(key)
        data = str(data)
        with open(self.filename,'r+') as file:
              # First we load existing data into a dict.
            file_data = json.load(file)
            
            if name in file_data[key]:
                done = False
                number = 0
                while not done:
                    if name+str(number) in file_data[key]:
                        number += 1
                    else:
                        name = name+str(number)
                        done = True
                        
            file_data[key][name] = data
            # Sets file's current position at offset.
            file.seek(0)
            # convert back to json.
            json.dump(file_data, file)
        return "Saved"