import torch
from torch.utils.data import Dataset
import json
import pandas as pd

# class defenition for dataset
class CheeseDescriptionsDataset(Dataset):
    def __init__(self, annotation_file, loader_pipeline):
        self.annot_file = annotation_file
        self.df = self.load_data(self.annot_file)
        self.pipeline = loader_pipeline

    
    def load_data(self, annot_file):
        with open(annot_file, 'r', encoding='utf-8') as f:
            lines = json.load(f)
            task_input, task_output = [], []
            for file in lines:
                line = lines[file]
                for rhet_tag in line:
                    text = line[rhet_tag]['text']
                    slots = line[rhet_tag]['slots']
                    formatted_slot = ''
                    for slot_key, slot_value in slots.items():
                        formatted_slot+= '<'+slot_key+':'+slot_value+'>'
                    task_input.append(text)
                    task_output.append(formatted_slot)
        
        data = {'input':task_input, 'output':task_output}
        df = pd.DataFrame(data=data, columns=['input','output'])
        return df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # return specific items
        row = self.df.iloc[index]
        return self.pipeline(row)
    