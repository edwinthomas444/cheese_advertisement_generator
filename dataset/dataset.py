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
                    # Try training on only 1 task
                    # if rhet_tag == 'identification':
                    text = line[rhet_tag]['text']
                    slots = line[rhet_tag]['slots']
                    formatted_slot = ''
                    for slot_key, slot_value in slots.items():
                        formatted_slot+= '<'+slot_key+':'+slot_value+'>'
                    task_output.append(text)
                    task_input.append(formatted_slot)
            # print('\n len of task output: ', len(task_output))
        
        data = {'input':task_input, 'output':task_output}
        df = pd.DataFrame(data=data, columns=['input','output'])
        # df.to_csv('results/gt.csv')
        return df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # return specific items
        row = self.df.iloc[index]
        # print('\n',row['input'])
        return self.pipeline(row)
    
# Instance-wise inference Dataset
class CheeseDescriptionsTemplateDataset(Dataset):
    def __init__(self, template_file, loader_pipeline):
        self.template_file = template_file
        self.df = self.load_data(self.template_file)
        self.pipeline = loader_pipeline

    def load_data(self, template_file):
        task_output = []
        task_input = []
        with open(template_file, 'r', encoding='utf-8') as f:
            template = json.load(f)
            text = ''
            for rhet_tag in template:
                slots = template[rhet_tag]
                formatted_slot = ''
                for slot_key, slot_value in slots.items():
                    formatted_slot+= '<'+slot_key+':'+slot_value+'>'
                task_output.append(text)
                task_input.append(formatted_slot)
            # print('\n len of task output: ', len(task_output))
        
        data = {'input':task_input, 'output':task_output}
        df = pd.DataFrame(data=data, columns=['input','output'])
        df.to_csv('results/single_test.csv')
        return df

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        # return specific items
        row = self.df.iloc[index]
        # print('\n',row['input'])
        return self.pipeline(row)
    
