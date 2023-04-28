from torch.utils.data import Dataset
import os
from  typing import Dict,List,Tuple

from collections import Counter
import torch
import numpy as np



def read_dataset(path: str) -> List[Dict]:
    samples: List[Dict] = []
    pron_counter = Counter()
    with open(path) as f:
        next(f)
        for line in f:
            (
                id,
                text,
                pron,
                p_offset,
                entity_A,
                offset_A,
                is_coref_A,
                entity_B,
                offset_B,
                is_coref_B,
                _,
            ) = line.strip().split("\t")
            pron_counter[pron.lower()] += 1
            samples.append(
                {
                    "id": id,
                    "text": text,
                    "pron": pron,
                    "p_offset": int(p_offset),
                    "entity_A": entity_A,
                    "offset_A": int(offset_A),
                    "is_coref_A": is_coref_A,
                    "entity_B": entity_B,
                    "offset_B": int(offset_B),
                    "is_coref_B": is_coref_B,
                }
            )

    return samples



class CoreferenceDataset(Dataset):

    def __init__(
        self, 
        tokenizer,
        modality: str, 
        data_path : str, 
        truncate_up_to_pron: bool=True, 
        labeled: bool=True,
        inference : bool = False
        
    ):

        modality = modality+".tsv"
        self.folder = os.path.join(data_path,modality)
        self.truncate_up_to_pron = truncate_up_to_pron
        self.labeled = labeled

        self.tokenizer = tokenizer






        #self.data = pd.read_csv(filepath_or_buffer=self.folder, sep="\t")
        if inference :
            pass
        else :  
            self.data = read_dataset(self.folder)
    

        
        self.pronoun = "<p>",
        self.A_start = "<a>",
        self.A_finish = "</a>",
        self.B_start =  "<b>",
        self.B_finish =  "</b>"

        CLS = [self.tokenizer.cls_token]
        SEP = [self.tokenizer.sep_token]

        self.CLS = CLS
        self.SEP = SEP

        if inference :
            pass
        else :            
            self.dataset = self.pre_processing(CLS,SEP)
        
    def pre_processing(self,CLS,SEP):

        dataset = []

    
        for i, row in enumerate(self.data):
            elements = dict()

            tokens, offsets = self.tokenize_sentence(row)

    
 
            
            pronoun = tokens[offsets["<p>"][0]]
            A_entity = tokens[offsets["<a>"][0]:offsets["</a>"][0]]
            B_entity = tokens[offsets["<b>"][0]:offsets["</b>"][0]]


            nothing = CLS + tokens + SEP + [pronoun, "is", "neither"] + SEP
            A_sentence = CLS + tokens + SEP + [pronoun, "is"] + A_entity + SEP
            B_sentence = CLS + tokens + SEP + [pronoun, "is"] + B_entity + SEP   
            
            
            list_alternatives = [nothing,A_sentence, B_sentence]
            tokens_list = []

            for instances in list_alternatives :
                tokens_list.append(self.tokenizer.convert_tokens_to_ids(instances))
            

            elements['tokens'] = tokens_list
            elements['offsets'] = self._get_offsets_list(offsets)



            #generate gt 
            if row['is_coref_A'] in ["TRUE", True]:
                elements['labels'] = 1

            elif row['is_coref_B'] in ["TRUE", True]:
                elements['labels'] = 2

            else:
                elements['labels'] = 0

            dataset.append(elements)         
        return dataset 
  
    def _get_offsets_list(self, offsets: Dict[str, List[int]]) -> List[int]:
        # 1 is added for the introduction of the CLS token
        offsets_A = [offsets["<a>"][0] + 1, offsets["</a>"][0] + 1]
        offsets_B = [offsets["<b>"][0] + 1, offsets["</b>"][0] + 1]
        
        return  [offsets["<p>"][0] + 1] + offsets_A + offsets_B
  
    def _insert_tag(self, text: str, offsets: Tuple[int, int], 
                    start_tag: str, end_tag: str = None) -> str:
        start_off, end_off = offsets 

        # Starting tag only
        if end_tag is None:
            text = text[:start_off] + start_tag + text[start_off:]
            return text

        text = text[:start_off] + start_tag + text[start_off:end_off] + end_tag + text[end_off:]
        return text

    def tokenize_sentence(self, row: Dict):
        tag_labels = {
            "pronoun_tag": "<p>",
            "start_A_tag": "<a>",
            "end_A_tag": "</a>",
            "start_B_tag": "<b>",
            "end_B_tag": "</b>"
        }
        
        tokens = []
        tag_labels = tag_labels
        offsets = {tag: [] for tag in tag_labels.values()}



        text = row['text']
        pronoun = row['pron']
        A_entity = row['entity_A']
        B_entity = row['entity_B']

        # Sort the offsets in ascending order
        break_points = sorted([
            (tag_labels["pronoun_tag"], row['p_offset']),
            (tag_labels["start_A_tag"], row['offset_A']),
            (tag_labels["end_A_tag"], row['offset_A'] + len(A_entity)),
            (tag_labels["start_B_tag"], row['offset_B']),
            (tag_labels["end_B_tag"], row['offset_B'] + len(B_entity)),
        ], key=lambda x: x[1])

        # When a new tag is inserted, the offset of the next tag
        # changes by the length of the inserted tag.
        len_added_tags = 0
        for tag, offset in break_points:
            offset += len_added_tags
            text = self._insert_tag(text, (offset, None), tag)
            len_added_tags += len(tag)

        # Truncate the text at the last tag inserted and append the pronoun at the end
        if self.truncate_up_to_pron:
            text = text[:offset+len(tag)] + pronoun

        # Also the tags are added to the tokens
        for token in self.tokenizer.tokenize(text):    
            tokens.append(token)

            if token in [*tag_labels.values()]:
                if "/" in token: # End token
                    offsets[token].append(len(tokens)-1)
                else:
                    offsets[token].append(len(tokens)) 
        

     
        
        return tokens, offsets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def prapare_batch (self,senteces,device):
        

        batch = []
      


        for i, instance in enumerate(senteces):
            elements = dict()

            tokens, offsets = self.tokenize_sentence(instance)


            
            pronoun = tokens[offsets["<p>"][0]]
            A_entity = tokens[offsets["<a>"][0]:offsets["</a>"][0]]
            B_entity = tokens[offsets["<b>"][0]:offsets["</b>"][0]]


            nothing = self.CLS + tokens + self.SEP + [pronoun, "is", "neither"] + self.SEP
            A_sentence = self.CLS + tokens + self.SEP + [pronoun, "is"] + A_entity + self.SEP
            B_sentence = self.CLS + tokens + self.SEP + [pronoun, "is"] + B_entity + self.SEP   
            
            
            list_alternatives = [nothing,A_sentence, B_sentence]
            tokens_list = []

            for instances in list_alternatives :
                tokens_list.append(self.tokenizer.convert_tokens_to_ids(instances))
            

            elements['tokens'] = tokens_list
            elements['offsets'] = self._get_offsets_list(offsets)
            batch.append(elements)  

        #GET MAX LENGHT
        #total number of lists = batch_size x 3
        pad: int=0
        truncate: int=512

        input = {}
        list_ = []
        batch_size = len(batch)
        total_n_sequences = batch_size*3
        
        for samples in batch :
            list_.append(samples["tokens"][0])
            list_.append(samples["tokens"][1])
            list_.append(samples["tokens"][2])
    

        max_len = min(max((len(x) for x in list_)),truncate)

        
        
        zero_padding = np.full((total_n_sequences, max_len), pad, dtype=np.int64)


        #insert each token sequence in the geneted sentence
        for i,tokens in enumerate(list_):
            lenght_original_tonized_sequnce = len(tokens)
            zero_padding[i,:lenght_original_tonized_sequnce] = tokens
        



        tokens_padded = torch.tensor(zero_padding, device=device)
        tokens_padded = tokens_padded.view(batch_size,3,max_len)
        

        input["tokens"] = tokens_padded






        return input
        
