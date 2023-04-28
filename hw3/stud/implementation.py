import re
import numpy as np
from typing import List, Tuple, Dict
from model import Model
from transformers import AutoTokenizer,AutoModel
from .dataset import CoreferenceDataset
import torch.nn as nn
import torch


def build_model_123(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2 and 3 of the Coreference resolution pipeline.
            1: Ambiguous pronoun identification.
            2: Entity identification
            3: Coreference resolution
    """
    return RandomBaseline(True, True)


def build_model_23(device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2 and 3 of the Coreference resolution pipeline.
            2: Entity identification
            3: Coreference resolution
    """
    return RandomBaseline(False, True)


def build_model_3(device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements step 3 of the Coreference resolution pipeline.
            3: Coreference resolution
    """
    return StudentModel(False, False)


class RandomBaseline(Model):
    def __init__(self, predict_pronoun: bool, predict_entities: bool):
        self.pronouns_weights = {
            "his": 904,
            "her": 773,
            "he": 610,
            "she": 555,
            "him": 157,
        }
        self.predict_pronoun = predict_pronoun
        self.pred_entities = predict_entities

    def predict(
        self, sentences: List[Dict]
    ) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:
        predictions = []
        for sent in sentences:
            text = sent["text"]
            toks = re.sub("[.,'`()]", " ", text).split(" ")
            if self.predict_pronoun:
                prons = [
                    tok.lower() for tok in toks if tok.lower() in self.pronouns_weights
                ]
                if prons:
                    pron = np.random.choice(prons, 1, self.pronouns_weights)[0]
                    pron_offset = text.lower().index(pron)
                    if self.pred_entities:
                        entity = self.predict_entity(
                            predictions, pron, pron_offset, text, toks
                        )
                    else:
                        entities = [sent["entity_A"], sent["entity_B"]]
                        entity = self.predict_entity(
                            predictions, pron, pron_offset, text, toks, entities
                        )
                    predictions.append(((pron, pron_offset), entity))
                else:
                    predictions.append(((), ()))
            else:
                pron = sent["pron"]
                pron_offset = sent["p_offset"]
                if self.pred_entities:
                    entity = self.predict_entity(
                        predictions, pron, pron_offset, text, toks
                    )
                else:
                    entities = [
                        (sent["entity_A"], sent["offset_A"]),
                        (sent["entity_B"], sent["offset_B"]),
                    ]
                    entity = self.predict_entity(
                        predictions, pron, pron_offset, text, toks, entities
                    )
                predictions.append(((pron, pron_offset), entity))
        return predictions

    def predict_entity(self, predictions, pron, pron_offset, text, toks, entities=None):
        entities = (
            entities if entities is not None else self.predict_entities(entities, toks)
        )
        entity_idx = np.random.choice([0, len(entities) - 1], 1)[0]
        return entities[entity_idx]

    def predict_entities(self, entities, toks):
        offset = 0
        entities = []
        for tok in toks:
            if tok != "" and tok[0].isupper():
                entities.append((tok, offset))
            offset += len(tok) + 1
        return entities

class StudentModel(Model):

    def __init__(self, predict_pronoun: bool, predict_entities: bool):


        self.predict_pronoun = predict_pronoun
        self.pred_entities = predict_entities
        


        delimitaror = ["<a>","</a>","<b>","</b>","<p>"]
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased",never_split=delimitaror)
        tokenizer.add_tokens(delimitaror,special_tokens=True)

        PATH_BERT = "hw3/stud/saved/model_bert.pth"
        
        #if only model
        auto_model = torch.load(PATH_BERT,map_location=torch.device('cpu'))
        auto_model.eval()

        #if saved with config file
        #auto_model = AutoModel.from_pretrained("bert-base-cased",output_hidden_states=True)
        #auto_model = AutoModel.from_pretrained(PATH_BERT)

        self.dataset = CoreferenceDataset(tokenizer,"train","",inference=True)
        config = {}

        self.model = ConferenceResolution(auto_model,tokenizer,config)
        

        PATH = "hw3/stud/saved/model.pth"
        self.model.load_state_dict(torch.load(PATH,map_location=torch.device('cpu')))
        #self.model = torch.load(PATH,map_location=torch.device('cpu'))

        self.model.eval()
        
    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary

    def predict(
        self, sentences: List[Dict]
    ) -> List[Tuple[Tuple[str, int], Tuple[str, int]]]:

        predictions = []
        #print("instances",sentences)
        input_model = self.dataset.prapare_batch(sentences,"cpu")
        #print("input_model",input_model["tokens"].size())
        with torch.no_grad():
            #False indicate that is inference mode
            output,_ = self.model(input_model,False)
            predicted = torch.argmax(output, dim=1)
        
        #print(output)
        #print(predicted)

        for i,sentence in enumerate (sentences) :

            pron = sentence["pron"]
            pron_offset = sentence["p_offset"]
            
            entities = [
            (sentence["entity_A"], sentence["offset_A"]),
            (sentence["entity_B"], sentence["offset_B"]),
                    ]
            idx = predicted[i]
            if idx == 0:
                "noting"
                entity = ()
                
            elif idx == 1 :
                entity = entities[0]
            elif idx == 2:
                entity = entities[1]

            predictions.append(((pron, pron_offset),entity))


  

        return predictions

class ConferenceResolution(nn.Module):

    def __init__(self, bert_model,tokenizer,config):
        super().__init__()
        # note  : config is not needed with this deployment case
        self.bert_model = bert_model

        self.bert_model.resize_token_embeddings(len(tokenizer.vocab))
        self.criterion= torch.nn.CrossEntropyLoss()

        self.normalize = nn.BatchNorm1d(768)
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(768, 1)


        self.classifier0 = nn.Linear(768, 768)
        self.classifier1 = nn.Linear(768, 1)
        self.relu = nn.ReLU()
        self.dropout0 = nn.Dropout(0.3)




    def forward(self, sample,training = True):
        #[Batch,3,Tokens]
        bert_input = sample['tokens']
        
        b,_,l = bert_input.size()

        #[Batch*3,Tokens]
        bert_input = bert_input.view(-1,l)

    
        #[Batch*3,Tokens,hidden_size]
        bert_outputs = self.bert_model(bert_input, attention_mask=(bert_input > 0).long(),token_type_ids=None, output_hidden_states=True)

        out = bert_outputs.pooler_output

        #from CLS extraction
        #[Batch*3,1,hidden_size]
        #[Batch*3,hidden_size]
        pooled_output = self.normalize(out)
        pooled_output = self.dropout(pooled_output)
        
        #[Batch*3]
        #logits = self.classifier(pooled_output)
        logits = self.classifier0(pooled_output)
        logits = self.relu(logits)
        logits = self.dropout0(logits)
        logits = self.classifier1(logits)

        #[Batch,3]
        output = logits.view(-1, 3)
        loss = None

        if training:
            labels = sample['labels']
            loss = self.criterion(output, labels)

        return output,loss