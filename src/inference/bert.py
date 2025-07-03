import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


class BertInferencePipeline:
    def __init__(self, model, config=None, device=None):
        self.model = model
        self.config = config
        self.device = device
        self.problem_type = self.config['problem_type']
        
        self.is_classification = 'classification' in self.problem_type
        
        if self.is_classification:
            self.softmax = torch.nn.Softmax(dim=1)
    
    @torch.no_grad()
    def run(self, data_loader):
        preds = []
        labels = []
        patient_ids = []
        pred_probas = []
        
        for batch in tqdm(data_loader):
            outputs = self.model(**{key:batch[key].to(self.device) for key in batch if key != 'person_id'})

            if self.problem_type == 'regression':
                predictions = outputs.logits.cpu().numpy()
                batch_pred = predictions.tolist()
            else:
                predictions = self.softmax(outputs.logits).cpu().numpy()
                batch_pred = (predictions[:, 1] > self.config['threshold']).astype(int).tolist()
                pred_probas.extend(batch_pred)

            preds.extend(batch_pred)
            
            batch_labels = torch.squeeze(batch['labels'], axis=1).tolist()
            labels.extend(torch.squeeze(batch['labels'], axis=1).tolist())

            patient_ids.extend(batch['person_id'])
        
        result = {
            'patient_id': patient_ids,
            'predictions': preds, 
            'labels': labels,
        }
        
        if self.is_classification:
            result['pred_probabilities'] = np.array(pred_probas)

        predictions_df = pd.DataFrame(result)

        return predictions_df
