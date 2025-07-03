import itertools

import numpy as np

import torch
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence


class CustomDataset(IterableDataset):
    """
    An object that handles the loading of data
    from dataframe contraining tables formatted like:

    person_id | sorted_event_tokens                                                                      | day_position_tokens  |
    0         | ["AGE:25", "ETHNICITY:UNK", "GENDER:F", "HCPCS:pr_a", "ICD10CM:dx_a", "LOINC:lx_a:LOW"]  | [0, 0, 0, 1, 2, 3]   |
    1         | ["AGE:26", "ETHNICITY:UNK", "GENDER:F", "HCPCS:pr_a", "ICD10CM:dx_a", "LOINC:lx_a:LOW"]  | [0, 0, 0, 1, 1, 1]   |
    ...
    """
    def __init__(
        self,
        dataframe,
        tokenizer,
        max_length=512,
        include_person_ids=False,
        shuffle=False,
    ):
        """
        Args:
            dataframe (df.DataFrame):
            tokenizer (tokenizers.models.Model):
                The tokenizer model to handle tokenization
            max_length (int):
                The sequence length to pad all outputs to. e.g.
                the max length for BERT is 512, so set it accordingly.
            include_person_ids:
                Whether to include person ids in the loaded data
        """
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

        self.include_person_ids = include_person_ids
        self.shuffle = shuffle
        super().__init__()

    def __len__(self):
        return self.dataframe.shape[0]

    def __iter__(self):
        """
        Iterate through the entire dataset, returning entries one-by-one
        """
        for i, row in self.dataframe.iterrows():
            inputs = row["sorted_event_tokens"]
            position_ids = row["day_position_tokens"]
            if inputs[0] != "[CLS]":
                inputs = itertools.chain(
                    ["[CLS]"], inputs
                )  # add the cls token and encode
                position_ids = [0] + list(position_ids)  # account for the cls token
            encoded_inputs = np.array(
                [self.tokenizer.tokenize(el)[0].id for el in inputs]
            )
            encoded_inputs = encoded_inputs[
                0 : min(self.max_length, len(encoded_inputs))
            ]
            position_ids = position_ids[
                0 : min(self.max_length, len(encoded_inputs))
            ]
            to_return = {
                "input_ids": encoded_inputs,
                "attention_mask": np.ones(len(encoded_inputs), np.int8),
                "position_ids": position_ids,
            }
            if self.include_person_ids:
                to_return["person_id"] = [row["person_id"]]
                
            yield to_return


class LabelledDataset(CustomDataset):
    """
    A class to iterate through a dataset of labelled dataset for training a BERT model.
    """
    def __init__(self, labelled_data_column_name, *args, **kwargs):
        self.labelled_data_column = labelled_data_column_name
        super().__init__(*args, **kwargs)

    def __iter__(self):
        """
        Iterate through the entire dataset, returning entries one-by-one
        """
        for i, row in self.dataframe.iterrows():
            inputs = row["sorted_event_tokens"]
            position_ids = row["day_position_tokens"]
            if inputs[0] != "[CLS]":
                inputs = itertools.chain(
                    ["[CLS]"], inputs
                )  # add the cls token and encode
                position_ids = [0] + list(position_ids)  # account for the cls token
            encoded_inputs = np.array(
                [self.tokenizer.tokenize(el)[0].id for el in inputs]
            )
            encoded_inputs = encoded_inputs[
                0 : min(self.max_length, len(encoded_inputs))
            ]
            position_ids = position_ids[
                0 : min(self.max_length, len(encoded_inputs))
            ]

            labels = np.array([row[self.labelled_data_column]])
            
            to_return = {
                "input_ids": encoded_inputs,
                "attention_mask": np.ones(len(encoded_inputs), np.int8),
                "position_ids": position_ids,
                "labels": labels.astype(np.float32) if labels.dtype == np.float64 else labels,
            }
            
            if self.include_person_ids:
                to_return["person_id"] = [row["person_id"]]
            
            yield to_return


class LabelledDataCollator:
    """
    A data collator that takes a set of data points from the CustomDataset defined above.
    It collates them into a batch and pads accordingly.
    """
    def __call__(self, lst_of_points):
        input_ids = [torch.tensor(item["input_ids"]) for item in lst_of_points]
        attention_mask = [
            torch.tensor(item["attention_mask"]) for item in lst_of_points
        ]
        position_ids = [torch.tensor(item["position_ids"]) for item in lst_of_points]
        labels = [torch.tensor(item["labels"]) for item in lst_of_points]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0)
        labels = torch.stack(labels)
                
        to_return = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
        }

        if "person_id" in lst_of_points[0]:
            person_ids = []
            for point in lst_of_points:
                person_ids += point["person_id"]
            to_return["person_id"] = person_ids
                
        return to_return


class PretrainingDataCollator:
    """
    A "data collator" in the hugging-face library is a
    class that is meant to merge instances of data into a batch.
    This data collator also handles the masking
    of input tokens for masked language modelling.
    """

    def __init__(
        self,
        tokenizer,
        tokenizer_vocabulary,
        mask_token="[MASK]",
        prediction_fraction=0.15,
        masking_fraction=0.8,
        random_replacement_fraction=0.1,
        override_maxlen=None,
    ):
        """
        tokenizer (tokenizers.models.Model):
            The tokenizer model to handle tokenization
        tokenizer_vocabulary (dict[string]:int):
            A dictionary of keys that are the tokens in
            the vocabulary, and the values are the token id.
        mask_token:
            the masking token in the vocabulary, normally "[MASK]"
        prediction_fraction:
            The fraction of tokens to perform
            masked-language-modelling for
        masking_fraction:
            The fraction of the masked-language-modelling
            tokens that are replaced by the mask token
        random_replacement_fraction:
            The fraction of the masked-language-mdelling
            tokens that are replaced by random words.
        """

        self.tokenizer = tokenizer
        self.tokenizer_vocabulary = tokenizer_vocabulary

        # get all non special token ids, not including the [mask] or [unk] tokens, for example.
        self.nonspecial_token_ids = [
            token_id
            for token, token_id in tokenizer_vocabulary.items()
            if token[0] != "[" and token[-1] != "]"
        ]

        self.prediction_fraction = (
            prediction_fraction  # percentage of tokens to perform MLM for
        )

        # fraction of MLM tokens replaced by the mask token
        self.masking_fraction = masking_fraction
        # fraction of MLM tokens replaced by a random token
        self.random_replacement_fraction = random_replacement_fraction

        # fraction of MLM tokens not replaced by their original word
        self.unchanged = 1.0 - self.masking_fraction - self.random_replacement_fraction
        assert self.unchanged >= 0.0

        self.mask_token_id = self.tokenizer.tokenize(mask_token)[0].id

        self.override_maxlen = override_maxlen

        super().__init__()

    def __call__(self, list_of_data):
        length_cutoff = max(len(el["input_ids"]) for el in list_of_data)
        if self.override_maxlen is not None:
            length_cutoff = self.override_maxlen

        inputs = np.zeros((len(list_of_data), length_cutoff), dtype=int)
        position_ids = np.zeros((len(list_of_data), length_cutoff), dtype=int)
        attention_mask = np.zeros((len(list_of_data), length_cutoff), dtype=int)

        labels = (
            np.zeros(inputs.shape, dtype=int) - 100
        )  # set the labels to -100 by default (i.e. don't perform MLM on these tokens)

        for i, data in enumerate(list_of_data):
            encoded_inputs = data["input_ids"]
            one_position_id_list = data["position_ids"]
            assert len(encoded_inputs) == len(one_position_id_list)

            ##########################################
            # randomly select self.prediction_fraction
            # fraction of the tokens for masked language modelling
            # transfer these tokens to the labels as targets for training.
            indices = np.arange(0, len(encoded_inputs))
            random_probs = np.random.uniform(0, 1, size=len(indices))
            mlm_indices = indices[random_probs < self.prediction_fraction]
            labels[i, mlm_indices] = encoded_inputs[
                mlm_indices
            ]  # label the masked tokens with the inputs

            # select self.masking_fraction of the mlm tokens to be replaced by the mask token
            random_probs = np.random.uniform(0, 1, size=len(mlm_indices))
            masking_choices = random_probs <= self.masking_fraction
            masking_indices = mlm_indices[masking_choices]

            # select self.random_replacement_fraction
            # of the tokens to be replaced by the random words
            random_replacement_choices = np.logical_not(masking_choices) & (
                random_probs
                < (self.masking_fraction + self.random_replacement_fraction)
            )  # replace these indices with random words
            random_replacement_indices = mlm_indices[random_replacement_choices]

            # now that we've selected the tokens to be masked,
            # replaced or kept the same, modify the input accoringly

            # these tokens are to be replaced by the mask token
            encoded_inputs[masking_indices] = self.mask_token_id

            # replace these inputs with random other words
            encoded_inputs[random_replacement_indices] = np.random.choice(
                self.nonspecial_token_ids, size=len(random_replacement_indices)
            )

            # do nothing for those tokens that are to be kept as their original
            # pass

            # package everything back into the output arrays created before this loop:
            inputs[i, : len(encoded_inputs)] = encoded_inputs
            position_ids[i, : len(one_position_id_list)] = one_position_id_list
            attention_mask[i, : len(encoded_inputs)] = 1

        to_return = {
            "input_ids": torch.from_numpy(inputs),
            "attention_mask": torch.from_numpy(attention_mask),
            "position_ids": torch.from_numpy(position_ids),
            "labels": torch.from_numpy(labels),
        }

        if "token_type_ids" in list_of_data[0]:
            token_type_ids = torch.zeros(to_return["input_ids"].size(), dtype=torch.int)

            for i, item in enumerate(list_of_data):
                these_ids = torch.tensor(item["token_type_ids"])
                token_type_ids[i, 0 : len(these_ids)] = these_ids

            to_return["token_type_ids"] = token_type_ids

        return to_return
