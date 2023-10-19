import torch
from torch.nn.functional import cosine_similarity
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def compute_cosine_similarity(embeddings_1, embeddings_2):
    # Compute cosine similarity between embeddings_1 and embeddings_2
    similarities = cosine_similarity(embeddings_1, embeddings_2)

    return similarities


class SentenceDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        return self.sentences[index]


class collate_cl:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Tokenize sentences
        encoded_inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt')
        return encoded_inputs


class mpnet_embed_class():
    def __init__(self, device='cuda', nli=True):
        self.device = device

        if nli:
            model = AutoModel.from_pretrained('sentence-transformers/nli-mpnet-base-v2')
            tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/nli-mpnet-base-v2')
        else:
            model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
            tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')

        model.to(device)
        self.model = model
        self.tokenizer = tokenizer
        self.collate_fn = collate_cl(tokenizer)

    def get_mpnet_embed_batch(self, predictions, ground_truth, batch_size=10):

        dataset_1 = SentenceDataset(predictions)
        dataset_2 = SentenceDataset(ground_truth)

        dataloader_1 = DataLoader(dataset_1, batch_size=batch_size, collate_fn=self.collate_fn, num_workers=1)
        dataloader_2 = DataLoader(dataset_2, batch_size=batch_size, collate_fn=self.collate_fn, num_workers=1)

        # Compute token embeddings
        embeddings_1 = []
        embeddings_2 = []

        with torch.no_grad():
            for count, (batch_1, batch_2) in enumerate(zip(dataloader_1, dataloader_2)):
                if count % 50 == 0:
                    print(count, ' out of ', len(dataloader_2))
                batch_1 = {key: value.to(self.device) for key, value in batch_1.items()}
                batch_2 = {key: value.to(self.device) for key, value in batch_2.items()}

                model_output_1 = self.model(**batch_1)
                model_output_2 = self.model(**batch_2)

                sentence_embeddings_1 = mean_pooling(model_output_1, batch_1['attention_mask'])
                sentence_embeddings_2 = mean_pooling(model_output_2, batch_2['attention_mask'])

                embeddings_1.append(sentence_embeddings_1)
                embeddings_2.append(sentence_embeddings_2)

        # Concatenate embeddings
        embeddings_1 = torch.cat(embeddings_1)
        embeddings_2 = torch.cat(embeddings_2)

        # Normalize embeddings
        embeddings_1 = torch.nn.functional.normalize(embeddings_1, p=2, dim=1)
        embeddings_2 = torch.nn.functional.normalize(embeddings_2, p=2, dim=1)

        # Compute cosine similarity
        similarities = compute_cosine_similarity(embeddings_1, embeddings_2)

        # # Average cosine similarity
        # average_similarity = torch.mean(similarities)

        return similarities
