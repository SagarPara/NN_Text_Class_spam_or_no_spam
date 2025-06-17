import torch
from transformers import BertTokenizerFast, AutoModel
import torch.nn as nn
import numpy as np

print("Defining BERT_Arch function")

# Custom model class (must be redefined exactly as before)
class BERT_Arch(nn.Module):
    def __init__(self, bert):
        super(BERT_Arch, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        output = self.bert(sent_id, attention_mask=mask)
        cls_hs = output.last_hidden_state[:, 0]
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

print("Defining tokenizer function")

try:
    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("sentiment_tokenizer")

    # Load pretrained BERT model
    bert = AutoModel.from_pretrained("bert-base-uncased")

    # Initialize and load sentiment model
    model = BERT_Arch(bert)
    model.load_state_dict(torch.load("sentiment_model.pt", map_location=torch.device('cpu')))
    model.eval()
    print("Model and Tokenizer loade successfully")

except Exception as e:
    print(f"Error loading Model or Tokenizer: {e}")
    raise

print("Successfully load predict function")


# Prediction function
def predict(text_list):
    tokens = tokenizer.batch_encode_plus(
        text_list,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(tokens['input_ids'], tokens['attention_mask'])
        probs = torch.nn.functional.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

    return preds.numpy(), probs.numpy()


if __name__ == "__main__":
    # Example usage
    texts = ["This is great!", "Terrible product. Don't buy."]
    labels, confidences = predict(texts)

    for text, label, prob in zip(texts, labels, confidences):
        sentiment = "Positive" if label == 1 else "Negative"
        print(f"\nText: {text}\nSentiment: {sentiment} (Confidence: {prob[label]:.2f})")
