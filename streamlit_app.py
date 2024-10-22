import streamlit as st

# Set Streamlit page config (this must be the first Streamlit command)
st.set_page_config(page_title="LM-Action", page_icon="🤖", layout="wide")

import os
import torch
from transformers import BertTokenizer, BertModel
from app.chat_interface import chat_interface

# Define the path to your preloaded model and tokenizer
MODEL_PATH = os.path.join("app", "best_multilingual_bert_model.pth")

# Preload the model
@st.cache_resource
def load_model():
    # Load the saved state
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    
    # Print the keys in the checkpoint for debugging
    print("Checkpoint keys:", checkpoint.keys())
    
    # Check if 'model_state_dict' exists, if not, use the entire checkpoint as the state dict
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Define the custom model that matches the preloaded state_dict structure
    class CustomBertForClassification(torch.nn.Module):
        def __init__(self, num_labels):
            super().__init__()
            self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
            self.classifiers = torch.nn.ModuleDict({
                name: torch.nn.Sequential(
                    torch.nn.Linear(768, 768),
                    torch.nn.ReLU(),
                    torch.nn.Linear(768, 768),
                    torch.nn.ReLU(),
                    torch.nn.Linear(768, labels)
                ) for name, labels in num_labels.items()
            })

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            pooled_output = outputs.pooler_output
            return {name: classifier(pooled_output) for name, classifier in self.classifiers.items()}

    # Calculate num_labels from the state_dict structure
    num_labels = {}
    for key, value in state_dict.items():
        if key.startswith('classifiers.') and key.endswith('.4.weight'):
            classifier_name = key.split('.')[1]
            num_labels[classifier_name] = value.shape[0]

    # Create the model instance
    model = CustomBertForClassification(num_labels)

    # Manually load the state dict to match the keys
    model_dict = model.state_dict()
    for k, v in state_dict.items():
        if k.startswith('classifiers.'):
            parts = k.split('.')
            new_key = f"classifiers.{parts[1]}.{'.'.join(parts[2:])}"
            if new_key in model_dict:
                model_dict[new_key] = v
    
    model.load_state_dict(model_dict)
    model.eval()

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    return model, tokenizer, num_labels

# Load the preloaded model and tokenizer
model, tokenizer, num_labels = load_model()

# Define the label encoders with the correct format
label_encoders = {
    'r_l': {'Low': 0, 'Medium': 1, 'High': 2},
    'f': {'No': 0, 'Yes': 1},
    'l': {'No': 0, 'Yes': 1},
    'e': {'No': 0, 'Yes': 1},
    'rv': {'No': 0, 'Yes': 1},
    'li': {'No': 0, 'Yes': 1}
}

# Ensure label_encoders keys match num_labels keys and have the correct format
label_encoders = {key: label_encoders.get(key, {'Unknown': 0, 'Known': 1}) for key in num_labels.keys()}

# Add the sidebar
with st.sidebar:
    st.header("About")
    st.markdown("LM-Action Demo, a tool that helps classify tasks based on various features.")

    if st.button("Clear Chat History", key="clear_history"):
        st.session_state.messages = []
        st.rerun()

# Pass the model, tokenizer, num_labels, and label_encoders to the chat_interface function
chat_interface(model, tokenizer, num_labels, label_encoders)
