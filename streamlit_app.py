import streamlit as st
import os
import torch
from transformers import BertTokenizer, BertModel
from app.chat_interface import chat_interface

# Set Streamlit page config (this must be the first Streamlit command)
st.set_page_config(page_title="LM-Action", page_icon="🤖", layout="wide")

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
                name: torch.nn.Linear(768, labels) for name, labels in num_labels.items()
            })

        def forward(self, input_ids, attention_mask, token_type_ids=None):
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            pooled_output = outputs.pooler_output
            return {name: classifier(pooled_output) for name, classifier in self.classifiers.items()}

    # Calculate num_labels from the state_dict structure
    num_labels = {}
    for key, value in state_dict.items():
        if key.startswith('classifiers.'):
            parts = key.split('.')
            if len(parts) == 3 and parts[2] == 'weight':
                classifier_name = parts[1]
                num_labels[classifier_name] = value.shape[0]

    print("Calculated num_labels:", num_labels)  # Debug print

    # Create the model instance
    model = CustomBertForClassification(num_labels)

    # Manually load the state dict to match the keys
    model_dict = model.state_dict()
    for k, v in state_dict.items():
        if k in model_dict:
            model_dict[k] = v
    
    model.load_state_dict(model_dict)
    model.eval()

    print("Model structure:", model)  # Debug print

    # Load the tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    return model, tokenizer, num_labels

# Load the preloaded model and tokenizer
model, tokenizer, num_labels = load_model()

# Define the label encoders with the correct format
label_encoders = {
    'r_l': {'Low': 2, 'Medium': 1, 'High': 0},
    'f': {'No': 0, 'Yes': 1},
    'l': {'No': 0, 'Yes': 1},
    'e': {'No': 0, 'Yes': 1},
    'rv': {'No': 0, 'Yes': 1},
    'li': {'No': 0, 'Yes': 1}
}

# Ensure label_encoders keys match num_labels keys and have the correct format
label_encoders = {key: label_encoders.get(key, {'Unknown': 0, 'Known': 1}) for key in num_labels.keys()}

print("Label encoders:", label_encoders)  # Debug print

# Add the sidebar
with st.sidebar:
    st.header("About")
    st.markdown("LM-Action Demo, a tool that helps classify tasks based on various features.")

    if st.button("Clear Chat History", key="clear_history"):
        st.session_state.messages = []
        st.rerun()

    # Add flexible space
    st.markdown('<div style="flex-grow: 1;"></div>', unsafe_allow_html=True)

    # Add a note at the bottom of the sidebar
    st.markdown(
        "<div style='margin-top: auto; padding-top: 20px; border-top: 1px solid rgba(250, 250, 250, 0.2);'>"
        "<p style='color: rgba(250, 250, 250, 0.6); font-size: 0.8em; margin: 0; text-align: center;'>"
        "<strong>Note:</strong> This app is not complete and may produce biased or incorrect predictions."
        "</p>"
        "</div>",
        unsafe_allow_html=True
    )

# Pass the model, tokenizer, num_labels, and label_encoders to the chat_interface function
chat_interface(model, tokenizer, num_labels, label_encoders)
