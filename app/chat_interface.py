import streamlit as st
import torch
import json
import pandas as pd

def chat_interface(model, tokenizer, num_labels, label_encoders):
    st.title("LM-Action")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Recreate the assistant's response
                if isinstance(message["content"], str):
                    try:
                        response_content = json.loads(message["content"])
                    except json.JSONDecodeError:
                        st.markdown(message["content"])
                        continue
                else:
                    response_content = message["content"]

                st.markdown(response_content['user_friendly_explanation'])
                display_predictions(response_content['predictions'])
                st.markdown("---")  # Separator
                if response_content['message_type'] == 'error':
                    st.error(f"**Decision:** {response_content['custom_message']}")
                elif response_content['message_type'] == 'warning':
                    st.warning(f"**Decision:** {response_content['custom_message']}")
                elif response_content['message_type'] == 'success':
                    st.success(f"**Decision:** {response_content['custom_message']}")
                else:
                    st.info(f"**Decision:** {response_content['custom_message']}")
            else:
                st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Enter a task description:"):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)

        with torch.no_grad():
            bert_output = model.bert(**inputs).pooler_output
            logits = {name: classifier(bert_output) for name, classifier in model.classifiers.items()}

        task_labels = {
            'r_l': 'Risk level of the task',
            'f': 'Feasibility of the task',
            'l': 'Legality of the task',
            'e': 'Ethicality of the task',
            'rv': 'Reversibility of the task',
            'li': 'Limitation of the task'
        }

        predictions = {}
        for key, logit in logits.items():
            _, preds = torch.max(logit, dim=1)
            pred_index = preds.item()
            label_order = label_encoders[key]
            try:
                predicted_label = next(label for label, index in label_order.items() if index == pred_index)
            except StopIteration:
                predicted_label = list(label_order.keys())[0]
            
            task_label = task_labels.get(key, key)
            predictions[task_label] = predicted_label

        risk_level = predictions.get('Risk level of the task')
        legality = predictions.get('Legality of the task')
        ethicality = predictions.get('Ethicality of the task')

        if risk_level == 'High' and legality == 'No' and ethicality == 'No':
            message = "This task is not legal and it's risky to do."
            message_type = 'error'
        elif risk_level == 'Low' and legality == 'No':
            message = "It won't proceed because it's not legal."
            message_type = 'error'
        elif risk_level == 'Low' and legality == 'Yes' and ethicality == 'Yes':
            message = "Task is perfectly fine to execute."
            message_type = 'success'
        elif risk_level == 'High' and legality == 'Yes' and ethicality == 'Yes':
            message = "The task is risky to do; please consider the consequences."
            message_type = 'warning'
        else:
            message = "Please review the predicted labels for a detailed analysis."
            message_type = 'info'

        response_content = {
            "predictions": predictions,
            "user_friendly_explanation": "Here are the predicted task labels based on your input:",
            "custom_message": message,
            "message_type": message_type
        }

        with st.chat_message("assistant"):
            st.markdown(response_content['user_friendly_explanation'])
            display_predictions(response_content['predictions'])
            st.markdown("---")  # Separator
            if response_content['message_type'] == 'error':
                st.error(f"**Decision:** {response_content['custom_message']}")
            elif response_content['message_type'] == 'warning':
                st.warning(f"**Decision:** {response_content['custom_message']}")
            elif response_content['message_type'] == 'success':
                st.success(f"**Decision:** {response_content['custom_message']}")
            else:
                st.info(f"**Decision:** {response_content['custom_message']}")

        st.session_state.messages.append({"role": "assistant", "content": response_content})

        return logits

    return None

def get_color(label, value):
    # Function to map label and value to a specific color
    if label == 'Risk level of the task':
        if value == 'High':
            return '#FF4B4B'  # Red
        elif value == 'Medium':
            return '#FF914D'  # Dark Orange
        elif value == 'Low':
            return '#4CAF50'  # Green
    elif label in ['Feasibility of the task', 'Legality of the task', 'Ethicality of the task', 'Reversibility of the task']:
        if value == 'Yes':
            return '#4CAF50'  # Green
        elif value == 'No':
            return '#FF4B4B'  # Red
    elif label == 'Limitation of the task':
        if value == 'Yes':
            return '#FF4B4B'  # Red
        elif value == 'No':
            return '#4CAF50'  # Green
    # Default color
    return '#D3D3D3'  # Light Gray

def display_predictions(predictions):
    # Function to display predictions in a more compact layout using columns
    labels = list(predictions.keys())
    values = list(predictions.values())

    # Calculate the number of columns per row
    cols_per_row = 3  # Increased to 3 columns per row for a more compact layout
    total_labels = len(labels)
    rows = (total_labels + cols_per_row - 1) // cols_per_row  # Ceiling division

    idx = 0
    for _ in range(rows):
        cols = st.columns(cols_per_row, gap="small")
        for col in cols:
            if idx < total_labels:
                label = labels[idx]
                value = values[idx]
                color = get_color(label, value)
                text_color = 'white' if color != '#FF914D' else 'black'  # Ensure text is readable
                with col:
                    # Create a styled box using HTML with reduced padding and font size
                    html_content = f'''
                    <div style="
                        background-color: {color};
                        padding: 10px;
                        border-radius: 5px;
                        text-align: center;
                        color: {text_color};
                        font-weight: bold;
                        margin-bottom: 5px;
                    ">
                        <h5 style="margin: 0 0 5px 0; font-size: 14px;">{label}</h5>
                        <p style="font-size: 18px; margin: 0;">{value}</p>
                    </div>
                    '''
                    st.markdown(html_content, unsafe_allow_html=True)
                idx += 1
            else:
                break
