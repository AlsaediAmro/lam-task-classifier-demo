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
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process the input and get predictions
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        print("Tokenizer inputs:", inputs)  # Debug print

        try:
            with torch.no_grad():
                outputs = model(**inputs)
            print("Raw model outputs:", outputs)  # Debug print
        except Exception as e:
            print(f"Error during model inference: {str(e)}")
            outputs = {}

        predictions = {}
        if isinstance(outputs, dict):
            for name, logits in outputs.items():
                print(f"Processing {name}:")  # Debug print
                print(f"Logits shape: {logits.shape}")  # Debug print
                _, preds = torch.max(logits, dim=1)
                pred_index = preds.item()
                print(f"Predicted index: {pred_index}")  # Debug print
                label_order = label_encoders[name]
                print(f"Label order: {label_order}")  # Debug print
                try:
                    predicted_label = next(label for label, index in label_order.items() if index == pred_index)
                except StopIteration:
                    print(f"No matching label found for index {pred_index}")  # Debug print
                    predicted_label = "Unknown"
                predictions[name] = predicted_label
                print(f"Predicted label: {predicted_label}")  # Debug print
        else:
            print("Model output is not a dictionary")

        print("Final predictions:", predictions)  # Debug print

        # Prepare the response content
        response_content = {
            "predictions": predictions,
            "user_friendly_explanation": "Here are the predicted task labels based on your input:",
            "custom_message": get_custom_message(predictions),
            "message_type": get_message_type(predictions)
        }

        # Display assistant's response
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

        # Add assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_content})

    return None

def get_custom_message(predictions):
    risk_level = predictions.get('r_l')
    legality = predictions.get('l')
    ethicality = predictions.get('e')

    if risk_level == 'High' and legality == 'No' and ethicality == 'No':
        return "This task is not legal and it's risky to do."
    elif risk_level == 'Low' and legality == 'No':
        return "It won't proceed because it's not legal."
    elif risk_level == 'Low' and legality == 'Yes' and ethicality == 'Yes':
        return "Task is perfectly fine to execute."
    elif risk_level == 'High' and legality == 'Yes' and ethicality == 'Yes':
        return "The task is risky to do; please consider the consequences."
    else:
        return "Please review the predicted labels for a detailed analysis."

def get_message_type(predictions):
    risk_level = predictions.get('r_l')
    legality = predictions.get('l')
    ethicality = predictions.get('e')

    if risk_level == 'High' and legality == 'No' and ethicality == 'No':
        return 'error'
    elif risk_level == 'Low' and legality == 'No':
        return 'error'
    elif risk_level == 'Low' and legality == 'Yes' and ethicality == 'Yes':
        return 'success'
    elif risk_level == 'High' and legality == 'Yes' and ethicality == 'Yes':
        return 'warning'
    else:
        return 'info'

def get_color(label, value):
    # Function to map label and value to a specific color
    if label == 'r_l':
        if value == 'High':
            return '#FF4B4B'  # Red
        elif value == 'Medium':
            return '#FFA500'  # Orange
        elif value == 'Low':
            return '#4CAF50'  # Green
    elif label in ['f', 'l', 'e', 'rv']:
        if value == 'Yes':
            return '#4CAF50'  # Green
        elif value == 'No':
            return '#FF4B4B'  # Red
    elif label == 'li':
        if value == 'Yes':
            return '#FF4B4B'  # Red
        elif value == 'No':
            return '#4CAF50'  # Green
    # Default color
    return '#D3D3D3'  # Light Gray

def display_predictions(predictions):
    # Function to display predictions in a more compact layout using columns
    label_names = {
        'r_l': 'Risk level of the task',
        'f': 'Feasibility of the task',
        'l': 'Legality of the task',
        'e': 'Ethicality of the task',
        'rv': 'Reversibility of the task',
        'li': 'Limitation of the task'
    }

    # Calculate the number of columns per row
    cols_per_row = 3
    total_labels = len(predictions)
    rows = (total_labels + cols_per_row - 1) // cols_per_row  # Ceiling division

    idx = 0
    for _ in range(rows):
        cols = st.columns(cols_per_row, gap="small")
        for col in cols:
            if idx < total_labels:
                label = list(predictions.keys())[idx]
                value = predictions[label]
                color = get_color(label, value)
                text_color = 'white' if color != '#FFA500' else 'black'  # Ensure text is readable
                with col:
                    # Create a styled box using HTML with reduced padding and consistent font size
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
                        <p style="margin: 0 0 5px 0; font-size: 1em;">{label_names[label]}</p>
                        <p style="font-size: 1em; margin: 0; font-weight: bold;">{value}</p>
                    </div>
                    '''
                    st.markdown(html_content, unsafe_allow_html=True)
                idx += 1
            else:
                break
