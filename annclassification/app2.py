import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from io import BytesIO
import graphviz

# Load the trained model
model = tf.keras.models.load_model('annclassification/model.h5')

# Load the encoders and scaler
with open('annclassification/label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('annclassification/onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('annclassification/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Title and description
st.title('Customer Churn Prediction')
st.markdown("""
This app predicts the probability of customer churn using a neural network. 
Follow the flowchart below to understand the process:
""")

st.write("Implementation workflow: ")
# Create a more detailed flowchart
flowchart = graphviz.Digraph()

# User Inputs
flowchart.node('1', 'User Inputs', shape='box', style='filled', fillcolor='lightyellow')

# Data Preprocessing
flowchart.node('2', 'Data Preprocessing', shape='box', style='filled', fillcolor='lightblue')
flowchart.edge('1', '2', label='Input Data')

# Feature Scaling
flowchart.node('3', 'Feature Scaling', shape='box', style='filled', fillcolor='lightblue')
flowchart.edge('2', '3', label='Scaling with Standard Scaler')

# Model Initialization
flowchart.node('4', 'Model Initialization\n(Neural Network Setup)', shape='box', style='filled', fillcolor='lightgreen')
flowchart.edge('3', '4', label='Initialize Model')

# Forward Propagation
flowchart.node('5', 'Forward Propagation', shape='box', style='filled', fillcolor='lightgreen')
flowchart.edge('4', '5', label='Input Through Layers')

# Activation Function
flowchart.node('6', 'Activation Function\n(Sigmoid/ReLU)', shape='box', style='filled', fillcolor='lightgreen')
flowchart.edge('5', '6', label='Apply Activation Function')

# Loss Calculation
flowchart.node('7', 'Loss Calculation', shape='box', style='filled', fillcolor='lightcoral')
flowchart.edge('6', '7', label='Calculate Loss')

# Backpropagation
flowchart.node('8', 'Backpropagation', shape='box', style='filled', fillcolor='lightcoral')
flowchart.edge('7', '8', label='Adjust Weights')

# Epoch Selection
flowchart.node('9', 'Epoch Selection\nand Optimization', shape='box', style='filled', fillcolor='lightyellow')
flowchart.edge('8', '9', label='Repeat for N Epochs')

# Model Prediction
flowchart.node('10', 'Model Prediction', shape='box', style='filled', fillcolor='lightpink')
flowchart.edge('9', '10', label='Test Data Input')

# Output Prediction
flowchart.node('11', 'Output Prediction\nChurn Probability', shape='box', style='filled', fillcolor='lightpink')
flowchart.edge('10', '11', label='Prediction Result')

# Add flowchart edges to connect the steps
flowchart.edge('1', '2', label='Data Input')
flowchart.edge('2', '3', label='Preprocessing')
flowchart.edge('3', '4', label='Neural Network Setup')
flowchart.edge('4', '5', label='Forward Propagation')
flowchart.edge('5', '6', label='Apply Activation')
flowchart.edge('6', '7', label='Loss Calculation')
flowchart.edge('7', '8', label='Backpropagation')
flowchart.edge('8', '9', label='Epoch Optimization')
flowchart.edge('9', '10', label='Test Prediction')
flowchart.edge('10', '11', label='Churn Probability Output')

# Render the graph
st.graphviz_chart(flowchart)

# Custom Visualization Function
def visualize_model_architecture(model):
    fig, ax = plt.subplots(figsize=(8, 6))
    layer_names = [layer.name for layer in model.layers]
    node_counts = [layer.units if hasattr(layer, 'units') else 0 for layer in model.layers]

    # Generate positions for layers and nodes
    x_positions = np.arange(len(layer_names))
    y_positions = [np.linspace(-count / 2, count / 2, count) for count in node_counts]

    # Plot layers and connections
    for i, (x, y) in enumerate(zip(x_positions, y_positions)):
        if len(y) > 0:  # Only plot if the layer has nodes
            ax.scatter([x] * len(y), y, s=100, label=layer_names[i], zorder=2)
        if i > 0 and len(y_positions[i - 1]) > 0:
            for prev_y in y_positions[i - 1]:
                for curr_y in y:
                    ax.plot([x_positions[i - 1], x], [prev_y, curr_y], color='gray', zorder=1)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(layer_names, fontsize=10)
    ax.set_title("Neural Network Architecture", fontsize=14)
    ax.set_xlim(-0.5, len(layer_names) - 0.5)
    ax.set_ylim(-max(node_counts) / 2 - 1, max(node_counts) / 2 + 1)
    ax.set_xlabel("Layers", fontsize=12)
    ax.axis("off")
    ax.legend(fontsize=8)

    return fig

#Visualisation of the structure 
st.markdown("### Neural Network Structure")
fig = visualize_model_architecture(model)
st.pyplot(fig)


# Sidebar for input
st.sidebar.header('Input Features')
geography = st.sidebar.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.sidebar.selectbox('Gender', label_encoder_gender.classes_)
age = st.sidebar.slider('Age', 18, 92, 30)
balance = st.sidebar.number_input('Balance', value=0.0)
credit_score = st.sidebar.number_input('Credit Score', value=600)
estimated_salary = st.sidebar.number_input('Estimated Salary', value=50000)
tenure = st.sidebar.slider('Tenure', 0, 10, 5)
num_of_products = st.sidebar.slider('Number of Products', 1, 4, 2)
has_cr_card = st.sidebar.selectbox('Has Credit Card', [0, 1])
is_active_member = st.sidebar.selectbox('Is Active Member', [0, 1])

# Prepare the input data
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# One-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict churn
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display prediction
st.subheader('Prediction Results: Probability of Customer availing the service')
st.metric('Churn Probability', f'{prediction_proba:.2%}')

# Educational tooltips
st.markdown("""
### About the Inputs
- **Geography**: Customer's location.
- **Gender**: Male or Female.
- **Age**: Customer's age.
- **Balance**: Balance in the customer's account.
- **Credit Score**: A measure of the customer's creditworthiness.
- **Estimated Salary**: Customer's annual estimated salary.
- **Tenure**: Number of years the customer has been with the bank.
- **Number of Products**: Total products the customer has purchased.
- **Has Credit Card**: Whether the customer owns a credit card.
- **Is Active Member**: Whether the customer is an active member.
""")
