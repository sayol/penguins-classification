import pickle
import streamlit as st
import numpy as np
import pandas as pd

st.title(body="Penguins Classification")
st.header(body="Input Features")
island = st.radio(
    label="Island",
    options=["Torgersen", "Biscoe", "Dream"],
)
sex = st.radio(
    label="Sex",
    options=["Male", "Female"],
)
bill_length = st.slider(
    label="Bill Length (mm)",
    min_value=30.0,
    max_value=60.0,
    step=0.1,
)
bill_depth = st.slider(
    label="Bill Depth (mm)",
    min_value=10.0,
    max_value=25.0,
    step=0.1,
)
flipper_length = st.slider(
    label="Flipper Length (mm)",
    min_value=170.0,
    max_value=200.0,
    step=1.0,
)
body_mass = st.slider(
    label="Body Mass (g)",
    min_value=2700.0,
    max_value=3500.0,
    step=100.0,
)

with open(file="pipeline.pkl", mode="rb") as file:
    model = pickle.load(file=file)
    X_inputs = pd.DataFrame(
        {
            "island": np.array([island], dtype=np.str_),
            "bill_length_mm": np.array([bill_length], dtype=np.float64),
            "bill_depth_mm": np.array([bill_depth], dtype=np.float64),
            "flipper_length_mm": np.array([flipper_length], dtype=np.float64),
            "body_mass_g": np.array([body_mass], dtype=np.float64),
            "sex": np.array([sex], dtype=np.str_),
        },
    )

    y_outputs = model.predict(X_inputs)
    st.write(y_outputs)
