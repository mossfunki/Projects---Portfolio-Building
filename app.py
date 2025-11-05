import streamlit as st
import pandas as pd
import numpy as np

st.title("Economic Dashboard - Working Version")
st.success("✅ Packages loaded successfully!")

# Simple data display
data = pd.DataFrame({
    'Country': ['USA', 'China', 'Japan', 'Germany'],
    'GDP': [21.4, 14.3, 5.1, 4.2],
    'Growth': [2.3, 6.1, 1.6, 1.5]
})

st.dataframe(data)
st.write("All packages loaded correctly!")
