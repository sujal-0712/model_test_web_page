import streamlit as st

st.title("⚡ Multi-Domain Artificial Intelligence Workspace")
st.write("Welcome! This hub showcases four interactive production-ready AI applications built using classical Machine Learning and deep Convolutional Neural Networks.")

st.markdown("---")

# Create two distinct structural columns to display your capabilities
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Predictive Machine Learning")
    st.markdown("""
    * **Real Estate Valuation Engine:** Predicts structural housing prices utilizing localized economic metrics and geospatial data vectors.
    * **Clinical Risk Analytics:** Analyzes electronic health records (EHR) to predict 30-day hospital readmission probabilities for diabetic patients, featuring imbalanced data balancing solutions.
    """)

with col2:
    st.subheader("🔬 Advanced Deep Learning (Computer Vision)")
    st.markdown("""
    * **Autonomous Sign Vision:** A PyTorch CNN capable of sorting 43 distinct classes of roadside traffic indicators for self-driving vehicles.
    * **Biomedical Malaria Diagnostic:** A 3-layer texture-recognition CNN designed to automatically detect *Plasmodium* parasites in thin blood smear micro-graphs.
    """)

st.info("👈 Use the navigation sidebar to seamlessly jump into any live interactive sandbox demo!")