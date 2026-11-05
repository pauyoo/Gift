import streamlit as st
import pandas as pd
import joblib

# # T load my trained model
model = joblib.load("gift.pkl")
 ##My Page
# st.set_page_config(
#     page_title="Credit Default Prediction",
#     page_icon="💳",
#     layout="
st.title("Gift credit prediction model")
st.write("Enter customer information to predict the likelihood of credit default.")
st.divider()


# Colecting the data
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=85, value=30)
    employ = st.number_input("Years Employed", min_value=0, value=5)
    address = st.number_input("Years at Address", min_value=0, value=3)
    income = st.number_input("Annual Income", min_value=0, value=50000)

with col2:
    debtinc = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=25.0)
    creddebt = st.number_input("Credit Debt", min_value=0.0, value=5000.0)
    othdebt = st.number_input("Other Debt", min_value=0.0, value=2000.0)# # Prediction button
if st.button("Predict Default", use_container_width=True):

    # Arrange inputs in the same order as training data
    # Mapping my input fields with my x variables
    input_data = pd.DataFrame([{
        'age': age,
        'employ': employ,
        'address': address,
        'income': income,
        'debtinc': debtinc,
        'creddebt': creddebt,
        'othdebt': othdebt
    }])

# Make prediction
    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("High Risk: Customer is predicted to default.")
    else:
        st.success("Low Risk: Customer is predicted not to default.")

    # Display probability if model supports it
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(input_data)[0][1]

        st.metric(
            "Probability of Default",
            f"{probability * 100:.2f}%"
        )

        st.progress(float(probability))











# import streamlit as st
# import pandas as pd
# import joblib

# # T load my trained model
# model = joblib.load("gift.pkl")

# # My Page
# st.set_page_config(
#     page_title="Credit Default Prediction",
#     page_icon="💳",
#     layout="centered"
# )

# st.title("Credit Default Prediction System")

# st.write("Enter customer information to predict the likelihood of credit default.")

# st.divider()

# # Colecting the data
# st.subheader("Customer Information")

# col1, col2 = st.columns(2)

# with col1:
#     age = st.number_input("Age", min_value=18, max_value=100, value=30)
#     employ = st.number_input("Years Employed", min_value=0, value=5)
#     address = st.number_input("Years at Address", min_value=0, value=3)
#     income = st.number_input("Annual Income", min_value=0, value=50000)

# with col2:
#     debtinc = st.number_input("Debt-to-Income Ratio", min_value=0.0, value=25.0)
#     creddebt = st.number_input("Credit Debt", min_value=0.0, value=5000.0)
#     othdebt = st.number_input("Other Debt", min_value=0.0, value=2000.0)

# st.divider()

# # Prediction button
# if st.button("Predict Default", use_container_width=True):

#     # Arrange inputs in the same order as training data
#     # Mapping my input fields with my x variables
#     input_data = pd.DataFrame([{
#         'age': age,
#         'employ': employ,
#         'address': address,
#         'income': income,
#         'debtinc': debtinc,
#         'creddebt': creddebt,
#         'othdebt': othdebt
#     }])

#     # Make prediction
#     prediction = model.predict(input_data)[0]

#     st.subheader("Prediction Result")

#     if prediction == 1:
#         st.error("High Risk: Customer is predicted to default.")
#     else:
#         st.success("Low Risk: Customer is predicted not to default.")

#     # Display probability if model supports it
#     if hasattr(model, "predict_proba"):
#         probability = model.predict_proba(input_data)[0][1]

#         st.metric(
#             "Probability of Default",
#             f"{probability * 100:.2f}%"
#         )

#         st.progress(float(probability))
