import pandas as pd #Use Pandas to load and clean your data.
import numpy as np #
import streamlit as st
import plotly.express as px
import pickle
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Load your datasets
#Final Data
df = pd.read_csv(r"D:\Guvi Projects\BRCS2\Datae.csv")

#For Data Visualisation
df1 = pd.read_csv(r"D:\Guvi Projects\BRCS2\Data.csv")
df2 = pd.read_csv(r"D:\Guvi Projects\BRCS2\Data1.csv")

# Function to safely convert to sqrt
def safe_sqrt(value):
    try:
        return np.sqrt(float(value))  # Convert to float and take sqrt
    except (ValueError, TypeError):
        return np.nan  
# Define occupation types in alphabetical order with corresponding numeric codes
occupation_types = {
    0: 'Accountants',
    1: 'Cleaning staff',
    2: 'Cooking staff',
    3: 'Core staff',
    4: 'Drivers',
    5: 'HR staff',
    6: 'High skill tech staff',
    7: 'IT staff',
    8: 'Laborers',
    9: 'Low-skill Laborers',
    10: 'Managers',
    11: 'Medicine staff',
    12: 'Private service staff',
    13: 'Realty agents',
    14: 'Sales staff',
    15: 'Secretaries',
    16: 'Security staff',
    17: 'Waiters/barmen staff',
}


# Mapping for NAME_EDUCATION_TYPE
education_type_mapping = {'Secondary / secondary special': 0, 'Higher education': 1, 'Incomplete higher': 2, 'Lower secondary': 3, 'Academic degree': 4}

gender_mapping = {'F': 0, 'M': 1, 'XNA': 2}
own_car_mapping = {'N': 0, 'Y': 1,}
# Mapping for NAME_FAMILY_STATUS
family_status_mapping = {'Single / not married': 3, 'Married': 1, 'Civil marriage': 0, 'Widow': 4, 'Separated': 2}


# Download VADER lexicon if not already done
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# Function to analyze sentiment
def analyze_sentiment(text):
    sentiment_scores = sia.polarity_scores(text)
    compound = sentiment_scores['compound']
    
    if compound >= 0.05:
        return "Positive", sentiment_scores
    elif compound <= -0.05:
        return "Negative", sentiment_scores
    else:
        return "Neutral", sentiment_scores

# Main Streamlit code
# -------------------------------------------------- Logo & details on top

st.markdown("# :orange[*Bank*] *Risk* :orange[*Controller*] *System*")
st.markdown("""
<hr style="border: none; height: 5px; background-color: #FFFFFF;" />
""", unsafe_allow_html=True)

# Define tab options
tabs = ["Home", "Data Showcase", "ML Prediction", "ML Sentiment Analysis", "Data Visualization", "About"]
selected_tab = st.selectbox("Select a tab", tabs)

# Home tab content
if selected_tab == "Home":
    st.markdown("### :orange[*OVERVIEW* ]")
    st.markdown("### *The expected outcome of this project is a robust predictive model that can accurately identify customers who are likely to default on their loans. This will enable the financial institution to proactively manage their credit portfolio, implement targeted interventions, and ultimately reduce the risk of loan defaults.*")
    st.markdown("### :orange[*DOMAIN* ] ")
    st.markdown(" ### *Banking* ")
    st.markdown("""
                ### :orange[*TECHNOLOGIES USED*]     
                ### *PYTHON*
                ### *DATA PREPROCESSING*
                ### *EDA*
                ### *PANDAS*
                ### *NUMPY*
                ### *VISUALIZATION*
                ### *MACHINE LEARNING*
                ### *STREAMLIT GUI*
                """)

# Data Showcase tab content
elif selected_tab == "Data Showcase":
    st.header("Data Used")
    st.dataframe(df)

    st.header("Model Performance")
    data = {
        "Algorithm": ["Decision Tree","Random Forest","KNN","XGradientBoost"],
        "Accuracy": [91,91,98,95],
        "Precision": [88,87,96,93],
        "Recall": [92,95,99,97],
        "F1 Score": [92,91,98,95]
    }
    dff = pd.DataFrame(data)
    st.dataframe(dff)
    st.markdown(f"## The Selected Algorithm is :orange[*KNN*] and its Accuracy is   :orange[*98%*]")


elif selected_tab == "ML Prediction":
    st.markdown(f'## :violet[*Predicting Customers Default on Loans*]')
    st.write('<h5 style="color:#FBCEB1;"><i>NOTE: Min & Max given for reference, you can enter any value</i></h5>', unsafe_allow_html=True)

    with st.form("my_form"):
        col1, col2 = st.columns([5, 5])
        
        with col1:
            TOTAL_INCOME = st.text_input("TOTAL INCOME (Min: 25650.0 & Max: 117000000.0)", key='TOTAL_INCOME')
            AMOUNT_CREDIT = st.text_input("CREDIT AMOUNT (Min: 45000.0 & Max: 4050000.0)", key='AMOUNT_CREDIT')
            AMOUNT_ANNUITY = st.text_input("ANNUITY AMOUNT (Min: 1980.0 & Max: 225000.0)", key='AMOUNT_ANNUITY')
            OCCUPATION_TYPE_CODE = st.selectbox("OCCUPATION TYPE (0 to 17)", sorted(occupation_types.items()), format_func=lambda x: x[1], key='OCCUPATION_TYPE_CODE')[0]
            GENDER = st.selectbox("GENDER", list(gender_mapping.keys()), key='GENDER')
        with col2:
            OWN_CAR = st.selectbox("OWN CAR", list(own_car_mapping.keys()), key='OWN_CAR')
            EDUCATION_TYPE = st.selectbox("EDUCATION TYPE", list(education_type_mapping.keys()), key='EDUCATION_TYPE')
            FAMILY_STATUS = st.selectbox("FAMILY STATUS", list(family_status_mapping.keys()), key='FAMILY_STATUS')
            OBS_30_COUNT = st.text_input("OBS_30 COUNT (Min: 0 & Max: 348.0)", key='OBS_30_COUNT')
            DEF_30_COUNT = st.text_input("DEF_30 COUNT (Min: 0 & Max: 34.0)", key='DEF_30_COUNT')

        submit_button = st.form_submit_button(label="PREDICT STATUS")

    st.markdown("""
        <style>
        div.stButton > button:first-child {
            background-color: #FBCEB1;
            color: purple;
            width: 50%;
            display: block;
            margin: auto;
        }
        </style>
    """, unsafe_allow_html=True)

    # Validate input
    flag = 0 
    pattern = r"^(?:\d+|\d*\.\d+)$"

    for i in [TOTAL_INCOME, AMOUNT_CREDIT, AMOUNT_ANNUITY, OBS_30_COUNT, DEF_30_COUNT]:             
        if re.match(pattern, i):
            pass
        else:                    
            flag = 1  
            break

    if submit_button and flag == 1:
        if len(i) == 0:
            st.write("Please enter a valid number, space not allowed")
        else:
            st.write("You have entered an invalid value: ", i)  

    if submit_button and flag == 0:
        try:
            # Convert inputs to appropriate numeric types
            total_income = float(TOTAL_INCOME)
            amount_credit = float(AMOUNT_CREDIT)
            amount_annuity = float(AMOUNT_ANNUITY)
            occupation_type_code = int(OCCUPATION_TYPE_CODE)
            gender_code = gender_mapping[GENDER]
            own_car_code = own_car_mapping[OWN_CAR]
            education_type_code = education_type_mapping[EDUCATION_TYPE]
            family_status_code = family_status_mapping[FAMILY_STATUS]
            obs_30_count = float(OBS_30_COUNT)
            def_30_count = float(DEF_30_COUNT)

            # Construct sample array for prediction
            sample = np.array([
                [
                    safe_sqrt(total_income),
                    safe_sqrt(amount_credit),
                    safe_sqrt(amount_annuity),
                    occupation_type_code,
                    gender_code,
                    own_car_code,
                    education_type_code,
                    family_status_code,
                    safe_sqrt(obs_30_count),
                    safe_sqrt(def_30_count)
                ]
            ])

            # Load the model
            with open(r"knnmodel.pkl", 'rb') as file:
                knn = pickle.load(file)

            # Perform prediction
            pred = knn.predict(sample)

            # Display prediction result
            if pred == 0:
                st.markdown(f' ## :grey[The status is :] :green[Repay]')
            else:
                st.write(f' ## :grey[The status is ] :red[Won\'t Repay]')

        except ValueError as e:
            st.error(f"Error processing inputs: {e}")

# ML Sentiment Analysis tab content
elif selected_tab == "ML Sentiment Analysis":
    st.markdown("### :red[ML Sentiment Analysis]")
    st.write("")
    st.write("")

    # Initialize the sentiment analyzer
    nltk.download('vader_lexicon') #VADER (Valence Aware Dictionary and sEntiment Reasoner)
    sia = SentimentIntensityAnalyzer()

    # Create a function to analyze the sentiment
    def analyze_sentiment(text):
        sentiment = sia.polarity_scores(text)
        if sentiment['compound'] >= 0.05:
            return "Positive"
        elif sentiment['compound'] <= -0.05:
            return "Negative"
        else:
            return "Neutral"

    # Create a Streamlit app
    st.title("Sentiment Analysis App")

    # Get the user input
    text = st.text_input("Enter some text:")

    # Check if text is not empty
    if text:
        # Analyze the sentiment
        sentiment = analyze_sentiment(text)

        # Display the sentiment as a word
        st.write("Sentiment:", sentiment)

        # Get the sentiment scores
        sentiment = sia.polarity_scores(text)

        # Display the bar chart
        st.bar_chart({'Positive': sentiment['pos'], 'Negative': sentiment['neg'], 'Neutral': sentiment['neu']})


# Data Visualization tab content
elif selected_tab == "Data Visualization":
    st.subheader("Insights of Bank Risk Controller System")

    #--------------------------------------------------------------4

    # Bar Plot: Top 10 Occupation Types
    occupation_counts = df1['OCCUPATION_TYPE'].value_counts().reset_index()
    occupation_counts.columns = ['OCCUPATION_TYPE', 'COUNT']

    # Create a bar chart
    fig = px.bar(occupation_counts, y='OCCUPATION_TYPE', x='COUNT', color="COUNT", title='Occupation Type Counts', color_continuous_scale='PiYG')
    st.plotly_chart(fig, use_container_width=True)

    #--------------------------------------------------------------5

    INCOME_counts = df1['NAME_INCOME_TYPE'].value_counts().reset_index()
    INCOME_counts.columns = ['NAME_INCOME_TYPE', 'COUNT']

    # Create a line chart
    fig = px.line(INCOME_counts, x='NAME_INCOME_TYPE', y='COUNT', title='Income Type Counts')
    st.plotly_chart(fig, use_container_width=True)

    #--------------------------------------------------------------6

    family = df1['NAME_FAMILY_STATUS'].value_counts().reset_index()
    family.columns = ['NAME_FAMILY_STATUS', 'COUNT']

    # Create a pie chart
    fig = px.pie(family, names='NAME_FAMILY_STATUS', values='COUNT', title='Family Status Distribution')
    st.plotly_chart(fig, use_container_width=True)

    #--------------------------------------------------------------7

    EDUCATION_counts = df1['NAME_EDUCATION_TYPE'].value_counts().reset_index()
    EDUCATION_counts.columns = ['NAME_EDUCATION_TYPE', 'COUNT']

    # Create a bar chart
    fig = px.bar(EDUCATION_counts, x='NAME_EDUCATION_TYPE', y='COUNT', color='COUNT',
                 color_continuous_scale='Viridis', title='Occupation Type Counts')
    fig.update_layout(legend_title_text='Education Type')
    st.plotly_chart(fig, use_container_width=True)

    #--------------------------------------------------------------3

    fig2 = px.pie(df1, names='NAME_CONTRACT_TYPE_x', title='Distribution of Contract Types')
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig2, use_container_width=True)

    #--------------------------------------------------------------2

    dff = df2[['AMT_INCOME_TOTAL_sqrt',
               'AMT_CREDIT_x_sqrt', 'AMT_ANNUITY_x_sqrt',
               'OCCUPATION_TYPE_sqrt', 'NAME_EDUCATION_TYPE_sqrt',
               'OBS_30_CNT_SOCIAL_CIRCLE_sqrt', "TARGET"]]

    # Calculate the correlation matrix
    corr = dff.corr().round(2)

    # Plot the heatmap using Plotly Express
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="RdBu",
                    title="Correlation Matrix Heatmap")
    st.plotly_chart(fig, use_container_width=True)

    #--------------------------------------------------------------8

    fig1 = px.histogram(df1, x='OCCUPATION_TYPE', color='TARGET', barmode='group')
    fig1.update_layout(title='Countplot of TARGET by OCCUPATION_TYPE', xaxis_title='OCCUPATION_TYPE', yaxis_title='Count')
    st.plotly_chart(fig1, use_container_width=True)

# About tab content
elif selected_tab == "About":
    st.markdown("""
        ## About Bank Risk Controller System
        This application is developed as part of the Bank Risk Controller System project. It aims to provide a predictive model for identifying customers likely to default on their loans, leveraging machine learning and data analysis techniques.
        For more information, contact us at [abdulsalam47492@gmail.com].
    """)

