import streamlit as st
import numpy as np
import plotly.express as px
import ast
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from testClassifiers import *

"""
Build a streamlit app that takes in a prompt (str), model (str), and labeled data (csv)
Uses the prompt and model to label the data set with a GPT classifier, then returns evalution metrics for the prompt
"""
st.title("IH/IA Classifier Evaluation")

col1, col2 = st.columns(2)
OPTIONS_IH = ['Acknowledges Personal Beliefs', 'Engages Respectfully with Diverse Perspectives',
             'Recognizes limitations in ones own knowledge or beliefs', 'Reconsiders beliefs when presented with new evidence',
             'Seeks out new information', 'Displays Empathy']
OPTIONS_IA = ['Polarizing or Tribalistic Language', 'Condescending Attitude', 'Ad Hominem', 'Displays Prejudice',
    'Close-minded Absolutism','Overinflated Expertise']
LLM_CHOICES = ['claude-3-5-haiku-latest','gpt-4.1', 'gpt-4o-mini', 'o4-mini', 'gpt-4.1-mini', 'gpt-4.1-nano', "gpt-3.5-turbo", "gpt-4", 'gpt-4o','gpt-4-turbo','gpt-3.5-turbo-16k', 'claude-sonnet-4-20250514']

with col1:
    myPrompt = st.text_area("Enter Your Prompt", height = 750)
    
    
with col2:
    csvFile = st.file_uploader("Add CSV of labeled data set to compare against")
    labelType = st.radio("Select Label Scheme", ['coarse', 'sub'])
    if labelType == 'sub':
            ihLabels = st.multiselect("Select IH Labels:", OPTIONS_IH)
            iaLabels = st.multiselect("Select IA Labels:", OPTIONS_IA)
    numDataPoints = st.number_input("How many lines of the CSV to include", 0, 350, 1)
    myModel = st.selectbox("Choose LLM", LLM_CHOICES)
    numIterations = st.number_input("How many times would you like to test this?", 0, 30, 1)
    saveOption = None
    saveOption = st.checkbox("Save labeled output?")
    myContext = None
    myContext = st.checkbox("Provide Context?")

run = st.button("Go")

if labelType == 'sub':
    mySubLabels = list(set(ihLabels) | (set(iaLabels)))
else:
    mySubLabels = ['IH', 'IA', 'Neutral']


# display the name when the submit button is clicked
# .title() is used to get the input text string
if run and csvFile and myPrompt and numIterations and numDataPoints:
    # Save the prompt!
    file = open('results/prompt.txt', 'w')
    file.write(myPrompt)
    file.close()
    if numDataPoints > 348:
        df = prepareTestData(csvFile, labelType).sample(frac=1)
    else: 
        df = prepareTestData(csvFile, labelType).sample(numDataPoints) # Start with smaller dataframe just to test
    #baselineF1 = get_baselinef1_kmost(df['classification'])
    
    #st.markdown('## Baseline F1: ')
    #st.write(baselineF1)
    IHClassifier = myClassifier(myModel, myPrompt, labelType, mySubLabels)
    f1s, f1sWeighted, coarsef1s, coarsef1sweighted = evaluate(IHClassifier, df, myContext, numIterations, saveOption)
    
    st.markdown(f'### Average F1 {labelType}: ')
    st.write(round(np.mean(f1sWeighted), 2))

    # Put F1 scores in usable format and print
    f1Df = pd.DataFrame(f1s)
    f1Averages = f1Df.mean()
    st.markdown(f"### Average F1 Scores {labelType}")
    st.write(f1Averages)

    f1Df["Step"] = f1Df.index
    # Melt for line plot (optional for better formatting)
    df_melted = f1Df.melt(id_vars="Step", var_name="Label", value_name="Score")
    # Plot using Plotly
    fig = px.line(df_melted, x="Step", y="Score", color="Label", markers=True, title="F1 Scores per Iteration")
    st.plotly_chart(fig)
    
    if labelType == 'sub':
        st.markdown('### Average F1 Coarse Roll-Up: ')
        st.write(round(np.mean(coarsef1sweighted), 2))

        # Put F1 scores in usable format and print
        f1DfCoarse = pd.DataFrame(coarsef1s)
        f1AveragesCoarse = f1DfCoarse.mean()
        st.markdown("### Average F1 Scores")
        st.write(f1AveragesCoarse)

        f1DfCoarse["Step"] = f1DfCoarse.index
        # Melt for line plot (optional for better formatting)
        df_meltedCoarse = f1DfCoarse.melt(id_vars="Step", var_name="Label", value_name="Score")
        # Plot using Plotly
        fig = px.line(df_meltedCoarse, x="Step", y="Score", color="Label", markers=True, title="F1 Scores per Iteration")
        st.plotly_chart(fig)

        labeled_df = pd.read_csv(f"results/{myModel}-0.csv")
        labeled_df['0'] = labeled_df['0'].apply(ast.literal_eval)
        labeled_df['classification'] = labeled_df['classification'].apply(ast.literal_eval)

        # Binarize
        y_true = list(labeled_df['classification'])
        y_pred = list(labeled_df['0'])
        mlb = MultiLabelBinarizer(classes=mySubLabels)
        mlb.fit(mySubLabels)
        y_true_bin = mlb.fit_transform(y_true)
        y_pred_bin = mlb.transform(y_pred)
        labels = mlb.classes_[0]

        # Compute confusion matrices
        

        # Plot each confusion matrix
        st.title("Confusion Matrix")
        cm = confusion_matrix(y_true_bin, y_pred_bin)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, cmap="Blues", fmt='g', xticklabels=["Not " + labels, labels], yticklabels=["Not " + labels, labels])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix for label: {labels}')
        st.pyplot(fig)



else:
    st.write("Please enter all the information above!")