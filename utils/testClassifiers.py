import openai
import anthropic
from pydantic import BaseModel, ValidationError, field_validator, ValidationInfo
from typing import List
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm
import swifter
from typing import List, Union
from key import *
import numpy as np



ANTHROPIC_MODELS = ['claude-sonnet-4-20250514',  'claude-3-5-haiku-latest']
OPEN_AI_MODELS = ['gpt-4.1', 'gpt-4o-mini', 'o4-mini', 'gpt-4.1-mini', 'gpt-4.1-nano', "gpt-3.5-turbo", "gpt-4", 'gpt-4o','gpt-4-turbo','gpt-3.5-turbo-16k',]

def prepareTestData(filePath, labelScheme):
    """
    filepath(str): file path where data lives
    labelScheme (str: coarse or sub) if you want to label IH/IA/Neutral (coarse) or the individual sublabels
    return dataframe of only two columns: comment_to_label, classification 
    """
    def parse_labels(s: str):
        if pd.isna(s):
            return ["None"]
        parts = s.split(",")
        labels = [p.strip() for p in parts if p.strip()]
        if not labels:
            return ["None"]
        return labels
    
    raw_df = pd.read_csv(filePath)
    raw_df['comment_to_label'] = raw_df.apply(lambda row: row['comment_1'] if row['focal_post'] == 'comment_1' else row['comment_2'], axis=1)
    #raw_df['context'] = raw_df.apply(lambda row: row['comment_1'] if row['focal_post'] == 'comment_2' else row['submission_text'], axis=1)
    if labelScheme == 'coarse':
        raw_df['classification'] = raw_df['classification'].apply(lambda x: [x])
    if labelScheme == 'sub':
        raw_df = raw_df[['comment_to_label', 'training_label']]
        raw_df['classification'] = raw_df['training_label'].apply(parse_labels, 1)
      
    df = raw_df[['comment_to_label', 'classification']]
    # Remove bot comments from dataframe
    df = df[~df['comment_to_label'].str.contains(r'\*I am a bot', na=False)]
    return df
    
def classify_labels(row_labels):
    """
    Classify a row based on counts of IH and IA labels.

    Args:
        row (str): The multi-label string (e.g., "Acknowledges Personal Beliefs, Ad Hominem").

    Returns:
        str: Classification result ("IH", "IA", "Neutral").
    """
    # All possibel IH/IA labels
    IH_LABELS = ['Acknowledges Personal Beliefs', 'Engages Respectfully with Diverse Perspectives',
             'Recognizes limitations in ones own knowledge or beliefs',
             'Seeks out new information']
    IA_LABELS = ['Polarizing or Tribalistic Language', 'Condescending Attitude',
    'Close-minded Absolutism']

    # Split the row into individual labels (assumes comma-separated labels)
    ih_count, ia_count = 0,0
    for label in row_labels:
            if label in IH_LABELS: ih_count += 1
            if label in IA_LABELS: ia_count += 1

    if ih_count > ia_count: return ['IH']
    if ia_count > ih_count: return ["IA"]

    return ["Neutral"]


def get_f1_score(y_true, y_pred, myLabels):
    """
    y_true (List[List]): list of list of ground truth data
    y_pred (List[List]): list of list ofpredicted data
    myLabels (List[str]): all possible labels from the set
    """
    mlb = MultiLabelBinarizer(classes=myLabels)
    mlb.fit([])
    y_true_bin = mlb.transform(y_true)
    y_pred_bin = mlb.transform(y_pred)
    # mlb.classes_ is the list of string labels in fixed order
    myAvg = None
    if len(mlb.classes_) == 1:
        myAvg = 'binary'
    f1_scores = f1_score(y_true_bin, y_pred_bin, average=myAvg, zero_division=0)
    print(f1_scores)
    if myAvg != 'binary': 
        f1_dict = {label: score for label, score in zip(mlb.classes_, f1_scores)}
    else: f1_dict = {mlb.classes_[0]:f1_scores}
    print(f1_dict)
    f1_weighted = f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0)
    
    return f1_dict, f1_weighted

def evaluate(classifier, df, context=None, n=5, save=None):
    f1_scores_list, f1_weighted_scores_list, coarse_scores_list, coarse_weighted_scores_list = [], [], [],[]
    myLabels = classifier.get_labels()

    for i in range(0, n):
        if context:
            # You want to include the previous message in the thread for context
            predLabels = df.swifter.apply(lambda x: classifier.label(x.comment_to_label, x.context), 1)
        else:
            predLabels = df.swifter.apply(lambda x: classifier.label(x.comment_to_label), 1)
        
        f1_dict, myF1_weighted = get_f1_score(list(df['classification']), list(predLabels), myLabels)

        f1_scores_list.append(f1_dict)
        f1_weighted_scores_list.append(myF1_weighted)

        if classifier.get_type() == 'sub':
            """
            We want to calculate a rolled up IH/IA/Neutral score to see if the classifier is at least getting a coarse labeling right
            Go through the predicted labels and classify them (count IH vs IA labels) and return a single lable
            Everything still needs to be a list of lists to work nicely with other functions
            """
            df['coarse_truth'] = df.apply(lambda x: classify_labels(x.classification), 1)
            coarse_pred = []
            # Go through the list
            for label in predLabels:
                coarse_pred.append(classify_labels(label))
            # Just hardcode the coarse labels here for ease/since they won't change
            coarse_dict, coarse_weighted = get_f1_score(list(df['coarse_truth']), coarse_pred, ['IH', 'IA', 'Neutral'])
            coarse_scores_list.append(coarse_dict)
            coarse_weighted_scores_list.append(coarse_weighted)


        if save:
            saveDf = pd.DataFrame(predLabels)
            saveDf = df.join(saveDf)
            saveDf.to_csv(f"results/{classifier.get_version()}-{i}.csv")

    return f1_scores_list, f1_weighted_scores_list, coarse_scores_list, coarse_weighted_scores_list

class myClassifier():

    def __init__(self, myModel, myPrompt, myType, mySubLabels):
        self.version = myModel
        self.prompt = myPrompt
        self.type = myType
        self.labels = mySubLabels
        self.brand= None
        self.anthropic_or_openai()

    def anthropic_or_openai(self):
        if self.version in OPEN_AI_MODELS:
            self.client = openai.OpenAI(api_key=OPEN_API_KEY)
            self.brand = 'openai'
        elif self.version in ANTHROPIC_MODELS:
            self.client = anthropic.Anthropic(api_key=ANTH_API_KEY)
            self.brand = 'anthropic'
        
    def label(self, text, context=None, max_retries=3):
        def call_gpt(myMessage):
            if self.brand == 'openai':
                return self.client.responses.create(
                        model=self.version,
                        input=(myMessage),
                        temperature = .5
                        ).output_text
            if self.brand == 'anthropic':
                response=self.client.messages.create(
                        model=self.version,
                        max_tokens=1024,
                        messages=[
                        {"role": "user", "content": f"{myMessage}"}]
                    ).content
                for content_block in response:
                    if content_block.type == 'text': return content_block.text
            
        def validate_ouput(myOutput, allowedLabels=[]):
            if self.type == 'coarse': return GPTLabelResponseCoarse(labels=myOutput)
            if self.type == 'sub': return MultiLabelResponse.model_validate({'labels':myOutput}, context={'allowed_labels':allowedLabels})
        
        toPassLabels = self.labels.copy()
        if self.type == 'sub': toPassLabels.append('None')  # Need to append None here because GPT can explicitly output None if nothing applies, but we do not want None as a label in the F1 calcs  

        formatting_prompt = generate_format_prompt(toPassLabels)
        if context: message = f"{self.prompt}\n{formatting_prompt}\nThe text to label is:\n{text}\nThe context is\n{context}."
        else: message = f"{self.prompt}\n{formatting_prompt}\nThe text to label is:\n{text}"

        # Try and get label from GPT, try three times if format is wrong
        for _ in range(max_retries):
            try:
                response = call_gpt(message)
                parsed = validate_ouput(response, toPassLabels)
                return parsed.labels
            except ValidationError as e:
                print(e)
                print(response)
                continue
            except Exception as e:
                print(f"Unexpected error: {e}")
                continue

        return ['None']


    def get_version(self):
        return self.version
    
    def get_type(self):
        return self.type
    
    def get_labels(self):
        return self.labels

    def __str__(self):
        return f"Model={self.version}\nPrompt={self.prompt}"


class GPTLabelResponseCoarse(BaseModel):
    """
    Validate that coarse labels (IH/IA/Neutral) are formatted correctly 
    Return a list, even though it is a single label so it conforms to our list of list structure needed later
    """
    labels: Union[str, List[str]]  # Final parsed label will be a single string

    @field_validator("labels", mode="before")
    @classmethod
    def validate_and_parse_labels(cls, v: Union[str, List[str]]) -> str:
        if isinstance(v, list):
            raise ValueError("Only one label is allowed")
        if not isinstance(v, str):
            raise ValueError("Expected a string label or comma-separated string")

        label_list = [part.strip() for part in v.split(",") if part.strip()]
        
        if len(label_list) != 1:
            raise ValueError("Exactly one label is required")

        label = label_list[0]
        if label not in ['IH', 'IA', 'Neutral']:
            raise ValueError(f"Invalid label: {label}")

        return [label]


# Validate sub labels
class MultiLabelResponse(BaseModel):
    """
    Validate that sub labels (variable) are formatted correctly
    """
    labels: Union[str, List[str]]

    @field_validator("labels", mode="before")
    @classmethod
    def validate_and_parse_labels(cls, v: Union[str, List[str]], info: ValidationInfo):
        allowedLabels = info.context.get("allowed_labels", set())
        if isinstance(v, list):
            label_list = v
        elif isinstance(v, str):
            # Remove any accidental extra spaces around commas
            label_list = [part.strip() for part in v.split(",")]
        else:
            raise ValueError("Expected a comma-separated string or a list of labels")

        # Check for invalid labels
        invalid = [label for label in label_list if label not in allowedLabels]
        if invalid:
            raise ValueError(f"Invalid label(s): {invalid}")
        
        if "None" in label_list and len(label_list) > 1:
            raise ValueError("'None' must appear by itself with no other labels")

        return label_list

def generate_format_prompt(allowedLabels):
    return (
        "Respond with one of the following labels."
        "Do not add any explanations.\n\n"
        "Valid labels:\n" +
        ",".join(allowedLabels)
    )

if __name__ == "__main__":
    main()