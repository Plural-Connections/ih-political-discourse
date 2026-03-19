# ih-political-discourse
Code and data for "Detecting and enhancing intellectual humility in online political discourse"

## process-experiment-data
Contains code to process raw comments from the RCT experiment, .py files are for formatting raw data and require a key.py to run with relevant API keys
Files should be run in order that they are named (1, 2, 3)
.r files are used to do statistical analysis and create figures for paper taking in the output of the .py files

##ihclassifier
Contains code to test different prompts for classification with a streamlit interface
To run, you will need a key.py file with relevant API keys
classifierPrompt.txt was the final prompt used for classification throughout the paper
