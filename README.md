# Before running our programs

**Install requirements:**\
`pip install -r requirements.txt`

**Download spaCy model:**\
`python -m spacy download xx_ent_wiki_sm`

# Running the dictionary baseline
This program is located in the 'dictionary_baseline' directory.\
`python3 dictionary_base.py`

# Running the svm baseline
This program is located in the 'svm' directory.\
<br>
**To use the full dataset for the svm baseline run:**\
`python svm.py --dataset full`\
<br>
*This may take a very long time so we recommend using the default dataset which uses the first 50.000 lines of the full datasets.*

# Running the roBERTa model:
To run our main model:\
`python3 main.py`
