# Install requirements
pip install -r requirements.txt

# Download spaCy model
python -m spacy download xx_ent_wiki_sm

# To use the full dataset for the svm baseline
python svm.py --dataset full
# This may take a very long time so we recommend using the default dataset which uses the first 50.000 lines of the full datasets

# To run our main model
python main.py
