import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords (run once)
nltk.download('stopwords')

# Load dataset
data = pd.read_csv('../dataset/sms_spam.csv')

# Get English stopwords
stop_words = set(stopwords.words('english'))

# Text cleaning function
def clean_text(text):
    text = text.lower()                      # Convert to lowercase
    text = re.sub(r'\d+', '', text)          # Remove numbers
    text = re.sub(r'[^\w\s]', '', text)      # Remove punctuation
    words = text.split()                     # Tokenization
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Apply preprocessing
data['cleaned_text'] = data['text'].apply(clean_text)

# Save cleaned dataset
data.to_csv('../output/cleaned_sms_spam.csv', index=False)

print("Text preprocessing completed successfully.")
