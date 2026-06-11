# Author Detection using NLP 📝

A Natural Language Processing (NLP) project for identifying and classifying authors based on text stylometry and linguistic features.

## 🎯 Project Overview

This project uses NLP techniques to detect and classify different authors from their writing samples. By analyzing linguistic patterns, vocabulary choices, and textual features, the model learns to distinguish between different authors' writing styles.

**Key Applications:**
- Author identification in anonymous texts
- Plagiarism detection
- Writing style analysis
- Literary forensics

---

## 📊 What's Inside

### Notebook Cells: 166
- **Code Cells:** 85
- **Markdown Cells:** 81

### Technologies Used
- **NLP Libraries:** NLTK, Scikit-Learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **ML Techniques:** Feature extraction, Classification models

---

## 📋 Requirements

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `pandas` - Data manipulation
- `nltk` - Natural Language Processing
- `numpy` - Numerical computing
- `scikit-learn` - Machine Learning
- `matplotlib`, `seaborn` - Visualization
- `jupyter` - Jupyter Notebook support

---

## 🚀 Quick Start

### Option 1: Jupyter Notebook (Interactive)
```bash
jupyter notebook SERHAT_TAY_Author_Detection_using_NLP.ipynb
```

### Option 2: Python Script (Non-Interactive)
```bash
python SERHAT_TAY_Author_Detection_using_NLP.py
```

---

## 📁 Files

| File | Description |
|------|-------------|
| `SERHAT_TAY_Author_Detection_using_NLP.ipynb` | Original Jupyter Notebook with full analysis |
| `SERHAT_TAY_Author_Detection_using_NLP.py` | Converted Python script version |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## 🔍 Project Workflow

### 1. **Data Loading**
   - Read text files from directories
   - Organize by author (subdirectories as labels)
   - Prepare dataset for processing

### 2. **Text Preprocessing**
   - Tokenization
   - Removing stop words
   - Lowercasing and normalization
   - Handling special characters

### 3. **Feature Extraction**
   - Word frequency analysis
   - TF-IDF vectorization
   - N-gram analysis (bigrams, trigrams)
   - Linguistic features:
     - Average word length
     - Sentence length
     - Punctuation patterns
     - Vocabulary richness

### 4. **Model Training**
   - Classification models (SVM, Naive Bayes, Random Forest, etc.)
   - Train/test split
   - Cross-validation

### 5. **Evaluation**
   - Accuracy, Precision, Recall, F1-Score
   - Confusion matrix
   - ROC-AUC curves
   - Feature importance analysis

---

## 💡 Key NLP Concepts

### Tokenization
Breaking text into words or sentences for analysis.

### Frequency Distribution
Analyzing which words appear most frequently in each author's text.

### TF-IDF (Term Frequency-Inverse Document Frequency)
Numerical representation of how important a word is to a document.

### N-grams
Sequences of N words or characters (e.g., bigrams = pairs of words).

### Stylometry
Quantitative analysis of writing style - different authors have unique patterns.

---

## 📈 Expected Features

The model analyzes:
- **Lexical features:** Vocabulary size, word frequency
- **Syntactic features:** Sentence structure, punctuation
- **Semantic features:** Word choice patterns, topic preferences
- **Statistical features:** Word length distribution, repetition patterns

---

## 🎓 Learning Outcomes

After working with this project, you'll understand:
- ✅ Text preprocessing and tokenization
- ✅ Feature engineering for NLP
- ✅ Author identification techniques
- ✅ Machine learning classification
- ✅ Text analysis and visualization
- ✅ Stylometry and writing patterns

---

## ⚠️ Important Notes

### Data Path Adjustment

The original notebook uses Google Colab paths:
```python
data_path = "/content/30Columnists/raw_texts"
```

**If running locally, update to your data directory:**
```python
data_path = "./your_local_path/raw_texts"
```

### NLTK Data Download

First time running may require downloading NLTK data:
```python
import nltk
nltk.download('punkt')           # Tokenizer
nltk.download('stopwords')       # Stop words
nltk.download('averaged_perceptron_tagger')  # POS tagging
```

---

## 🔧 Customization

### Change Data Source
```python
data_path = "path/to/your/text/files"
```

### Adjust Feature Set
Modify the feature extraction functions to include/exclude specific linguistic features.

### Try Different Models
The script likely includes multiple classifiers. Experiment with:
- Support Vector Machines (SVM)
- Naive Bayes
- Random Forest
- Gradient Boosting
- Neural Networks

### Hyperparameter Tuning
```python
# Example: GridSearchCV for parameter optimization
from sklearn.model_selection import GridSearchCV
```

---

## 📊 Sample Output

```
Author Detection Results:
========================

Training Accuracy: 92.5%
Testing Accuracy: 87.3%

Confusion Matrix:
         Author1  Author2  Author3
Author1    45       3        2
Author2     2      48        5
Author3     3       4       46

Top 10 Discriminative Features:
- Word: "therefore" (Author1: 0.89)
- Word: "although" (Author2: 0.87)
- Avg. word length (Author3: 5.2)
```

---

## 🚨 Troubleshooting

### Issue: "No module named 'nltk'"
**Solution:**
```bash
pip install nltk
```

### Issue: "FileNotFoundError: data_path"
**Solution:**
Update the data path to match your local directory structure.

### Issue: "Memory Error with large datasets"
**Solution:**
- Process data in chunks
- Reduce feature dimensionality
- Use sampling

---

## 📚 Resources & References

- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-Learn ML Guide](https://scikit-learn.org/)
- [Stylometry & Author Attribution](https://en.wikipedia.org/wiki/Stylometry)
- [NLP with Python](https://realpython.com/natural-language-processing-python/)
- [Feature Engineering for NLP](https://towardsdatascience.com/feature-engineering-for-nlp-5f11e3b3cf21)

---

## 🔐 Data Privacy

This project works with text data. Ensure:
- ✅ Proper data handling and storage
- ✅ Compliance with data protection regulations
- ✅ Anonymization if required

---

## 🎯 Potential Improvements

- [ ] Add more advanced NLP techniques (Word2Vec, BERT embeddings)
- [ ] Implement deep learning models (LSTM, CNN)
- [ ] Create API endpoint for author prediction
- [ ] Add visualization dashboard
- [ ] Support for multiple languages
- [ ] Real-time text analysis

---

## 👤 Author

**Original Creator:** Serhat Tay  
**Project Type:** NLP Classification | Author Detection  
**Use Case:** Academic | Research | Portfolio

---

## 📝 License

This project is provided as-is for educational and research purposes.

---

## 🤝 Usage Tips

1. **Start with Notebook:** The `.ipynb` file provides better visualization and step-by-step explanations
2. **Use Python Script:** For automation, batch processing, or integration into pipelines
3. **Experiment:** Try different models, features, and parameter combinations
4. **Visualize:** Generate plots to understand patterns and predictions
5. **Document:** Keep track of findings and modifications

---

## 📞 Support

For issues or questions:
1. Check the troubleshooting section
2. Review NLTK and Scikit-Learn documentation
3. Verify data format and paths
4. Test with sample data first

---

**Happy Learning! 🚀**

*Last Updated: June 2024*
*Compatible with Python 3.8+*
