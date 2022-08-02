# Group 8 Data Science Applications Clustering Final Project 

# Book Recommender Bot
A simple and well designed structure is essential for any machine learning project, project template that combines **simplicity, best practice for CODE structure** and **good CODE design**. 
The main idea is that there's much same stuff you do every time when you start our machine learning project, so wrapping all this shared stuff will help you to change just the core idea every time you start our machine learning project. 

**So, here’s a simple readme template that help you get into our project faster and just focus on your notice and explanations, etc.)**

In order to decrease repeated code shanks, increase the time that can read the code in, flexibility an reusability we used a functional programming structure that focused on split all problems in our project in functions and use that functions many times in many places in the code without repeating the code.
 

**Our Project consists of 5 parts :**
- Classification part notebook: sentiment analysis for user reviews  
- Clustering part notebook: collect books that is similar due to its reviews clustering 
- Recommender System part notebook: contains multiple implementations for recommender algorithms 
- dialogue flow part: we connect here with google dialogue flow to build chatbot that deals with users and show the best book. 
- Deployment part notebook: we build a local server using 'ngrok' and connect it with dialogue flow through the API. 


# Requirements
- [numpy](https://numpy.org/) (The fundamental package for scientific computing with Python)
- [pandas](https://pandas.pydata.org/) (pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.) 
- [google.colab.drive](https://colab.research.google.com/) (library to integrate google colab with google drive)
- [requests](https://pypi.org/project/requests/) (library to easily use API with python)
- [json](https://docs.python.org/3/library/json.html) (library to easily use JSON with python)
- [sklearn](https://scikit-learn.org/stable/) (Machine Learning and Data Analysis Library in Python)
- [nltk](https://www.nltk.org/) (The fundamental package for Natural Language Processing with Python)
- [matplotlib](https://matplotlib.org/) (Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python)
- [seaborn](https://seaborn.pydata.org/) (Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.)
- [gensim](https://radimrehurek.com/gensim/index.html) (library for training of vector embeddings, topic modelling, document indexing and similarity retrieval with large corpora – Python or otherwise.)
- [pyldavis](https://pyldavis.readthedocs.io/en/latest/readme.html) (pyLDAvis is designed to help users interpret the topics in a topic model that has been fit to a corpus of text data. The package extracts information from a fitted LDA topic model to inform an interactive web-based visualization.)
- [transformers](https://pypi.org/project/transformers/2.1.0/) (State-of-the-art Natural Language Processing for TensorFlow 2.0 and PyTorch

)
- [wordcloud](https://pypi.org/project/wordcloud/) (A little word cloud generator in Python)
- [spacy](https://spacy.io/) (spaCy is a free open-source library for Natural Language Processing in Python.)
- [tqdm](https://tqdm.github.io/) (Instantly make your loops show a smart progress meter - just wrap any iterable with)
- [imblearn](https://pypi.org/project/imblearn/) (Toolbox for imbalanced dataset in machine learning.)
- [gzip](https://docs.python.org/3/library/gzip.html) (This module provides a simple interface to compress and decompress files just like the GNU programs gzip and gunzip would.)
- [fastapi](https://fastapi.tiangolo.com/) (FastAPI framework, high performance, easy to learn, fast to code, ready for production)
- [pydentic](https://pypi.org/project/pydantic/) (Data validation and settings management using python type hints)
- [pyngrok](https://pypi.org/project/pyngrok/) (A Python wrapper for ngrok.)
- [unicorn](https://pypi.org/project/unicorn/) (Unicorn CPU emulator engine)
- [pickle](https://docs.python.org/3/library/pickle.html) (Python object serialization)
- [torch](https://pytorch.org/) (An open source machine learning framework that accelerates the path from research prototyping to production deployment)
 
# Run the Code
- Upload the Classification, Clustering and deployment parts ipynb code file into "Google Colab"  
- Press "Run All" in the control panel or "Restart Kernel and Run All" to run all code for Classification and  Clustering notebooks

- after get model publish files, we will Change Deployment the path that will add these publish file to  "%cd /content/drive/MyDrive/Data_set/recommender_system" in the deployment code notebook
- Press "Run All" in the control panel or "Restart Kernel and Run All" to run all code for deployment notebook
- In case of run each code cell alone, press the run button that appear at each code cell
 tents
- In case of run each code cell alone, press the run button that appear at each code cell
 tents
- copy ngork URL and paste it to dialogue flow (you will face a problem because we need to add the same structure of dialogue flow to your intents

 
# Contributing
Any kind of enhancement or contribution is welcomed.