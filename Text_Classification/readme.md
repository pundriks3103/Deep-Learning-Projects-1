# Text Classification using CNN
---

Text classification is the process of assigning tags or categories to text according to its content. It’s one of the fundamental tasks in Natural Language Processing (NLP) with broad applications such as sentiment analysis, topic labeling, spam detection, and intent detection.

Different approaches to text classification model:

1. Rule Based Systems
  
   Rule-based approaches classify text into organized groups by using a set of handcrafted linguistic rules. These rules instruct the system to use semantically relevant elements of a text to identify relevant categories based on its content. (For eg. - counting the number of positive and negative words for sentiment analysis.)

2. Machine Learning Approach

    Instead of relying on manually crafted rules, text classification with machine learning learns to make classifications based on past observations. By using pre-labeled examples as training data, a machine learning algorithm can learn the different associations between pieces of text and that a particular output (i.e. tags) is expected for a particular input (i.e. text).
  
## Steps using ML Approach

1. Feature Extraction

    A method is used to transform each text into a numerical representation in the form of a vector. After this, the array size of all different sentences will be different. So, we padd each array so that the size of every array is same. The most effective padding is when we choose length of longest array as the maximum length of each sentence. 

2. Classifier Model

    Many models can be used such as Naive Bayes Classifier, SVM, CNN or RNN. In this repository, I have used CNN Model.

## Example on Dataset

1. Go through the [Example](https://github.com/shubhamjain02/Deep-Learning-Projects/blob/master/Text_Classification/text_classification_example.ipynb) notebook to see an example where this model is used on the dataset give by the company Mylo.

