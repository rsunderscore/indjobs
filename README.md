# indjobs
Read job descriptions from indeed.  
 - tokenize
 - remove stopwords - done automatically
 - stemming - not used in most methodologies
 - tf/idf
 - unsupervised classification
   - clustering - KNN, Kmeans
   - hierarchical - Decision Trees
   - SVM
   - Naive Bayes
   - cosine similarity
   - logistic regression

## References
 - [Document Classification Using Python and Machine Learning](https://www.digitalvidya.com/blog/document-classification-python-machine-learning/) - Muneeb Ahmad
 - [Naive Bayes Document Classification in Python](https://towardsdatascience.com/naive-bayes-document-classification-in-python-e33ff50f937e) - Kelly Epley
 - [Machine Learning, NLP: Text Classification using scikit-learn, python and NLTK](https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a) - Javed Shaikh 
   - do count vectorizer first?
   - also does SVM
  ```py  - from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
```
 - [Text Classification with Python and Scikit-Learn](https://stackabuse.com/text-classification-with-python-and-scikit-learn/) - Usman Malik -- this one is more about sentiment
   - can pass stopwords to CountVectorizer
   - also uses `TfidfTransformer`
   - RandomForest

> The script above uses CountVectorizer class from the sklearn.feature_extraction.text library. There are some important parameters that are required to be passed to the constructor of the class. The first parameter is the max_features parameter, which is set to 1500. This is because when you convert words to numbers using the bag of words approach, all the unique words in all the documents are converted into features. All the documents can contain tens of thousands of unique words. But the words that have a very low frequency of occurrence are unusually not a good parameter for classifying documents. Therefore we set the max_features parameter to 1500, which means that we want to use 1500 most occurring words as features for training our classifier.
 - :star: start here  [Classification of text documents using sparse features](https://scikit-learn.org/stable/auto_examples/text/plot_document_classification_20newsgroups.html) - scikit learn web site
   - more algorithm choices (15 total) incl: peceptron, Ridge, SGD, linearSVC, 3x Bayes Variants, Rocchio (nearest centroid)
   - HashingVectorizer
   - use tfidf transformer directly (without first doing count)?

Algorithm options:
- K-nearest neighbour algorithms
- SVM
- Decision trees such as ID3 or C4.5
- Naive Bayes classifier
- Expectation maximization (EM)
- Instantaneously trained neural networks
- Latent semantic indexing
- Support vector machines (SVM)
- Artificial neural network
- Concept Mining
- Rough set-based classifier
- Soft set-based classifier
- Multiple-instance learning
- Natural language processing approaches

more algorithms from [sklearn site](https://scikit-learn.org/stable/modules/clustering.html#)
- Spectral
- DBSCAN

## Clustering

Clustering cannot use the same accuracy measures as classification because the numberic label assignments are arbitrary (e.g. the label predicted label of 0 is the first group determined and un related to the y_train label encoded as 0).  

- https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
- https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation

Evaluating clustering
- **Inertia** - how internall coherent clusters are
    - works best with ball shaped clusters
    - how close all cluster members are to the centroid for their cluster
    - some of squared distances from all points to nearest centroid
    - lower numbers are better
    - kmeans.inertia_ (kmeans attempts to minimize)
        - can be used to determine the optimal number of clustesrs (if ball shaped)
- **purity**
    1. assign a label to the class based on the class that is most frequent in that cluster
    2. count the number of correctly assigned documents and divide by the total number of items
    1. not implemented directly in sklearn (probaly b/c usefulness is limited)
    1. using contingency matrix (aka confusion matrix) - where TP is based on the most common of the actual label assignments for that cluster
        - $ \frac{TP}{N} $ where $ N= TP+TN+FP+FN$
        - sklearn.metrics.cluster.contingency_matrix(y_act, y_pred)
- **NMI (normalized mutual information)** - compensates for high purity obtaine when n_clusters is large - assumes max likelihood estimatse.  Given cluster as predicted cluster membership and...  class as actual labels based on human judgement
    - information gain from knowing the class - if the clustering is random compared to classes then no information is gained (0 to 1)
    - normalizing by entropy penalizes large settings of n_clusters, b/c it increases with n_clusters making overall result smaller
    - log of 
        - probability an item being in both cluster and class (intersection)
        - div by the product of the probability of it being in the class and the probability of it being in the cluster
    - divided by entropy
    - `sklearn.metrics.normalized_mutual_info_score(labels_true, labels_pred)` *- OR -* `adjusted_mutual_info_score` *- OR -*  `mutual_info_score`
        - e.g. how often do predicted assignments match with actual
        - adjusted is adjusted for chance, others are not
        - ignores group(label) names
        - values tend toward 0
- **Rand index** - penalizes both false positive and false negative  viewing the clustering as a series of decisions what is the percentage of these decisions that was correct
    - accuracy formula
    - `sklearn.metrics.adjusted_rand_score(labels_true, labels_pred)`
        - renaming labels doesn't change the score
        - values tend toward 1 even for very bad mappings
    - similarity of the pair cluster assignments, ignoring permutations
    - create confusion matrix of 
        - TP - two documents are similar and in same cluster (correct)
        - TN - two dissimilar documents documents in different clusters (correct)
        - FP - two dissimilar documents documents in same cluster (error)
        - FN - two similar documents assigned to diff clusters

$ \mbox{RI} = \frac{\mbox{TP}+\mbox{TN}}{\mbox{TP}+\mbox{FP}+\mbox{FN}+\mbox{TN}} $

- **F measure** - penalty for FP and NP and also allows for weights on these
    - Separating similar documents is sometimes worse than putting pairs of dissimilar documents in the same cluster.
    - use a weight $ \beta $ gives more weight to recall (over precision) by penalizing false negatives more than false 
positives -- e.g. it is better to say that two things are related (in the same cluster) and they aren't then to say they aren't related (in diff clusters) and they are

$ \begin{pmatrix}6\\2 \end{pmatrix} = C(6,2) = $ combination of 6 items taken 2 at a time - there is no corresponding way to represent permutations

## interpreting multi-class confusion matrix
[helps a bit but wording is confusing](https://www.analyticsvidhya.com/blog/2021/06/confusion-matrix-for-multi-class-classification/)

|population = P+N|pred_pos|pred_neg|
| --- | --- | --- |
|act_pos|TP|FN|
|act_neg|FP|TN|

- TP = 
    - prediction was correct - the diagonal of CF
    - CF\[0,0\]
- FN = 
    - predicted something other than actual 
    - for row 0 the sum of all values in columns\[1:\] for this row
    - The sum of valuesiun the row except the TP value

- FP = 
    - predicted value was incorrect
    - for column 0 the sum of all values in rows\[1:\] 
    - The sum of values of corresponding column except the TP value.

- TN = 
    - we poredicted it wasn't this, and it isn't - 
    - The sum of everything that isn't in the row or column for the class

performance:
- Accuracy (ACC) = (TP + TN)/(P + N) = (true_pos+true_neg) / (act_pos+act_neg)
- precision= TP/PP = true_postive / sum(pred_pos)
- recall/sensitivity = TP/P  = true_positive / sum(act_pos)

#### notes from a bad site with lots of tricksy pop-up ads
- accuracy
    - range 0 to 1
    - calc = (TP+TN) divided by 'total number of a dataset P+N' (what does that mean? everything)
    - balanced accruacy average of proprotion of corrects of each class individually 
    
#### notes from a less bad site
https://medium.com/analytics-vidhya/confusion-matrix-accuracy-precision-recall-f1-score-ade299cf63cd
same author as the link above?
- accuracy = $ \large \frac{(TN+TP)}{(TN+TP+FP+FN)} $
- precision = $ \large \frac{TP}{(TP+FP)} $
- recall = $ \large\frac{TP}{(TP+FN)} $
-  F1 $ = 2 \times \large \frac{(precision \times recall)}{(precision + recall)} $ - is this being used in place of the accuracy score because accuracy == bad

#### notes from BMC site
https://www.bmc.com/blogs/confusion-precision-recall/
- Precision = TP/(TP + FP)
- Recall = TP/(TP + FN)
