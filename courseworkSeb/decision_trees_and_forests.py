"""
Header


"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn import ensemble
from sklearn.tree import *
from sklearn import tree

##############################################

# Split & Shuffle Dataset
scaled_data = pd.read_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/scaled_dataset.csv')
clean_data = pd.read_csv('/Users/seb/PycharmProjects/ACS341/courseworkSeb/clean_dataset.csv')
clean_data = clean_data.astype('float')

train, test = train_test_split(scaled_data, test_size=0.2) #using 20% of our data.

train_target = train['Failed_Yes']
test_target = test['Failed_Yes']
train = train.drop('Failed_Yes', axis=1)
test = test.drop('Failed_Yes', axis=1)


##############################################
# Decision trees - analysing alphas (pruning)
decision_tree = DecisionTreeClassifier(random_state=0)
path = decision_tree.cost_complexity_pruning_path(train, train_target)
ccp_alphas, impurities = path.ccp_alphas, path.impurities


fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")
plt.savefig('/Users/seb/PycharmProjects/ACS341/courseworkSeb/graphs_outputted/impurity_vs_alpha.png')

clfs = []

for ccp_alpha in ccp_alphas:
    decision_tree = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    decision_tree.fit(train, train_target)
    clfs.append(decision_tree)


print(
    "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)


clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [decision_tree.tree_.node_count for decision_tree in clfs]
depth = [decision_tree.tree_.max_depth for decision_tree in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()
plt.savefig('/Users/seb/PycharmProjects/ACS341/courseworkSeb/graphs_outputted/tree_nodes_depth.png')


train_scores = [decision_tree.score(train, train_target) for decision_tree in clfs]
test_scores = [decision_tree.score(test, test_target) for decision_tree in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("Alpha")
ax.set_ylabel("Accuracy")
ax.set_title("Accuracy vs Alpha for Training and Testing Sets")
ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.savefig('/Users/seb/PycharmProjects/ACS341/courseworkSeb/graphs_outputted/dtree_train_test.png')


######################################################################
# decision_tree = decision_tree.fit(train, train_target)

final_tree = DecisionTreeClassifier(ccp_alpha=0.02, random_state=0)
final_tree = final_tree.fit(train, train_target)
plt.figure(figsize=(15, 15))


tree.plot_tree(final_tree, feature_names=train.columns)
plt.savefig('/Users/seb/PycharmProjects/ACS341/courseworkSeb/graphs_outputted/final_decision_tree.png')

#############################################
# Evaluation metrics
# https://scikit-learn.org/stable/modules/model_evaluation.html

score = final_tree.score(test, test_target)
predictions = final_tree.predict(test)
print("The accuracy is: " + str(score*100) + "%")

tree_confusion_matrix = confusion_matrix(test_target, predictions)

# print(tree_confusion_matrix)
f = plt.figure(figsize=(30, 25))
ax = sns.heatmap(tree_confusion_matrix,
                 annot=True,
                 fmt=".3f",
                 linewidths=1,
                 vmin=0,
                 vmax=100,
                 square=True,
                 cmap='flare',
                 annot_kws={"size": 70 / np.sqrt(len(tree_confusion_matrix))})


plt.ylabel('Actual label', fontsize=45)
plt.xlabel('Predicted label', fontsize=45)
all_sample_title = 'Decision Tree Confusion Matrix'
cax = plt.gcf().axes[-1]
cax.tick_params(labelsize=50)
plt.title(all_sample_title, fontsize=50)
f.savefig('/Users/seb/PycharmProjects/ACS341/courseworkSeb/graphs_outputted/tree_corr_matrix.png')


tree_accuracy = accuracy_score(test_target, predictions)
print("The tree accuracy is given as: " + str(tree_accuracy*100)+"%")

tree_precision = precision_score(test_target, predictions)
print("The tree precision is given as: " + str(tree_precision*100)+"%")

tree_recall = recall_score(test_target, predictions)
print("The tree recall is given as: " + str(tree_recall*100)+"%")

tree_confusion_matrix = confusion_matrix(test_target, predictions)
print("The confusion matrix: ", tree_confusion_matrix)

# plt.show()
