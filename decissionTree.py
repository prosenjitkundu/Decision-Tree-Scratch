import numpy as np
import pandas as pd


class Node:
  def __init__(self, entropy, attribute, attribute_value, value=None, leaf_value=None, is_leaf=False):
    self.entropy = entropy
    self.attribute = attribute
    self.attribute_value = attribute_value
    self.decission = []
    self.value = value
    self.leaf_value = leaf_value
    self.is_leaf = is_leaf


class DecissionTree:
  def __init__(self):
    self.root = None

  def calculateEntropy(self, x):
    dictionary = {}
    for i in np.unique(x[:, -1]):
      dictionary[i] = 0
    n_data, _ = np.shape(x)
    for row in x:
      dictionary[row[-1]] += 1
    entropy = 0
    for item in dictionary.items():
      if item[1] != 0:
        entropy -= (item[1]/n_data) * np.log2(item[1]/n_data)
    return entropy

  def bestSplit(self, car_data, labels):
    best_nchild = 0
    information_gain = -float('inf')
    best_entropy = 0
    n_datamain, n_features = np.shape(car_data)
    target_split = []
    target_attr = []
    parent_entropy = self.calculateEntropy(car_data)
    print(parent_entropy)
    best_attr = -1
    for feature in range(n_features-1):
      n_feature = []
      t_attr = []
      ent = 0
      n_child = 0
      for attr in np.unique(car_data[:, feature]):
        t_attr.append(attr)
        n_child += 1
        x = np.array([x for x in car_data if x[feature] == attr])
        n_data, _ = np.shape(x)
        entropy = self.calculateEntropy(x)
        ent += n_data/n_datamain * entropy
        n_feature.append(x)
      if(parent_entropy-ent > information_gain):
        information_gain = parent_entropy - ent
        best_entropy = ent
        target_split = n_feature.copy()
        target_attr = t_attr.copy()
        best_attr = feature
        best_nchild = n_child
    target_split = np.delete(target_split, best_attr, 2)
    attr_value = labels[best_attr]
    labels = np.delete(labels, best_attr, 0)
    return {"labels": labels, "information_gain": information_gain, "entropy": best_entropy, "target_attr": target_attr, "attr": best_attr, "attr_value": attr_value, "target_split": target_split, "n_child": best_nchild}

  def buildDT(self, car_data, labels, value=None):
    n_data, n_features = np.shape(car_data)
    if n_data == 0:
      return Node(None, None, value, "unacc", True)
    if n_features == 1:
      leaf = self.createLeaf(car_data, value)
      return leaf
    split_dict = self.bestSplit(car_data, labels)
    if(split_dict["information_gain"] > 0):
      root = Node(split_dict["entropy"],
                  split_dict["attr"],
                  split_dict["attr_value"],
                  value, None, False)
      for i in range(np.shape(split_dict["target_split"])[0]):
        #print(i)
        root.decission.append(self.buildDT(
            split_dict["target_split"][i], split_dict["labels"], split_dict["target_attr"][i]))
      return root
    else:
      leaf = self.createLeaf(car_data, value)
      return leaf

  def createLeaf(self, car_data, value):
    x = list(car_data[:, -1])
    m = max(x, key=x.count)
    return Node(None, None, None, value, m, True)

  def predict(self,data):
    while(self.root.is_leaf !=True):
      pass


  def predictData(self,test_data):
    pass

  def fitData(self, car_data, labels=["price", "maint", "doors", "persons", "lug_boot", "safety", "buy"]):
    labels = np.array(labels)
    self.root = self.buildDT(car_data, labels, 0)

  def printDTRecursive(self, root, parent_attr, count, file):
    if(root):
      for i in range(count):
        file.write("|       ")
      if root.is_leaf == True:

        file.write(str(parent_attr)+" = "+str(root.value) +
                   " : "+str(root.leaf_value)+"\n")
      else:
        file.write(str(parent_attr)+" = "+str(root.value)+"\n")
        for i in range(len(root.decission)):
          self.printDTRecursive(
              root.decission[i], root.attribute_value, count+1, file)

  def printDT(self):
    file = open("dt.txt", "w")
    count = -1
    root = self.root
    if(root):
      for i in range(len(root.decission)):
        self.printDTRecursive(
            root.decission[i], root.attribute_value, count+1, file)
    file.close()


if __name__ == "__main__":
    columns = ["price", "maint", "doors",
               "persons", "lug_boot", "safety", "buy"]
    car_data = pd.read_csv(
        "project1 1.data", header=None, names=columns).values
    dtclassifier = DecissionTree()
    dtclassifier.fitData(car_data, columns)
    dtclassifier.printDT()
