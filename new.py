import math
import pandas as pd
import csv
# Calcul de l'entropie
def entropy(data):
    total = len(data)
    counts = data['Survived'].value_counts()
    ent = 0
    for count in counts:
        prob = count / total
        ent -= prob * math.log2(prob)
    return ent

# Calcul du gain d'information
def information_gain(data, split_attr):
    total_entropy = entropy(data)
    values = data[split_attr].unique()
    weighted_entropy = 0
    for value in values:
        subset = data[data[split_attr] == value]
        weighted_entropy += (len(subset) / len(data)) * entropy(subset)
    return total_entropy - weighted_entropy

# Trouver le meilleur attribut pour séparer les données
def find_best_split(data, attributes):
    best_gain = 0
    best_attr = None
    print(f"Attribute: {attributes}")
    for attr in attributes:
        gain = information_gain(data, attr)
        print(f"best_gain: {gain}")
        if gain > best_gain:
            best_gain = gain
            best_attr = attr
            print(f"best_attrrrrrrrrrrrrrrrrrrrrrrrrr: {best_attr}")
            print(f"Colonnes disponibles : {data.columns}")

    return best_attr

# Construction de l'arbre de décision
def build_tree(data, attributes):
    # Si toutes les instances ont la même classe, retourner cette classe
    if len(data['Survived'].unique()) == 1:
        return data['Survived'].iloc[0]
    
    # Si plus d'attributs, retourner la majorité
    if not attributes:
        return data['Survived'].mode()[0]
    
    # Trouver le meilleur attribut pour la séparation
    best_attr = find_best_split(data, attributes)
    tree = {best_attr: {}}
    
    if best_attr is None:
        print("Erreur : 'best_attr' est None.")
        return 1  # ou gérer l'erreur
    # Construire des branches pour chaque valeur de l'attribut
    for value in data[best_attr].unique():
        subset = data[data[best_attr] == value]
        subtree = build_tree(subset, [attr for attr in attributes if attr != best_attr])
        tree[best_attr][value] = subtree
    
    return tree

# Prédiction avec l'arbre de décision
def predict(tree, instance):
    if not isinstance(tree, dict):
        return tree
    attr = next(iter(tree))
    value = instance[attr]
    if value in tree[attr]:
        return predict(tree[attr][value], instance)
    else:
        return 0  # Si la valeur n'est pas dans l'arbre, on retourne une valeur par défaut

# Charger les données
data = pd.read_csv('train.csv')
data2 = pd.read_csv('test.csv')
# Prétraitement des données
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Attributs pour la construction de l'arbre
attributes = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

# Construire l'arbre de décision
tree = build_tree(data, attributes)

# Exemple de prédiction
example_instance = data.iloc[0]

predictions = []
for index, rows in data2.iterrows():
    passenger_id = rows['PassengerId']
    predicted_class = predict(tree, rows)
    predictions.append({'PassengerId': passenger_id, 'Survived': predicted_class})

# Fonction pour écrire les résultats dans un fichier CSV
def write_csv(filename, data, fieldnames):
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
# Écrire les résultats dans un fichier CSV
fieldnames = ['PassengerId', 'Survived']
write_csv('submission.csv', predictions, fieldnames)


print(f"Prédiction pour l'instance: {predictions}")
