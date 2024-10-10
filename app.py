import streamlit as st
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

st.title("DT Classifier By Mr.Sajan Sah")
st.sidebar.header("Hyper Parameters (Mr.Sajan)")

# Model parameters
criterion = st.sidebar.selectbox("Criterion", ["gini", "entropy"])
splitter = st.sidebar.selectbox("Splitter", ["best", "random"])
max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 50, 2)
min_samples_leaf = st.sidebar.slider("Min Samples Leaf", 1, 50, 1)
max_features = st.sidebar.slider("Max Features", 1, 2, 2)
max_leaf_nodes = st.sidebar.slider("Max Leaf Nodes", 2, 50, 10)
min_impurity_decrease = st.sidebar.slider("Min Impurity Decrease", 0.0, 0.5, 0.0)

# Button to trigger model training and visualization
if st.sidebar.button("Run Algorithm"):
    # Generate dataset
    X, y = make_classification(n_samples=500, n_features=2, n_classes=2, n_clusters_per_class=1, n_informative=2, n_redundant=0, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Classifier
    clf = DecisionTreeClassifier(
        criterion=criterion,
        splitter=splitter,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease
    )

    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Display accuracy
    st.write(f"Accuracy for Decision Tree: {accuracy:.2f}")

    # Plot decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotting
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=ListedColormap(['#FF0000', '#0000FF']))
    ax.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap=ListedColormap(['#FF0000', '#0000FF']))

    # Display plot in Streamlit
    st.pyplot(fig)

    # Plot the decision tree
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_tree(clf, filled=True, feature_names=['Col1', 'Col2'], class_names=['Class 0', 'Class 1'], ax=ax)
    st.pyplot(fig)
