import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
import seaborn as sns
import numpy as np

class Predict:
    def __init__(self):
        faces = fetch_lfw_people(min_faces_per_person=60)
        self.num_images, self.h, self.w = faces.images.shape
        self.X = faces.data
        self.y = faces.target
        self.target_names = faces.target_names
        self.num_classes = self.target_names.shape[0]
        print("Data loaded.")
        print("Number of Images: ", self.num_images)
        print("Number of Features: ", self.X.shape[1])
        print("Number of classes: ", self.num_classes)

    def get_model(self):
        num_components = 150
        pca = PCA(n_components=num_components)
        params = {'C': [1,3, 5, 10], 'gamma': [0.001, 0.005, 0.0001, 0.0005,0.01,0.1], 'kernel': ['linear', 'rbf'], 'tol': [0.001, 0.0005, 0.0001]}
        svc = SVC()
        classfier = GridSearchCV(svc, params)
        self.model = make_pipeline(pca, classfier)
        self.best_parameters = svc.get_params()

def plot_result(images, titles, names_actual, h, w, n_row=5, n_col=5, fig_title="Prediction Result"):
    fig = plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=1, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        ax = fig.add_subplot(n_row, n_col, i + 1)
        ax.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        fc = 'black'
        if titles[i]!=names_actual[i] :
            fc = 'red'
        title = "Predicted: "+titles[i]+"\nActual: "+names_actual[i]
        ax.set_title(title, size=12,color=fc)
        plt.xticks(())
        plt.yticks(())
    if fig_title: 
        fig.suptitle(fig_title+'\n', fontsize=20)

    plt.savefig("results.png")
    plt.show(block=True)

def main():
    obj = Predict()
    obj.get_model()
    X_train, X_test, y_train, y_test = train_test_split(obj.X, obj.y, test_size=0.2, random_state=29)
    obj.model = obj.model.fit(X_train, y_train)
    pred = obj.model.predict(X_test)
    print('Best Parameters: \n ', obj.best_parameters)
    plot_result(X_test, obj.target_names[pred], obj.target_names[y_test], obj.h, obj.w)
    print(classification_report(y_test, pred, target_names=obj.target_names))
    conf_mat = confusion_matrix(y_test, pred)
    sns.heatmap(conf_mat, fmt='.2%', cmap='Blues')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.savefig('heatmap.png')

main()