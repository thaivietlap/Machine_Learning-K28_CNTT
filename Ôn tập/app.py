import streamlit as st
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits, fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import time

st.title("Machine Learning Models Showcase")

# --- Naive Bayes ---
st.header("Câu 1: Naive Bayes trên tập dữ liệu Iris")
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_nb = GaussianNB()
model_nb.fit(X_train, y_train)
y_pred = model_nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write("Độ chính xác của mô hình Naive Bayes:", accuracy)

# Câu 2: Vẽ ma trận nhầm lẫn
st.subheader("Câu 2: Ma trận nhầm lẫn cho Naive Bayes trên Iris")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(cmap=plt.cm.Blues, ax=ax)
st.pyplot(fig)

# --- K-Nearest Neighbors (KNN) ---
st.header("Câu 3: K-Nearest Neighbors (KNN) trên tập dữ liệu Wine")
data = load_wine()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

k = 5
model_knn = KNeighborsClassifier(n_neighbors=k)
model_knn.fit(X_train, y_train)
y_pred = model_knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='macro')
precision = precision_score(y_test, y_pred, average='macro')

st.write(f"Độ chính xác của mô hình KNN với k = {k}:", accuracy)
st.write("Độ nhạy:", recall)
st.write("Độ chính xác (precision):", precision)

# Câu 4: Thử nghiệm với các giá trị k khác nhau
st.subheader("Câu 4: Độ chính xác của KNN theo các giá trị k khác nhau trên tập dữ liệu Wine")
k_values = [1, 3, 5, 7, 9]
accuracies = []
for k in k_values:
    model_knn = KNeighborsClassifier(n_neighbors=k)
    model_knn.fit(X_train, y_train)
    y_pred = model_knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

fig, ax = plt.subplots()
ax.plot(k_values, accuracies, marker='o')
ax.set_xlabel("Số lượng láng giềng (k)")
ax.set_ylabel("Độ chính xác")
ax.set_title("Độ chính xác của KNN theo giá trị của k trên tập dữ liệu Wine")
st.pyplot(fig)

# --- Cây quyết định ---
st.header("Câu 5: Decision Tree trên tập dữ liệu Breast Cancer")
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
accuracy = model_dt.score(X_test, y_test)
st.write("Độ chính xác của mô hình cây quyết định:", accuracy)

fig, ax = plt.subplots(figsize=(15, 10))
plot_tree(model_dt, filled=True, feature_names=data.feature_names, class_names=data.target_names, ax=ax)
st.pyplot(fig)

# --- Support Vector Machine (SVM) ---
st.header("Câu 6: SVM với kernel tuyến tính trên tập dữ liệu Digits")
data = load_digits()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_svm = SVC(kernel='linear')
model_svm.fit(X_train, y_train)
accuracy = model_svm.score(X_test, y_test)
st.write("Độ chính xác của SVM với kernel tuyến tính:", accuracy)

# Câu 7: So sánh các kernel khác
st.subheader("Câu 7: So sánh các kernel của SVM")
kernels = ['linear', 'rbf', 'poly']
for kernel in kernels:
    model_svm = SVC(kernel=kernel)
    start_time = time.time()
    model_svm.fit(X_train, y_train)
    train_time = time.time() - start_time
    accuracy = model_svm.score(X_test, y_test)
    st.write(f"Kernel {kernel}: Độ chính xác = {accuracy}, Thời gian huấn luyện = {train_time:.2f} giây")

# --- Multilayer Perceptron (MLP) ---
st.header("Câu 8: Multilayer Perceptron (MLP) trên tập dữ liệu MNIST")
st.write("Vui lòng chờ... Tập dữ liệu lớn đang được tải.")
data = fetch_openml('mnist_784', version=1)
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, random_state=42)
mlp.fit(X_train, y_train)
accuracy = mlp.score(X_test, y_test)
st.write("Độ chính xác của MLP trên tập kiểm tra:", accuracy)
