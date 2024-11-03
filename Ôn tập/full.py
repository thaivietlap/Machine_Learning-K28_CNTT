# Import các thư viện cần thiết
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

# --- Naive Bayes ---
# Câu 1: Naive Bayes trên tập dữ liệu Iris
data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_nb = GaussianNB()
model_nb.fit(X_train, y_train)
y_pred = model_nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Câu 1: Độ chính xác của mô hình Naive Bayes:", accuracy)

# Câu 2: Vẽ ma trận nhầm lẫn
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Naive Bayes on Iris Dataset")
plt.gcf().manager.set_window_title("Câu 2")
plt.show()

# --- K-Nearest Neighbors (KNN) ---
# Câu 3: KNN trên tập dữ liệu Wine
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

print(f"Câu 3: Độ chính xác của mô hình KNN với k = {k}:", accuracy)
print("Độ nhạy:", recall)
print("Độ chính xác (precision):", precision)

# Câu 4: Thử nghiệm với các giá trị k khác nhau
k_values = [1, 3, 5, 7, 9]
accuracies = []
for k in k_values:
    model_knn = KNeighborsClassifier(n_neighbors=k)
    model_knn.fit(X_train, y_train)
    y_pred = model_knn.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))

plt.plot(k_values, accuracies, marker='o')
plt.xlabel("Số lượng láng giềng (k)")
plt.ylabel("Độ chính xác")
plt.title("Độ chính xác của KNN theo giá trị của k trên tập dữ liệu Wine")
plt.gcf().canvas.set_window_title("Câu 4")
plt.show()

# --- Cây quyết định ---
# Câu 5: Cây quyết định trên tập dữ liệu Breast Cancer
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model_dt = DecisionTreeClassifier(random_state=42)
model_dt.fit(X_train, y_train)
accuracy = model_dt.score(X_test, y_test)
print("Câu 5: Độ chính xác của mô hình cây quyết định:", accuracy)

plt.figure(figsize=(15, 10))
plot_tree(model_dt, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.title("Decision Tree for Breast Cancer Dataset")
plt.show()

# --- Support Vector Machine (SVM) ---
# Câu 6: SVM với kernel tuyến tính trên tập dữ liệu Digits
data = load_digits()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_svm = SVC(kernel='linear')
model_svm.fit(X_train, y_train)
accuracy = model_svm.score(X_test, y_test)
print("Câu 6: Độ chính xác của SVM với kernel tuyến tính:", accuracy)

# Câu 7: So sánh các kernel khác
kernels = ['linear', 'rbf', 'poly']
for kernel in kernels:
    model_svm = SVC(kernel=kernel)
    start_time = time.time()
    model_svm.fit(X_train, y_train)
    train_time = time.time() - start_time
    accuracy = model_svm.score(X_test, y_test)
    
    print(f"Câu 7: Kernel {kernel}: Độ chính xác = {accuracy}, Thời gian huấn luyện = {train_time:.2f} giây")

# --- Multilayer Perceptron (MLP) ---
# Câu 8: MLP với hai tầng ẩn trên tập dữ liệu MNIST
data = fetch_openml('mnist_784', version=1)
X, y = data.data, data.target

scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=20, random_state=42)
# mlp.fit(X_train, y_train)
# accuracy = mlp.score(X_test, y_test)
# print("Độ chính xác của MLP trên tập kiểm tra:", accuracy)
# Huấn luyện MLP với 2 tầng ẩn và tăng số lần lặp tối đa
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, random_state=42)
mlp.fit(X_train, y_train)
accuracy = mlp.score(X_test, y_test)
print("Câu 8: Độ chính xác của MLP trên tập kiểm tra:", accuracy)
