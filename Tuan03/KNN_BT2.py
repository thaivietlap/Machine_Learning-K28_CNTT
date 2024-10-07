# import streamlit as st
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split

# # Hàm load data
# def loadCsv(filename) -> pd.DataFrame:
#     df = pd.read_csv(filename)
#     return df

# # Hàm biến đổi định tính (One-hot encoding)
# def transform(data, columns_trans): 
#     for i in columns_trans:
#         unique = data[i].unique() + '-' + i
#         matrix_0 = np.zeros((len(data), len(unique)), dtype = int)
#         frame_0 = pd.DataFrame(matrix_0, columns = unique)
#         for index, value in enumerate(data[i]):
#             frame_0.at[index, value + '-' + i] = 1
#         data[unique] = frame_0
#     return data

# # Hàm scale dữ liệu (Min-Max Scaling)
# def scale_data(data, columns_scale): 
#     for i in columns_scale:  
#         _max = data[i].max()
#         _min = data[i].min()
#         min_max_scaler = lambda x: round((x - _min) / (_max - _min), 3) if _max != _min else 0
#         data[i] = data[i].apply(min_max_scaler)
#     return data

# # Hàm tính khoảng cách Cosine
# def cosine_distance(train_X, test_X): 
#     dict_distance = dict()
#     for index, value in enumerate(test_X, start = 1):
#         for j in train_X:
#             result = np.sqrt(np.sum((j - value)**2))
#             if index not in dict_distance:
#                 dict_distance[index] = [result]
#             else:
#                 dict_distance[index].append(result)
#     return dict_distance

# # Hàm dự đoán dựa trên k khoảng cách gần nhất
# def pred_test(k, train_X, test_X, train_y):
#     lst_predict = list()
#     dict_distance = cosine_distance(train_X, test_X)
#     train_y = train_y.to_frame(name='target').reset_index(drop=True)
#     frame_concat = pd.concat([pd.DataFrame(dict_distance), train_y], axis=1)
#     for i in range(1, len(dict_distance) + 1):
#         sort_distance = frame_concat[[i, 'target']].sort_values(by=i, ascending=True)[:k]
#         target_predict = sort_distance['target'].value_counts(ascending=False).index[0]
#         lst_predict.append([i, target_predict])
#     return lst_predict

# # Đọc dữ liệu
# st.title('KNN Drug Classification')

# uploaded_file = st.file_uploader("Tải lên file CSV", type="csv")
# if uploaded_file is not None:
#     # Load dữ liệu
#     data = loadCsv(uploaded_file)

# # Biến đổi dữ liệu
# df = transform(data, ['Sex', 'BP', 'Cholesterol']).drop(['Sex', 'BP', 'Cholesterol'], axis=1)
# df = transform(data, ['Sex', 'BP', 'Cholesterol']).drop(['Sex', 'BP', 'Cholesterol'], axis=1)
# scale_data(df, ['Age', 'Na_to_K'])

# # Hiển thị dữ liệu đã được biến đổi
# st.write("Transformed Data:", df.head())

# # Tạo data_X và target
# data_X = df.drop(['Drug'], axis=1).values
# data_y = df['Drug']

# # Chia dữ liệu thành train và test
# X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=0)

# # Số k gần nhất
# k = st.slider('Select K value for KNN', 1, 10, 6)

# # Dự đoán
# test_pred = pred_test(k, X_train, X_test, y_train)
# df_test_pred = pd.DataFrame(test_pred).drop([0], axis=1)
# df_test_pred.index = range(1, len(test_pred) + 1)
# df_test_pred.columns = ['Predict']

# # Thực tế
# df_actual = pd.DataFrame(y_test)
# df_actual.index = range(1, len(y_test) + 1)
# df_actual.columns = ['Actual']

# # Kết quả dự đoán so với thực tế
# result_df = pd.concat([df_test_pred, df_actual], axis=1)

# # Hiển thị kết quả
# st.write("Prediction vs Actual:", result_df)
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Hàm load data
def loadCsv(filename) -> pd.DataFrame:
    df = pd.read_csv(filename)
    return df

# Hàm biến đổi định tính (One-hot encoding)
def transform(data, columns_trans): 
    for i in columns_trans:
        unique = data[i].unique() + '-' + i
        matrix_0 = np.zeros((len(data), len(unique)), dtype=int)
        frame_0 = pd.DataFrame(matrix_0, columns=unique)
        for index, value in enumerate(data[i]):
            frame_0.at[index, value + '-' + i] = 1
        data = pd.concat([data, frame_0], axis=1)  # Thay đổi ở đây
    return data

# Hàm scale dữ liệu (Min-Max Scaling)
def scale_data(data, columns_scale): 
    for i in columns_scale:  
        _max = data[i].max()
        _min = data[i].min()
        min_max_scaler = lambda x: round((x - _min) / (_max - _min), 3) if _max != _min else 0
        data[i] = data[i].apply(min_max_scaler)
    return data

# Hàm tính khoảng cách Euclidean
def euclidean_distance(train_X, test_X):  # Đổi tên hàm
    dict_distance = dict()
    for index, value in enumerate(test_X, start=1):
        for j in train_X:
            result = np.sqrt(np.sum((j - value) ** 2))
            if index not in dict_distance:
                dict_distance[index] = [result]
            else:
                dict_distance[index].append(result)
    return dict_distance

# Hàm dự đoán dựa trên k khoảng cách gần nhất
def pred_test(k, train_X, test_X, train_y):
    lst_predict = list()
    dict_distance = euclidean_distance(train_X, test_X)  # Sửa gọi hàm ở đây
    train_y = train_y.to_frame(name='target').reset_index(drop=True)
    frame_concat = pd.concat([pd.DataFrame(dict_distance), train_y], axis=1)
    for i in range(1, len(dict_distance) + 1):
        sort_distance = frame_concat[[i, 'target']].sort_values(by=i, ascending=True)[:k]
        target_predict = sort_distance['target'].value_counts(ascending=False).index[0]
        lst_predict.append([i, target_predict])
    return lst_predict

# Đọc dữ liệu
st.title('KNN Drug Classification')

uploaded_file = st.file_uploader("Tải lên file CSV", type="csv")
if uploaded_file is not None:
    # Load dữ liệu
    data = loadCsv(uploaded_file)

    # Biến đổi dữ liệu
    df = transform(data, ['Sex', 'BP', 'Cholesterol']).drop(['Sex', 'BP', 'Cholesterol'], axis=1)
    df = scale_data(df, ['Age', 'Na_to_K'])  # Sửa thứ tự

    # Hiển thị dữ liệu đã được biến đổi
    st.write("Transformed Data:", df.head())

    # Tạo data_X và target
    data_X = df.drop(['Drug'], axis=1).values
    data_y = df['Drug']

    # Chia dữ liệu thành train và test
    X_train, X_test, y_train, y_test = train_test_split(data_X, data_y, test_size=0.2, random_state=0)

    # Số k gần nhất
    k = st.slider('Select K value for KNN', 1, 10, 6)

    # Dự đoán
    test_pred = pred_test(k, X_train, X_test, y_train)
    df_test_pred = pd.DataFrame(test_pred).drop([0], axis=1)
    df_test_pred.index = range(1, len(test_pred) + 1)
    df_test_pred.columns = ['Predict']

    # Thực tế
    df_actual = pd.DataFrame(y_test)
    df_actual.index = range(1, len(y_test) + 1)
    df_actual.columns = ['Actual']

    # Kết quả dự đoán so với thực tế
    result_df = pd.concat([df_test_pred, df_actual], axis=1)

    # Hiển thị kết quả
    st.write("Prediction vs Actual:", result_df)
