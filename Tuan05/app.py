import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import streamlit as st

# Function to calculate SVM (Hard Margin)
def hard_margin_svm():
    # Define data points
    x = np.array([[1., 3.], [2., 2.], [1., 1.]])
    y = np.array([[1.], [1.], [-1.]])
    
    # Calculate H matrix
    H = np.dot(x * y, (x * y).T)

    # Construct matrices for QP
    n = x.shape[0]
    P = matrix(H)
    q = matrix(-np.ones((n, 1)))
    G = matrix(-np.eye(n))
    h = matrix(np.zeros(n))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))

    # Solver parameters for precision
    solvers.options['abstol'] = 1e-10
    solvers.options['reltol'] = 1e-10
    solvers.options['feastol'] = 1e-10

    # Perform QP
    sol = solvers.qp(P, q, G, h, A, b)
    
    # Solution λ
    lamb = np.array(sol['x'])
    
    # Calculate w and b
    w = np.sum(lamb * y * x, axis=0)
    sv_idx = np.where(lamb > 1e-5)[0]
    sv_x = x[sv_idx]
    sv_y = y[sv_idx]
    b = np.mean(sv_y - np.dot(sv_x, w))
    
    # Plot data points and decision boundary
    plt.figure(figsize=(5,5))
    color= ['red' if a == 1 else 'blue' for a in y]
    plt.scatter(x[:, 0], x[:, 1], s=200, c=color, alpha=0.7)
    plt.xlim(0, 4)
    plt.ylim(0, 4)

    # Decision boundary
    x1_dec = np.linspace(0, 4, 100)
    x2_dec = -(w[0] * x1_dec + b) / w[1]
    plt.plot(x1_dec, x2_dec, c='black', lw=1.0, label='decision boundary')
    
    # Support vectors
    plt.scatter(sv_x[:, 0], sv_x[:, 1], s=50, marker='o', c='white')

    plt.title("Hard Margin SVM")
    plt.legend()
    plt.show()

    return lamb.flatten(), w, b


# Function to calculate SVM (Soft Margin)
def soft_margin_svm(C):
    x = np.array([[0.2, 0.869], [0.687, 0.212], [0.822, 0.411], [0.738, 0.694], [0.176, 0.458]])
    y = np.array([-1, 1, 1, 1, -1]).astype('float').reshape(-1, 1)

    N = x.shape[0]
    H = np.dot(y * x, (y * x).T)

    P = matrix(H)
    q = matrix(-np.ones((N, 1)))
    A = matrix(y.reshape(1, -1))
    b = matrix(np.zeros(1))

    G = matrix(np.vstack([-np.eye(N), np.eye(N)]))
    h = matrix(np.hstack([np.zeros(N), np.ones(N) * C]))

    # Solve QP
    sol = solvers.qp(P, q, G, h, A, b)
    lamb = np.array(sol['x'])

    # Calculate w and b
    w = np.sum(lamb * y * x, axis=0)
    sv_idx = np.where(lamb > 1e-5)[0]
    sv_x = x[sv_idx]
    sv_y = y[sv_idx]
    b = np.mean(sv_y - np.dot(sv_x, w))

    # Plot
    plt.figure(figsize=(5,5))
    color = ['red' if a == 1 else 'blue' for a in y]
    plt.scatter(x[:, 0], x[:, 1], s=200, c=color, alpha=0.7)
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Decision boundary
    x1_dec = np.linspace(0, 1, 100)
    x2_dec = -(w[0] * x1_dec + b) / w[1]
    plt.plot(x1_dec, x2_dec, c='black', lw=1.0, label='decision boundary')

    plt.title(f"Soft Margin SVM (C={C})")
    plt.legend()
    plt.show()

    return lamb.flatten(), w, b

# Streamlit interface
st.title("Support Vector Machine (SVM) Visualization")

option = st.sidebar.selectbox(
    "Choose the SVM type",
    ("Hard Margin SVM", "Soft Margin SVM"))

if option == "Hard Margin SVM":
    st.write("### Hard Margin SVM")
    lamb, w, b = hard_margin_svm()
    st.write(f"λ = {lamb}")
    st.write(f"w = {w}")
    st.write(f"b = {b}")

elif option == "Soft Margin SVM":
    st.write("### Soft Margin SVM")
    C = st.sidebar.slider("Select C value", 0.1, 100.0, 50.0)
    lamb, w, b = soft_margin_svm(C)
    st.write(f"λ = {lamb}")
    st.write(f"w = {w}")
    st.write(f"b = {b}")

