"""
ReLU Activation Function Visualiser (Streamlit)

Name        : Loh Xiang Yi
Student ID  : SD24035

Web app description:
    An interactive Streamlit web app that visualises the Rectified Linear Unit (ReLU)
    activation function, which outputs max(0, x): the input if it is positive, otherwise zero.
    The app displays the formula, a plot of f(x) and its derivative f'(x)
    with an adjustable x-axis range, the values of f(x) and f'(x) for a
    chosen test input, and a single-neuron demo (weighted sum + bias passed
    through the activation function). It also explains the function's
    properties and why it matters in neural networks.

Course      : BSD3513 Introduction to Artificial Intelligence
Assignment  : Lab Report 3 - Chapter 4: Neural Networks

Run: streamlit run SD24035_Lab3_ReLU.py
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="ReLU Visualiser", page_icon="🧠", layout="wide")


# ---------- activation function and its derivative ----------
def activation(x):
    return np.maximum(0, x)


def derivative(x):
    return np.where(x > 0, 1.0, 0.0)


# ---------- page ----------
st.title("🧠 Rectified Linear Unit (ReLU)")
st.caption("BSD3513 Introduction to Artificial Intelligence | Loh Xiang Yi | SD24035 | Lab Report 3")
st.markdown(
    "This web app visualises the Rectified Linear Unit (ReLU) activation function. "
    "ReLU outputs the input if it is positive, otherwise zero, and is the default choice for hidden layers. "
    "The app shows the formula, an interactive plot of f(x) and its derivative f'(x) with an adjustable x-axis range, "
    "the values of f(x) and f'(x) for a chosen test input, and a single-neuron demo."
)
st.latex(r"f(x)=\max(0,x)")
st.latex(r"f'(x)=\begin{cases}1 & x>0\\0 & x\le 0\end{cases}")

st.sidebar.header("Controls")
x_min, x_max = st.sidebar.slider("x-axis range", -15.0, 15.0, (-6.0, 6.0), 0.5)
show_deriv = st.sidebar.checkbox("Show derivative f'(x)", True)
x_val = st.sidebar.slider("Test input x", float(x_min), float(x_max), 1.0, 0.1)

x = np.linspace(x_min, x_max, 600)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(x, activation(x), lw=2.5, label="f(x) = ReLU")
if show_deriv:
    ax.plot(x, derivative(x), lw=2, ls="--", label="f'(x)")
ax.scatter([x_val], [activation(x_val)], color="red", zorder=5, label="test input")
ax.axhline(0, color="grey", lw=0.8)
ax.axvline(0, color="grey", lw=0.8)
ax.set_xlabel("x (weighted sum + bias)")
ax.set_ylabel("output")
ax.set_title("Rectified Linear Unit (ReLU)")
ax.grid(alpha=0.3)
ax.legend()
col1, col2 = st.columns([2, 1])
col1.pyplot(fig)
col2.metric("f(x)", f"{float(activation(x_val)):.4f}")
col2.metric("f'(x)", f"{float(derivative(x_val)):.4f}")
col2.markdown("**Properties**\n\n- Range: [0, ∞)\n\n- Cheap to compute\n\n- Avoids vanishing gradient for x>0\n\n- Risk: 'dying ReLU' (gradient 0 for x≤0)")

# ---------- single neuron demo ----------
st.subheader("Single-neuron demo")
c1, c2, c3 = st.columns(3)
x1 = c1.number_input("Input x1", value=0.5)
x2 = c1.number_input("Input x2", value=-1.0)
w1 = c2.number_input("Weight w1", value=0.8)
w2 = c2.number_input("Weight w2", value=0.3)
b = c3.number_input("Bias b", value=0.1)
z = w1 * x1 + w2 * x2 + b
st.write(f"Weighted sum: z = w1·x1 + w2·x2 + b = **{z:.4f}**")
st.write(f"Neuron output: a = ReLU(z) = **{float(activation(z)):.4f}**")

st.subheader("Why does it matter?")
st.markdown("ReLU adds non-linearity while keeping gradients constant (1) for positive inputs, so deep networks train faster than with sigmoid/tanh.")