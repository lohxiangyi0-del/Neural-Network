"""
Sigmoid Activation Function Visualiser (Streamlit)

Name        : Loh Xiang Yi
Student ID  : SD24035
Course      : BSD3513 Introduction to Artificial Intelligence
Assignment  : Lab Report 3 - Chapter 4: Neural Networks

Web app description:
    An interactive Streamlit web app that visualises the Sigmoid
    activation function, which squashes any input into the range (0, 1).
    The app displays the formula, a plot of f(x) and its derivative f'(x)
    with an adjustable x-axis range, the values of f(x) and f'(x) for a
    chosen test input, and a single-neuron demo (weighted sum + bias passed
    through the activation function). It also explains the function's
    properties and why it matters in neural networks.

Run: streamlit run SD24035_Lab3_Sigmoid.py
"""
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Sigmoid Visualiser", page_icon="🧠", layout="wide")


# ---------- activation function and its derivative ----------
def activation(x):
    return 1 / (1 + np.exp(-x))


def derivative(x):
    s = activation(x)
    return s * (1 - s)


# ---------- page ----------
st.title("🧠 Sigmoid")
st.caption("BSD3513 Introduction to Artificial Intelligence | Loh Xiang Yi | SD24035 | Lab Report 3")
st.markdown(
    "This web app visualises the Sigmoid activation function. "
    "Sigmoid squashes any input into the range (0, 1), so its output can be read as a probability. It is commonly used in output layers for binary classification. "
    "The app shows the formula, an interactive plot of f(x) and its derivative f'(x) with an adjustable x-axis range, "
    "the values of f(x) and f'(x) for a chosen test input, and a single-neuron demo."
)
st.latex(r"f(x)=\frac{1}{1+e^{-x}}")
st.latex(r"f'(x)=f(x)\,(1-f(x))")


st.sidebar.header("Controls")
x_min, x_max = st.sidebar.slider("x-axis range", -15.0, 15.0, (-6.0, 6.0), 0.5)
show_deriv = st.sidebar.checkbox("Show derivative f'(x)", True)
x_val = st.sidebar.slider("Test input x", float(x_min), float(x_max), 1.0, 0.1)

x = np.linspace(x_min, x_max, 600)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(x, activation(x), lw=2.5, label="f(x) = Sigmoid")
if show_deriv:
    ax.plot(x, derivative(x), lw=2, ls="--", label="f'(x)")
ax.scatter([x_val], [activation(x_val)], color="red", zorder=5, label="test input")
ax.axhline(0, color="grey", lw=0.8)
ax.axvline(0, color="grey", lw=0.8)
ax.set_xlabel("x (weighted sum + bias)")
ax.set_ylabel("output")
ax.set_title("Sigmoid")
ax.grid(alpha=0.3)
ax.legend()
col1, col2 = st.columns([2, 1])
col1.pyplot(fig)
col2.metric("f(x)", f"{float(activation(x_val)):.4f}")
col2.metric("f'(x)", f"{float(derivative(x_val)):.4f}")
col2.markdown("**Properties**\n\n- Range: (0, 1)\n\n- Smooth and differentiable\n\n- Max gradient 0.25 at x=0\n\n- Saturates for large |x| (vanishing gradient)\n\n- Not zero-centred")

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
st.write(f"Neuron output: a = Sigmoid(z) = **{float(activation(z)):.4f}**")

st.subheader("Why does it matter?")
st.markdown("Sigmoid is used in output layers for binary classification. In deep hidden layers its small gradients make learning slow.")