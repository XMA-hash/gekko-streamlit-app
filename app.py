#python -m streamlit run app.py
###############################

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from gekko import GEKKO

st.title("Ajustament de models amb GEKKO")

# ============================
# ENTRADA DE DADES
# ============================
st.markdown("### Introdueix les dades X i Y")

x_input = st.text_input("Valors de X (separats per comes)", "0,1,2,3,4,5,6")
y_input = st.text_input("Valors de Y (separats per comes)", "1.2,2.3,2.9,3.8,5.1,5.9,6.8")

if st.button("Executar ajustament"):
    try:
        X = [float(x) for x in x_input.split(",")]
        Y = [float(y) for y in y_input.split(",")]
        assert len(X) == len(Y), "X i Y han de tenir la mateixa longitud."

        # -------------------------------------------------------------------------
        # FUNCIONS AUXILIARS
        # -------------------------------------------------------------------------
        def fit_error(m, y_pred, Y):
            return m.Intermediate(sum((y_pred[i] - Y[i])**2 for i in range(len(Y))))

        # =======================
        # MODEL T1
        # =======================
        m1 = GEKKO(remote=False)
        a1 = m1.Var(0, lb=-10, ub=10)
        b1 = m1.Var(0.1)
        c1 = m1.Var(0, lb=-10, ub=10)
        y1 = [a1 + b1*(X[i]-c1)**3 for i in range(len(X))]
        f1 = fit_error(m1, y1, Y)
        m1.Obj(f1)
        m1.solve(disp=False)

        # =======================
        # MODEL T2
        # =======================
        m2 = GEKKO(remote=False)
        a2 = m2.Var(0)
        b2 = m2.Var(0.1, lb=0.1)
        c2 = m2.Var(0.1, lb=0.1)
        d2 = m2.Var(0)
        y2 = [a2 + b2*m2.exp(-c2*(X[i]-d2)) for i in range(len(X))]
        f2 = fit_error(m2, y2, Y)
        m2.Obj(f2)
        m2.Equation(a2 + b2*m2.exp(-c2*(X[-1]-d2)) >= -10)
        m2.solve(disp=False)

        # =======================
        # MODEL T3
        # =======================
        m3 = GEKKO(remote=False)
        a3 = m3.Var(0)
        b3 = m3.Var(0.1, lb=0.1)
        c3 = m3.Var(0.1, lb=0.1)
        d3 = m3.Var(0)
        y3 = [a3 - b3*m3.exp(c3*(X[i]-d3)) for i in range(len(X))]
        f3 = fit_error(m3, y3, Y)
        m3.Obj(f3)
        m3.Equation(a3 - b3*m3.exp(c3*(X[0]-d3)) <= 10)
        m3.solve(disp=False)

        # =======================
        # MODEL T4
        # =======================
        m4 = GEKKO(remote=False)
        a4 = m4.Var(0, lb=-10, ub=10)
        b4 = m4.Var(0.1, lb=0.1)
        c4 = m4.Var(0.1, lb=0.1, ub=10)
        d4 = m4.Var(0, lb=-10, ub=10)
        y4 = [a4 - b4*m4.atan(c4*(X[i]-d4)) for i in range(len(X))]
        f4 = fit_error(m4, y4, Y)
        m4.Obj(f4)
        m4.Equation(a4 - b4*m4.atan(c4*(X[0]-d4)) <= 10)
        m4.Equation(a4 - b4*m4.atan(c4*(X[-1]-d4)) >= -10)
        m4.solve(disp=False)

        # =======================
        # RESULTATS
        # =======================
        results = {
            'T1': {'fobj': f1.value[0], 'a': a1.value[0], 'b': b1.value[0], 'c': c1.value[0]},
            'T2': {'fobj': f2.value[0], 'a': a2.value[0], 'b': b2.value[0], 'c': c2.value[0], 'd': d2.value[0]},
            'T3': {'fobj': f3.value[0], 'a': a3.value[0], 'b': b3.value[0], 'c': c3.value[0], 'd': d3.value[0]},
            'T4': {'fobj': f4.value[0], 'a': a4.value[0], 'b': b4.value[0], 'c': c4.value[0], 'd': d4.value[0]},
        }

        st.markdown("### Resultats numèrics")
        st.json(results)

        # =======================
        # VISUALITZACIÓ
        # =======================
        Y1_fit = [a1.value[0] + b1.value[0]*(x-c1.value[0])**3 for x in X]
        Y2_fit = [a2.value[0] + b2.value[0]*np.exp(-c2.value[0]*(x-d2.value[0])) for x in X]
        Y3_fit = [a3.value[0] - b3.value[0]*np.exp(c3.value[0]*(x-d3.value[0])) for x in X]
        Y4_fit = [a4.value[0] - b4.value[0]*np.arctan(c4.value[0]*(x-d4.value[0])) for x in X]

        plt.figure(figsize=(8,5))
        plt.plot(X, Y, 'ko', label='Dades')
        plt.plot(X, Y1_fit, 'r-', label='T1')
        plt.plot(X, Y2_fit, 'b-', label='T2')
        plt.plot(X, Y3_fit, 'g-', label='T3')
        plt.plot(X, Y4_fit, 'm-', label='T4')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Ajustament de models amb GEKKO')
        plt.legend()
        st.pyplot(plt)

        # =======================
        # AVALUACIÓ DEL MILLOR MODEL
        # =======================
        def sse(y_true, y_pred): return sum((y_true[i]-y_pred[i])**2 for i in range(len(y_true)))
        def rms(y_true, y_pred): return np.sqrt(sse(y_true, y_pred)/len(y_true))

        errors = {
            'T1': {'SSE': sse(Y, Y1_fit), 'RMS': rms(Y, Y1_fit)},
            'T2': {'SSE': sse(Y, Y2_fit), 'RMS': rms(Y, Y2_fit)},
            'T3': {'SSE': sse(Y, Y3_fit), 'RMS': rms(Y, Y3_fit)},
            'T4': {'SSE': sse(Y, Y4_fit), 'RMS': rms(Y, Y4_fit)},
        }
        st.markdown("### Errors de cada model")
        st.json(errors)

        best_sse = min(errors, key=lambda k: errors[k]['SSE'])
        best_rms = min(errors, key=lambda k: errors[k]['RMS'])

        st.success(f"Millor model segons SSE: **{best_sse}**")
        st.success(f"Millor model segons RMS: **{best_rms}**")

    except Exception as e:
        st.error(f"Error: {e}")
