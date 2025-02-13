import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas as pn
import streamlit as st
import xarray as xr

# from IPython.display import HTML, display

# -*- coding: utf-8 -*-
nan = np.nan
dx = 100.0
dy = 100.0
dt = 1.0  # % 60 mins
nsteps = 5
nx = 3
ny = 3
assert nx == ny
# % The grid
xBorder = np.arange(-0.5 * dx, dx * (nx + 0.5), dx)
yBorder = np.arange(-0.5 * dy, dy * (ny + 0.5), dy)
x = xBorder[:-1] + np.diff(xBorder) / 2.0
y = yBorder[:-1] + np.diff(yBorder) / 2.0

# u = np.flipud([[2.5,3.5,4.3,5],[5,7,8.6,10.],[7.5,10.5,13.0,15.0],[10,14,17.3,20]])
# v = np.flipud([[4.3,3.5,2.5,0.5],[8.6,7,5,0.5],[13,10.5,7.5,0.5],[17.3,14.0,10,0.5]])*(-1)
Tinit = np.ones((nx, ny)) * np.array([5, 6, 7, 8])[:nx]
Tinit = pn.DataFrame(
    Tinit, index=[f"{i} km" for i in y], columns=[f"{i} km" for i in x]
)


u = (
    np.array(
        [
            [2.5, 3.5, 4.3, 5],
            [5, 7, 8.6, 10.0],
            [7.5, 10.5, 13.0, 15.0],
            [10, 14, 17.3, 20],
        ]
    )
    * 3.6
)
v = (
    np.array(
        [
            [4.3, 3.5, 2.5, 0.5],
            [8.6, 7, 5, 0.5],
            [13, 10.5, 7.5, 0.5],
            [17.3, 14.0, 10, 0.5],
        ]
    )
    * (-1)
    * 3.6
)
u = np.around(u[:nx, :ny], 0)
v = np.around(v[:nx, :ny], 0)

u = pn.DataFrame(u, index=[f"{i} km" for i in y], columns=[f"{i} km" for i in x])
v = pn.DataFrame(v, index=[f"{i} km" for i in y], columns=[f"{i} km" for i in x])

xderiv = np.zeros((nx, ny))
yderiv = np.zeros((nx, ny))

# T = xr.DataArray(np.zeros((nsteps,nx,ny)) * np.nan, coords={"time":range(nsteps), "x":x, "y":y}, dims=["time", "x", "y"])
# Tref = xr.DataArray(np.zeros((nsteps,nx,ny)) * np.nan, coords={"time":range(nsteps), "x":x, "y":y}, dims=["time", "x", "y"])

T = []
Tref = []
T.append(Tinit)
Tref.append(Tinit)
for nn in range(1, nsteps):
    T.append(Tinit * np.nan)
    Tref.append(Tinit * np.nan)

time = ["8:00", "9:00", "10:00", "11:00", "12:00", "13:00"]

xbound = np.array([[-3, -3, -3, -3], [-2, -2, -2, -2], [-1, -1, -1, -1], [0, 0, 0, 0]])[
    :, :nx
]
ybound = np.array([[4, 5, 6, 8], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 10]])[:, :ny]


def plotWind(u, v, split=False):
    fig = plt.figure(figsize=(4, 3))
    sp = fig.add_subplot(111)
    Q = sp.quiver(x, y, u / 3.6, v / 3.6, units="dots", scale=0.2)
    # qk = sp.quiverkey(Q, 0.9, 0.95, 2, r'$2 \frac{m}{s}$',
    #           labelpos='E',
    #           coordinates='figure',
    #           fontproperties={'weight': 'bold'})
    foo = sp.set_ylim(ny * dy, -dy / 2.0)
    foo = sp.set_xlim(-dx / 2.0, nx * dx - dx / 2.0)
    foo = sp.set_xticks(x)
    foo = sp.set_yticks(y)
    foo = sp.set_title("Windfeld [km/h]")
    sp.set_xlabel("Ost-West (km)")
    sp.set_ylabel("Nord-Süd (km)")
    if split:
        Q2 = sp.quiver(
            x, y, u / 3.6, np.zeros_like(v), color="red", units="dots", scale=0.2
        )
        Q3 = sp.quiver(
            x, y, np.zeros_like(u), v / 3.6, color="blue", units="dots", scale=0.2
        )

    return fig


def plotTemp(
    T1,
    T2=None,
    title1="Temperaturfeld (°C)",
    title2="Temperaturfeld (°C)",
    plotDiff=True,
):
    T1 = T1.values
    try:
        T2 = T2.values
    except:
        pass
    fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(8, 6))
    sp = ax[0, 0]
    cm = sp.pcolormesh(xBorder, yBorder, np.ma.masked_invalid(T1), vmin=0, vmax=10)
    cb = fig.colorbar(cm)
    cb.set_label("Temperatur (°C)")
    foo = sp.set_ylim(ny * dy - dy / 2.0, -dy / 2.0)
    foo = sp.set_xlim(-dx / 2.0, nx * dx - dx / 2.0)
    foo = sp.set_xticks(x)
    foo = sp.set_yticks(y)
    foo = sp.set_title(title1)
    sp.set_xlabel("Ost-West (km)")
    sp.set_ylabel("Nord-Süd (km)")
    if T2 is not None:
        sp = ax[0, 1]
        cm = sp.pcolormesh(xBorder, yBorder, np.ma.masked_invalid(T2), vmin=0, vmax=10)
        cb = fig.colorbar(cm)
        cb.set_label("Temperatur (°C)")
        foo = sp.set_ylim(ny * dy - 50, -50)
        foo = sp.set_xlim(-50, nx * dx - 50)
        foo = sp.set_xticks(x)
        foo = sp.set_yticks(y)
        foo = sp.set_title(title2)
        sp.set_xlabel("Ost-West (km)")
        # sp.set_ylabel(u'Nord-Süd (km)')
        if plotDiff:
            sp = ax[1, 0]
            cm = sp.pcolormesh(
                xBorder,
                yBorder,
                np.ma.masked_invalid(T1 - T2),
                vmin=-5,
                vmax=5,
                cmap="PRGn",
            )
            cb = fig.colorbar(cm)
            cb.set_label("Temperatur (°C)")
            foo = sp.set_ylim(ny * dy - 50, -50)
            foo = sp.set_xlim(-50, nx * dx - 50)
            foo = sp.set_xticks(x)
            foo = sp.set_yticks(y)
            foo = sp.set_title("Differenz")
            sp.set_xlabel("Ost-West (km)")
            # sp.set_ylabel(u'Nord-Süd (km)')
        else:
            fig.delaxes(ax[1, 0])
    else:
        fig.delaxes(ax[0, 1])
        fig.delaxes(ax[1, 0])
    fig.delaxes(ax[1, 1])

    fig.tight_layout()
    return fig


def fieldWithBoundary(A, xb, yb, headline=None):
    Aplus = np.zeros((A.shape[0] + 1, A.shape[1] + 1))
    Aplus[1:, 1:] = A
    Aplus[0, 1:] = yb
    Aplus[1:, 0] = xb

    ind = np.concatenate(([x[0] - np.diff(x)[0]], x))
    col = np.concatenate(([y[0] - np.diff(y)[0]], y))

    # display(pn.DataFrame(Aplus,columns=col,index=ind))

    return pn.DataFrame(np.around(Aplus, 2), columns=col, index=ind)


st.write("# Forecast Factory: Vorbereitung")

st.write("## Das Gitter")
st.image("raster.png")
st.write("#### 👉 Merkt euch, welcher Gitterpunkt  ihr seid!")


st.write("## Stark Vereinfachte Vorhersagegleichung")
# st.image("equation.png")
st.latex(
    "T_{Zukunft} = T_{Jetzt} - \\Big(\\frac{\\Delta t}{\\Delta x} \\cdot \\Big\\{(u \\cdot [T_{Jetzt} - T_{jetzt, Westen}]) + (v \\cdot [T_{Jetzt} - T_{jetzt, Norden}]) \\Big\\}\\Big)"
)


st.write("## Windfeld")
st.pyplot(plotWind(u, v))

st.write(
    "Der Wind $\\vec{V}$ ist eigentlich ein Vektor mit den Komponenten West $u$ and Süd $v$ "
)
st.latex("\\vec{V} = \\begin{bmatrix}u \\\\ v \\end{bmatrix}")
st.pyplot(plotWind(u, v, split=True))


st.write("### Wind von Westen u [km/h]:")

st.dataframe(
    u,
)
st.write("### Wind von Süden v [km/h]:")
st.dataframe(
    v,
)


st.write("#### 👉 Schreibt euch die Windgeschwindigkeit für euren Gitterpunkt auf!")


st.write("## Temperatur um 8:00")
st.pyplot(plotTemp(T[0], title1="Temperatur (T$_{8:00}$, t=0)"))

st.dataframe(T[0])
st.write("#### 👉 Schreibt euch die Temperatur für euren Gitterpunkt auf!")

# for tt in range(nsteps):
#     st.write(tt)
#     st.dataframe(fieldWithBoundary(Tinit.values, xbound[tt], ybound[tt]))


st.write("# Forecast Factory: Rechnung")

for nn in range(1, nsteps):
    st.write(f"### 👉 Berechnet die Temperaturänderung {time[nn-1]} -> {time[nn]} Uhr")

    T[nn] = st.data_editor(T[nn], key=f"time{nn}")
    st.pyplot(
        plotTemp(
            T[nn - 1],
            T[nn],
            title1=f"Temperatur  {time[nn-1]}",
            title2=f"Temperatur {time[nn]}",
        )
    )


st.write("# Forecast Factory: Vergleich mit dem Computer")


st.code(
    """
def futureTemp(T, Twest, Tnord, u, v, dt, dx):
    A = dt / dx
    B = u * (T - Twest)
    C = v * (T - Tnord)
    D = B + C
    return T - (A * D)
        """
)


def futureTemp(T, Twest, Tnord, u, v, dt, dx):
    A = dt / dx
    B = u * (T - Twest)
    C = v * (T - Tnord)
    D = B + C
    return T - (A * D)


for n in range(1, nsteps):  # Time Loop
    for i in range(ny):  # Y Loop
        for j in range(nx):  # x Loop
            if i == 0:  # 1st row
                Tnord = ybound[n - 1, j]
            else:
                Tnord = Tref[n - 1].iloc[i - 1, j]
            if j == 0:  # 1. column
                Twest = xbound[n - 1, i]
            else:
                Twest = Tref[n - 1].iloc[i, j - 1]
            # estimate
            Tref[n].iloc[i, j] = futureTemp(
                Tref[n - 1].iloc[i, j], Twest, Tnord, u.iloc[i, j], v.iloc[i, j], dt, dx
            )
    st.write(f"### Schritt {n} um {time[n]} Uhr")
    st.dataframe(Tref[n])  # print results

    st.pyplot(
        plotTemp(
            T[n],
            Tref[n],
            title1="Schüler (%s)" % time[n],
            title2="Computer (%s)" % time[n],
            plotDiff=False,
        )
    )
