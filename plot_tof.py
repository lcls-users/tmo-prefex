# Requires plotly, dash, etc. use:
# pip install -r requirements_plot.txt
from typing import Optional
import numpy as np

import typer

from dash import Dash, dcc, html
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# https://dash-bootstrap-components.opensource.faculty.ai/docs/components/layout/
import dash_bootstrap_components as dbc

detectors = [  0,  22,  45, 112,
             180, 337, 135, 157,
              90, 202, 225, 247,
             270,  67, 315, 292, 'fzp' ]

# See https://plotly.com/python/imshow/
# Consider xarray:
# https://plotly.com/python/heatmaps/#display-an-xarray-image-with-pximshow
def mk_fig(trs=True, contour=True):
    # can create rgb-format data:
    #img = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    #                [[0, 255, 0], [0, 0, 255], [255, 0, 0]]
    #               ], dtype=np.uint8)
    #img = np.arange(100)[:,None] + np.arange(100)[None,:]
    img = np.load("correl.npy")
    N = img.shape[0]*img.shape[1]
    img, corr = scale_corr(img)
    if trs: # swap detector and TOF ordering (making TOF outer)
        img = np.transpose(img, (1,0,3,2))
    img = img.reshape((N,N))

    #img = io.imread('https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Crab_Nebula.jpg/240px-Crab_Nebula.jpg')
    fig = px.imshow(img, aspect='equal',
                    color_continuous_scale='gray',
                    #color_continuous_scale='RdBu_r',
                    #binary_string=True, #(speed opt.)
                    origin='lower')
    fig.update_layout(width=1000, height=1000,
                      margin={"t": 0, "b": 0, "r": 10, "l": 10, "pad": 0})
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    if contour:
        fig.add_trace(go.Contour(z=img, showscale=False,
                                 contours=dict(start=0.1, end=1.0, size=3, coloring='lines'),
                                 line_width=1))
    return fig

def scale_corr(img):
    s0 = img.shape
    N  = img.shape[0]*img.shape[1]
    img = img.reshape((N,N))

    # subtract the diagonal
    ctr = np.diag(img).copy()
    scale = (ctr + (ctr<=0.0))**0.5
    img /= scale[:,None]*scale[None,:]

    # remove self-correlations from this plot:
    img -= np.diag(np.diag(img) > 0.5)

    ctr = ctr.reshape(s0[:2])
    return img.reshape(s0), ctr

def mk_fig2(contour=False):
    img = np.load("correl.npy")
    img, corr = scale_corr(img)
    print("Loaded correl. array with shape:")
    print(img.shape)

    img = np.transpose(img, (0,2,1,3))
    # scale ea. detector-detector group by its max
    for u in img:
        for v in u:
            m = v.max()
            if m > 0:
                v /= m
    #img = np.log( img + 1e-8 )
    fig = px.imshow(img, aspect='equal',
                    color_continuous_scale='gray',
                    animation_frame=0,
                    facet_col=1,
                    facet_col_wrap=4,
                    binary_string=True,
                    labels={'facet_col':'detector'},
                    origin='lower')
    # Set facet titles
    for i, det in enumerate(detectors):
        fig.layout.annotations[i]['text'] = str(det)
    fig.update_layout(width=1000, height=1000,
                      margin={"t": 0, "b": 0, "r": 10, "l": 10, "pad": 0})
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
    if contour:
        fig.add_trace(go.Contour(z=img, showscale=False,
                                 contours=dict(start=0.1, end=1.0, size=3, coloring='lines'),
                                 line_width=1))
    return fig

def mk_card(i, j):
    return dbc.Card(
        dbc.CardBody(
            [
                html.H4(f"{i}-{j}"),
                #html.H2("100", id="card-value"),
                #html.P("Description", id="card-description")
                dcc.Graph(figure=mk_fig())
            ]
        )
    )

def plot_diags2():
    fig = make_subplots(
        rows=4, cols=4,
        subplot_titles=["%d"%i for i in detectors])

    for i,d in enumerate(detectors):
        row = i//4
        col = i% 4
        fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6]),
                      row=row, col=col)
    fig.update_layout(height=1000, width=1000,
                      title_text="Diagonals (tof histograms)")
    return fig

def plot_diags():
    img = np.load("correl.npy")
    img, diag = scale_corr(img)

    fig = go.Figure()

    for i,d in enumerate(detectors):
        fig.add_trace(go.Bar(
            x=np.arange(len(diag[i])),
            y=diag[i],
            name=f'{d}',
        ))

    # Here we modify the tickangle of the xaxis, resulting in rotated labels.
    fig.update_layout(barmode='group')#, xaxis_tickangle=-45)
    return fig

def main(debug: Optional[bool] = False,
         reload: Optional[bool] = False):
    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = dbc.Container([
        html.H1("Time-of-flight correlation"),
        dcc.Graph(figure=mk_fig2()),
        #dcc.Graph(figure=plot_diags()),
        #dbc.Row([dbc.Col([mk_card(i,j)]) for j in range(4)]) \
        #    for i in reversed(range(4))
    ])

    # Turn off reloader if inside Jupyter
    app.run_server(debug=debug, use_reloader=reload)

if __name__=="__main__":
    app = typer.Typer(pretty_exceptions_enable=False)
    app.command()(main)
    app()
