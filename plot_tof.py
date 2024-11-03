# Requires plotly, dash, etc. use:
# pip install -r requirements_plot.txt
import numpy as np

from dash import Dash, dcc, html
import plotly.express as px
import plotly.graph_objects as go
# https://dash-bootstrap-components.opensource.faculty.ai/docs/components/layout/
import dash_bootstrap_components as dbc

# See
# https://plotly.com/python/imshow/
# Consider xarray:
# https://plotly.com/python/heatmaps/#display-an-xarray-image-with-pximshow
def mk_fig(trs=True, contour=True):
    # can create rgb-format data:
    #img = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
    #                [[0, 255, 0], [0, 0, 255], [255, 0, 0]]
    #               ], dtype=np.uint8)
    #img = np.arange(100)[:,None] + np.arange(100)[None,:]
    img = np.load("correl.npy")
    s0 = img.shape
    N = img.shape[0]*img.shape[1]
    if trs:
        img = np.transpose(img, (1,0,3,2))
    img = img.reshape((N,N))

    # subtract the diagonal
    ctr = np.diag(img)
    ctr = ctr + (ctr <= 0.0)
    scale = ctr**0.5
    img /= scale[:,None]*scale[None,:]

    # remove self-correlations from this plot:
    img -= np.diag(np.diag(img) > 0.5)

    img = np.transpose(img.reshape(s0), (1,0,3,2)).reshape((N,N))
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

def main():
    app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

    app.layout = dbc.Container([
        html.H1("Time-of-flight correlation"),
        dcc.Graph(figure=mk_fig())
        #dbc.Row([dbc.Col([mk_card(i,j)]) for j in range(4)]) \
        #    for i in reversed(range(4))
    ])

    app.run_server(debug=True, use_reloader=False)  # Turn off reloader if inside Jupyter

if __name__=="__main__":
    main()
