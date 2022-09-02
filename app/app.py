from datetime import date
import requests as r
import plotly.express as px
import logging
import sys
import time
import math
import copy
import binascii
from random import shuffle
from PIL import Image
from io import BytesIO
import cairosvg
import base64
import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Init logging
logging.basicConfig(
    format='[%(asctime)s] [%(name)s:%(lineno)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S %z',
    stream=sys.stdout,
    level=10
)

log = logging.getLogger("PIL")
log.setLevel(logging.INFO)

log = logging.getLogger("urllib3.connectionpool")
log.setLevel(logging.INFO)

log = logging.getLogger("app")
log.setLevel(logging.INFO)

from dash_extensions.enrich import DashProxy, MultiplexerTransform, LogTransform, DashLogger
from dash_extensions.enrich import Input, Output, State, html, dcc, ALL, MATCH
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from dash_extensions import Lottie
import dash

app = DashProxy(
    __name__, 
    title="Collage maker", 
    transforms=[
        MultiplexerTransform(),  # makes it possible to target an output multiple times in callbacks
        LogTransform()  # makes it possible to write log messages to a Dash component
    ],
    external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
    ],
)
server = app.server

app.layout = html.Div(
    [
        dcc.Store(id='colors-dict', storage_type='memory'),
        dcc.Store(id='color-palette', storage_type='memory'),
        dbc.Row(
            [
                dbc.Col(
                    html.H1(["Collage Maker"]),
                    width="auto"
                ),
            ],
            justify="center",
            align="center",
            style={
                'margin-top': '30px',
            }
        ),
        # dbc.Row(
        #     [
        #         dbc.Col(
        #             html.H3(["Step 1: Choose a bunch of photos 📸"]),
        #             width="auto"
        #         ),
        #     ],
        #     justify="center",
        #     align="center",
        #     style={
        #         'margin-top': '20px',
        #     }
        # ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Button(
                        [
                            html.I(className="fas fa-arrow-left me-2"),
                            "Previous step",
                        ],
                        id='prev-step-btn',
                        color="primary",
                        style={
                            'display': 'none'
                        }
                    ),
                    width="2",
                    className="text-start"
                ),
                dbc.Col(
                    dcc.Slider(0, 3,
                        step=None,
                        marks={
                            0: 'Choose a bunch of photos 📸',
                            1: 'Extracting images predominant color 🔍',
                            2: 'Creating the color palette 🎨',
                            3: 'Choose the reference photo 🖼️',
                        },
                        id='step-slider',
                        value=0,
                        disabled=True
                    ),
                    width="8"
                ),
                dbc.Col(
                    [
                        dbc.Button(
                            [
                                "Next step",
                                html.I(className="fas fa-arrow-right ms-2")
                            ],
                            id='next-step-btn',
                            disabled=True,
                            color="primary",
                        ),
                    ],
                    width="2",
                    className="text-end"
                ),
            ],
            justify="center",
            align="center",
            style={
                'margin-top': '30px',
                'margin-bottom': '20px',
            }
        ),
        # STEP 1 DIV
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                dcc.Upload(
                                    id='upload-image',
                                    children=html.Div(
                                        [
                                            'Drag and Drop or ',
                                            html.A('Select the images', style={"cursor": "pointer", 'color': 'var(--bs-primary)'})
                                        ], 
                                        style={
                                            'font-size': '1.25rem',
                                            'width': '100%',
                                            'height': '100px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '5px',
                                            'textAlign': 'center',
                                            'margin': '10px',
                                            'padding-top': '15px'
                                        },
                                    ),
                                    accept="image/*",
                                    # Allow multiple files to be uploaded
                                    multiple=True
                                )
                            ],
                            width="12"
                        ),
                    ],
                    justify="center",
                    align="center",
                    style={
                        'margin-bottom': '20px',
                        'margin-top': '2%'
                    }
                ),
                dbc.Card(
                    [
                        dbc.CardBody(
                            [

                                dbc.Row(
                                    [
                                        dbc.Col(
                                            html.H3(
                                                "Uploaded images", 
                                                className="card-title text-center"
                                            ),
                                            width="6"
                                        ),
                                        dbc.Col(
                                            dbc.Button(
                                                [
                                                    html.I(className="fas fa-times me-2"),
                                                    "Delete all images",
                                                ],
                                                id='delete-images-btn',
                                                color="danger",
                                            ),
                                            width="3",
                                            className="text-end"
                                        )
                                    ],
                                    justify="end",
                                    align="center"
                                ),
                                html.Hr(),
                                dbc.Row(
                                    dbc.Col(
                                        [
                                            html.H4(
                                                [
                                                    "🤷‍♂️ No images uploaded yet"
                                                ], 
                                                className="text-center",
                                                style={
                                                    'margin-top': '1%',
                                                    'margin-bottom': '20px'
                                                }
                                            ),
                                            Lottie(
                                                options=dict(loop=True, autoplay=True), width="25%",
                                                url="https://assets5.lottiefiles.com/private_files/lf30_bn5winlb.json",
                                                isClickToPauseDisabled=True,
                                                style={
                                                    'margin-bottom': '3%',
                                                    'cursor': 'default'
                                                }
                                            )
                                        ]
                                    ),
                                    id='no-images-msg'
                                ),
                                dbc.Row(
                                    id='output-image-upload',
                                    align="end",
                                    justify="start"
                                ),
                            ],
                            style={
                                'min-height': '40vh',
                            }
                        ),
                    ],
                    style={
                        'margin-bottom': '100px'
                    }
                ),
            ],
            id="step-1-div"
        ),
        # STEP 2 AND 3 DIV
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.Tooltip(
                                                                "Change the seed to get different results.",
                                                                target="seed-label",
                                                                placement='top',
                                                            ),
                                                            dbc.InputGroupText(
                                                                [
                                                                    "Seed: ", 
                                                                ],
                                                                id="seed-label",
                                                            ),
                                                            dbc.Input(
                                                                id="seed",
                                                                type="number",
                                                                min=0,
                                                                step=1,
                                                                value=0,
                                                                required=True,
                                                                debounce=False,
                                                                placeholder="Seed (default: 0)",
                                                                style={
                                                                    "max-width": "80px"
                                                                }
                                                            )
                                                        ],
                                                    ),
                                                    width="auto"
                                                ),
                                                dbc.Col(
                                                    dbc.InputGroup(
                                                        [
                                                            dbc.Tooltip(
                                                                "Number of clusters of different colors to try to generate.",
                                                                target="n-colors-label",
                                                                placement='top',
                                                            ),
                                                            dbc.InputGroupText(
                                                                [
                                                                    "Number of Colors: ", 
                                                                ],
                                                                id="n-colors-label",
                                                            ),
                                                            dbc.Input(
                                                                id="n-colors",
                                                                type="number",
                                                                min=1,
                                                                step=1,
                                                                value=4,
                                                                required=True,
                                                                debounce=False,
                                                                placeholder="Number of colors to generate (default: 4)",
                                                                style={
                                                                    "max-width": "80px"
                                                                }
                                                            )
                                                        ],
                                                    ),
                                                    id="n-colors-col",
                                                    width="auto"
                                                ),
                                                dbc.Col(
                                                    dbc.Button(
                                                        [
                                                            "Run again",
                                                            html.I(className="fas fa-play ms-2")
                                                        ],
                                                        id='re-run-btn',
                                                        color="primary",
                                                    ),
                                                    width="auto"
                                                )
                                            ],
                                            justify="center",
                                            align="center",
                                        )
                                    ]
                                )
                            ]
                        )
                    ],
                    style={
                        'margin-top': '25px'
                    }
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            dcc.Loading(
                                [
                                    html.Div(
                                        dcc.Graph(
                                            id="main-colors-chart",
                                            style={
                                                'height': '60vh'
                                            }
                                        ),
                                        id='main-colors-chart-container'
                                    )
                                ],
                                id="loading-main-colors-chart",
                                type="default",
                            ),
                            width="6"
                        ),
                        dbc.Col(
                            dbc.Card(
                                [
                                    dbc.CardBody(
                                        [
                                            dbc.Row(
                                                [
                                                    dbc.Col(
                                                        html.H3(
                                                            "No points clicked", 
                                                            className="card-title text-center", 
                                                            id='related-images-title'
                                                        ),
                                                        width="12"
                                                    ),
                                                ]
                                            ),
                                            html.Hr(),
                                            dbc.Row(
                                                dbc.Col(
                                                    [
                                                        html.H6(
                                                            [
                                                                "Click on a data point in the chart to see the images with that predominant color"
                                                            ], 
                                                            className="text-center",
                                                            style={
                                                                'margin-top': '1%',
                                                            }
                                                        ),
                                                        Lottie(
                                                            options=dict(loop=True, autoplay=True), width="25%",
                                                            url="https://assets7.lottiefiles.com/packages/lf20_K0lHJ8.json",
                                                            isClickToPauseDisabled=True,
                                                            style={
                                                                'margin-bottom': '3%',
                                                                'cursor': 'default'
                                                            }
                                                        )
                                                    ]
                                                ),
                                                id='display-images-col',
                                                justify="center"
                                            ),
                                        ],
                                        style={
                                            'height': '60vh',
                                            'overflow-y': 'auto'
                                        }
                                    )
                                ]
                            ),
                            width="6"
                        )
                    ],
                    align="center",
                    style={
                        'margin-top': '20px'
                    }
                )
            ],
            id="step-2-div",
            style={
                'display': 'none',
            }
        ),
        # STEP 4 DIV
        html.Div(
            [
                dbc.Row(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    dcc.Upload(
                                                        id='upload-reference-image',
                                                        children=html.Div(
                                                            [
                                                                'Drag and Drop or ',
                                                                html.A('Select the reference image', style={"cursor": "pointer", 'color': 'var(--bs-primary)'})
                                                            ], 
                                                            style={
                                                                'font-size': '1.25rem',
                                                                'width': '100%',
                                                                'height': '100px',
                                                                'lineHeight': '60px',
                                                                'borderWidth': '1px',
                                                                'borderStyle': 'dashed',
                                                                'borderRadius': '5px',
                                                                'textAlign': 'center',
                                                                'margin': '10px',
                                                                'padding-top': '15px'
                                                            },
                                                        ),
                                                        accept="image/*",
                                                        # Allow multiple files to be uploaded
                                                        multiple=False
                                                    ),
                                                    width="6"
                                                ),
                                                dbc.Col(
                                                    dbc.Button(
                                                        [
                                                            "Generate collage",
                                                            html.I(className="fas fa-th ms-2")
                                                        ],
                                                        id='generate-collage-btn',
                                                        color="success",
                                                    ),
                                                    width="auto"
                                                )
                                            ],
                                            justify="center",
                                            align="center",
                                        )
                                    ]
                                )
                            ]
                        )
                    ],
                    style={
                        'margin-top': '25px'
                    }
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            id='reference-image-col',
                            width="6",
                            align="center",
                            style={
                                'height': '50vh'
                            }
                        ),
                        # dbc.Col(
                        #     [
                        #         Lottie(
                        #             options=dict(loop=True, autoplay=True), width="30%",
                        #             url="https://assets1.lottiefiles.com/private_files/lf30_dmituz7c.json",
                        #             isClickToPauseDisabled=True,
                        #             style={
                        #                 'margin-top': '6%',
                        #                 'cursor': 'default'
                        #             }
                        #         ),
                        #         html.H4("Creating collage...", id="collage-msg", className="text-center"),
                        #     ],
                        #     id='collage-image-loading-col',
                        #     width="6",
                        #     style={
                        #         'height': '50vh',
                        #         'display': 'none',
                        #     }
                        # ),
                        dbc.Col(
                            id='collage-image-col',
                            width="6",
                            align="center",
                            style={
                                'height': '50vh',
                            }
                        ),
                    ],
                    align="center",
                    style={
                        'margin-top': '20px',
                        'margin-bottom': '50px'
                    }
                )
            ],
            id="step-4-div",
            style={
                'display': 'none',
            }
        ),
        html.Footer(
            html.H6("© Copyright 2012 - Alejandro Marchán", id="copyright", className="text-center"),
            id='footer',
            style={
                'margin-top': 'auto',
                'height': '50px'
            }
        ),
        html.P(
            0,
            id='total-images',
            style={
                'display': 'none'
            }
        ),
        html.P(
            0,
            id='none-display-images',
            style={
                'display': 'none'
            }
        ),
        html.P(
            id='placeholder',
            style={
                'display': 'none'
            }
        )
    ],
    style={
        'width': '90%',
        'margin': 'auto',
        'height': '100vh',
        'display': 'flex',
        'flex-direction': 'column' 
    }
)

def parse_contents(i, contents, filename):
    return dbc.Col(
        [
            # html.H6(filename),
            html.Img(src=contents, width='100%', height='100%', className="uploaded-img"),
            dbc.Button(
                html.I(className="fas fa-times"),
                id={
                    'type': 'remove-image-btn',
                    'index': i
                },
                color="danger", 
                className="me-1",
                style={
                    'position': 'absolute',
                    'top': '-7px',
                    'right': '0'
                },
            ),
        ],
        id={
            'type': 'uploaded-image-col',
            'index': i
        },
        className="uploaded-image-col",
        style={
            'margin-bottom': '10px',
            'position': 'relative',
            'height': '113.08px'
        },
        width="1"
    )

@app.callback(
    Output('copyright', 'children'),
    Input('copyright', 'children'),
)
def set_copyright(dummy):
    return f"© Copyright {date.today().year} - Alejandro Marchán 👨‍💻"

##### START STEP CONTROLLER #####

@app.callback(
    Output('step-slider', 'value'),
    Output('step-1-div', 'style'),
    Output('step-2-div', 'style'),
    Output('step-4-div', 'style'),
    Output('next-step-btn', 'style'),
    Output('next-step-btn', 'disabled'),
    Output('prev-step-btn', 'style'),
    Output('prev-step-btn', 'disabled'),
    Output('n-colors-col', 'style'),
    Input('next-step-btn', 'n_clicks'),
    Input('prev-step-btn', 'n_clicks'),
    State('step-slider', 'value'),
    prevent_initial_call=True
)
def next_step(next_click, prev_click, step):
    # print('CALL NEXT STEP')
    next_step = step
    if dash.callback_context.triggered[0]['prop_id'] == 'next-step-btn.n_clicks':
        next_step = step + 1 if step < 3 else 3
        if next_click == None:
            return dash.no_update
    
    if dash.callback_context.triggered[0]['prop_id'] == 'prev-step-btn.n_clicks':
        next_step = step - 1 if step != 0 else 0   
        if prev_click == None:
            return dash.no_update

    log.info(f'Moving to step {next_step}')
    if next_step == 0:
        # step-slider.value, step-1-div.style, step-2-div.style, next-step-btn.style, next-step-btn.disabled, prev-step-btn.style, prev-step-btn.disabled
        return next_step, None, {'display': 'none'}, {'display': 'none'}, None, False, {'display': 'none'}, False, None
    elif next_step == 1:
        return next_step, {'display': 'none'}, None, {'display': 'none'}, None, False, None, False, {'display': 'none'}
    elif next_step == 2:
        return next_step, {'display': 'none'}, None, {'display': 'none'}, None, False, None, False, None
    elif next_step == 3:
        return next_step, {'display': 'none'}, {'display': 'none'}, None, {'display': 'none'}, False, None, False, None
    return next_step, {'display': 'none'}, {'display': 'none'}, {'display': 'none'}, None, False, None, False, None

##### END STEP CONTROLLER #####

##### START STEP 1 #####

@app.callback(
    Output('output-image-upload', 'children'),
    Output('no-images-msg', 'style'),
    Output('next-step-btn', 'disabled'),
    Output('upload-image', 'contents'),
    Output('total-images', 'children'),
    Output('none-display-images', 'children'),
    Input('upload-image', 'contents'),
    State('upload-image', 'filename'),
    State('output-image-upload', 'children'),
    State('none-display-images', 'children'),
    prevent_initial_call=True
)
def set_images(list_of_contents, list_of_names, prev_children, n_hidden_images):
    # print('CALL SET IMAGES')
    counter = 0 if prev_children == None or len(prev_children) == 0 else prev_children[-1]['props']['id']['index'] + 1
    children = []
    for c, n in zip(list_of_contents, list_of_names):
        children.append(parse_contents(counter, c, n))
        counter += 1
    return children if prev_children == None else prev_children + children, {'display': 'none'}, False, [], counter, n_hidden_images - 1

# @app.callback(
#     Output('output-image-upload', 'children'),
#     Input({'type': 'remove-image-btn', 'index': ALL}, 'n_clicks'),
#     State('output-image-upload', 'children'),
#     prevent_initial_call=True
# )
# def remove_image(n_clicks, children):
#     print(n_clicks)
#     index = None
#     for i, x in enumerate(n_clicks):
#         if x != None:
#             index = i
#             break
#     print(index)
#     if index != None:
#         del children[index]
#     return children

@app.callback(
    Output('no-images-msg', 'style'),
    Output('next-step-btn', 'disabled'),
    Input('none-display-images', 'children'),
    State('total-images', 'children'),
    prevent_initial_call=True
)
def show_no_image_msg(n_hidden_images, n_images):
    # print('CALL NO IMAGE MSG')
    if n_hidden_images != n_images:
        return {'display': 'none'}, False
    else:
        return None, True

@app.callback(
    Output('none-display-images', 'children'),
    Input({'type': 'remove-image-btn', 'index': ALL}, 'n_clicks'),
    State('none-display-images', 'children'),
    prevent_initial_call=True
)
def update_none_display_images(n_clicks, n_images):
    return n_images + 1

@app.callback(
    Output({'type': 'uploaded-image-col', 'index': MATCH}, 'style'),
    Input({'type': 'remove-image-btn', 'index': MATCH}, 'n_clicks'),
    prevent_initial_call=True
)
def remove_image(n_clicks):
    return {'display': 'none'}

@app.callback(
    Output('output-image-upload', 'children'),
    Output('total-images', 'children'),
    Output('none-display-images', 'children'),
    Input('delete-images-btn', 'n_clicks'),
    prevent_initial_call=True
)
def remove_all_images(n_clicks):
    return [], 0, -1

##### END STEP 1 #####

##### START STEP 2 AND 3 #####

def extract_dominant_color(base64_image, n_clusters, seed):
    image_type, image = base64_image.split(',')
    try:
        if 'svg' in image_type:
            svg_im = base64.b64decode(image)
            out = BytesIO()
            cairosvg.svg2png(bytestring=svg_im, write_to=out, output_width=150, output_height=150)
            im = Image.open(out)
        else:
            im = Image.open(BytesIO(base64.b64decode(image)))
    except Exception as e:
        log.error('An image could not be parsed')
        return [0, 0, 0], '000000'
    
    im = im.resize((150, 150))      # optional, to reduce time
    ar = np.asarray(im)
    shape = ar.shape
    if len(shape) < 3:
        return [255, 255, 255], 'FFFFFF'

    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

    codes, dist = scipy.cluster.vq.kmeans(ar, 5, seed=seed)

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = np.histogram(vecs, len(codes))    # count occurrences

    index_max = np.argmax(counts)                    # find most frequent
    peak = codes[index_max]
    colour = binascii.hexlify(bytearray(int(c) for c in peak[:3])).decode('ascii')

    return list(peak), colour

def ms(x, y, z, radius, resolution=20):
    """Return the coordinates for plotting a sphere centered at (x,y,z)"""
    u, v = np.mgrid[0:2*np.pi:resolution*2j, 0:np.pi:resolution*1j]
    X = radius * np.cos(u)*np.sin(v) + x
    Y = radius * np.sin(u)*np.sin(v) + y
    Z = radius * np.cos(v) + z
    return (X, Y, Z)

def plot_graph(df_list, marker_size=5, spheres=[], sphere_colors=[]):
    df = pd.DataFrame(df_list)

    fig = px.scatter_3d(df, 
        x='red', y='green', z='blue',
        color="color",
        color_discrete_sequence=[x['color'] for x in df_list], 
        hover_data=['color']
    )
    fig.update_traces(marker_size = marker_size)
    # Creating the sphere
    i = 0
    for sphere, sphere_color in zip(spheres, sphere_colors):
        (x_pns_surface, y_pns_surface, z_pns_suraface) = ms(*sphere)
        colorscale = [[0, sphere_color], [1, sphere_color]]
        fig.add_traces(go.Surface(name=f'Cluster {i}',
                                  x=x_pns_surface, 
                                  y=y_pns_surface, 
                                  z=z_pns_suraface, 
                                  opacity=0.3, 
                                  colorscale=colorscale, 
                                  showscale=False,
                                  hoverinfo=['name']
                        )
        )
        i += 1

    # fig.update_layout(
    #     margin=dict(l=20, r=30, t=20, b=20),
    # )
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=-0.25),
        eye=dict(x=1.25, y=1.25, z=1.25)
    )

    fig.update_layout(scene_camera=camera)

    return fig

def generate_colors_dict(images, n_clusters, seed):

    base64_images = [im['props']['children'][0]['props']['src'] for im in images if im['props']['style'] != {'display': 'none'}]

    colors_dict = {}
    for base64_image in base64_images:
        values, hex = extract_dominant_color(base64_image, n_clusters, seed)
        if hex not in colors_dict:
            colors_dict[hex] = {
                'color': '#' + hex,
                'images': [base64_image],
                'rgb': (values[0], values[1], values[2]),
                'red': values[0],
                'green': values[1],
                'blue': values[2]
            }
        else:
            colors_dict[hex]['images'].append(base64_image)

    return colors_dict

def generate_color_palette(colors_dict, n_clusters, seed):

    colors = list(colors_dict.keys())

    hex_colors = [elem['rgb'] for elem in colors_dict.values()]

    data = np.array(hex_colors).reshape(np.product(len(colors)), 3).astype(float)

    centroids, radius = scipy.cluster.vq.kmeans(data, n_clusters, seed=seed)

    clusters, distances = scipy.cluster.vq.vq(data, centroids)         # assign codes

    counts, bins = np.histogram(clusters, len(centroids))    # count occurrences

    cluster_max_distances = {}
    color_palette = {}

    for color, cluster, distance in zip(colors, clusters, distances):
        if cluster not in cluster_max_distances:
            cluster_max_distances[cluster] = {
                'max': distance,
                'min': distance,
                'color': '#' + color
            }
        else:
            cluster_max_distances[cluster]['max'] = max(cluster_max_distances[cluster]['max'], distance)
            if distance < cluster_max_distances[cluster]['min']:
                cluster_max_distances[cluster]['min'] = distance
                cluster_max_distances[cluster]['color'] = '#' + color

        if cluster not in color_palette:
            color_palette[cluster] = copy.deepcopy(colors_dict[color]['images'])
        else:
            color_palette[cluster] += colors_dict[color]['images']

    log.info(f'{len(list(centroids))} clusters generated')
    spheres = []
    sphere_colors = []
    for i, centroid in enumerate(centroids):
        sphere = centroid.tolist()
        sphere.append(cluster_max_distances[i]['max'])
        spheres.append(sphere)
        sphere_colors.append(cluster_max_distances[i]['color'])
        images = color_palette[i]
        if len(images) > 0:
            color_palette[cluster_max_distances[i]['color']] = images
        del color_palette[i]
    
    return color_palette, spheres, sphere_colors

@app.callback(
    # Output('main-colors-chart', 'figure'),
    Output('main-colors-chart-container', 'children'),
    Output('colors-dict', 'data'),
    Output('color-palette', 'data'),
    Input('step-slider', 'value'),
    Input('re-run-btn', 'n_clicks'),
    State('output-image-upload', 'children'),
    State('colors-dict', 'data'),
    State('color-palette', 'data'),
    State('n-colors', 'value'),
    State('seed', 'value'),
    prevent_initial_call=True
)
def build_graph(step, run, images, colors_dict, colors_palette, n_clusters, seed):
    # print('CALL EXTRACT COLOR')
    if step == 0:
        return dcc.Graph(
                id="main-colors-chart",
                style={
                    'height': '60vh'
                }
            ), None, None
            
    if step != 1 and step != 2:
        return dcc.Graph(
                id="main-colors-chart",
                style={
                    'height': '60vh'
                }
            ), colors_dict, colors_palette

    msg = 'to extract the images main colors'

    tic = time.process_time()
    
    color_palette = {}

    if step == 1:
        colors_dict = generate_colors_dict(images, n_clusters, seed) if colors_dict == None or dash.callback_context.triggered[0]['prop_id'] == 're-run-btn.n_clicks' else colors_dict

        df_list = list(colors_dict.values())

        fig = plot_graph(df_list)
    else:
        df_list = list(colors_dict.values())

        if n_clusters > len(colors_dict):
            n_clusters = len(colors_dict)

        color_palette, spheres, sphere_colors = generate_color_palette(colors_dict, n_clusters, seed)

        fig = plot_graph(df_list, spheres=spheres, sphere_colors=sphere_colors)
        
        msg = 'to generate the palette'

    tac = time.process_time()
    log.info(f'Took {tac - tic} seconds {msg}')

    return dcc.Graph(
        figure=fig,
        id="main-colors-chart",
        style={
            'height': '60vh'
        }
    ), colors_dict, color_palette

@app.callback(
    Output('display-images-col', 'children'),
    Output('related-images-title', 'children'),
    Input('main-colors-chart', 'clickData'),
    Input('next-step-btn', 'n_clicks'),
    Input('prev-step-btn', 'n_clicks'),
    State('colors-dict', 'data'),
    State('color-palette', 'data'),
    State('main-colors-chart', 'figure'),
    prevent_initial_call=True
)
def clicked_point(point_info, next_click, prev_click, colors_dict, color_palette, fig):
    if point_info == None:
        return dash.no_update

    if dash.callback_context.triggered[0]['prop_id'] == 'next-step-btn.n_clicks' or dash.callback_context.triggered[0]['prop_id'] == 'prev-step-btn.n_clicks':
        return [
            html.H6(
                [
                    "Click on a data point in the chart to see the images with that predominant color"
                ], 
                className="text-center",
                style={
                    'margin-top': '1%',
                }
            ),
            Lottie(
                options=dict(loop=True, autoplay=True), width="25%",
                url="https://assets7.lottiefiles.com/packages/lf20_K0lHJ8.json",
                isClickToPauseDisabled=True,
                style={
                    'margin-bottom': '3%',
                    'cursor': 'default'
                }
            )
        ], ["No points clicked"]

    if 'customdata' in point_info['points'][0]:
        color = point_info['points'][0]['customdata'][0][1:]
        r, g, b = colors_dict[color]['rgb']

        text_color = get_text_color(r, g, b)

        return [dbc.Col(html.Img(src=image, width='100%', height='180px', className="uploaded-img"), width="3") for image in colors_dict[color]['images']], \
            ["Images with the predominant color: ", dbc.Badge(color, color="#" + color, text_color=text_color, className="ms-1")]
    elif 'curveNumber' in point_info['points'][0]: 
        curveNumber = point_info['points'][0]['curveNumber']
        cluster_name = fig['data'][curveNumber]['name']
        color = fig['data'][curveNumber]['colorscale'][0][1]

        r, g, b = list(binascii.unhexlify(color[1:]))

        text_color = get_text_color(r, g, b)

        return [dbc.Col(html.Img(src=image, width='100%', height='109px', className="uploaded-img", style={'margin-bottom': '10px'}), width="2") for image in color_palette[color]], \
            ["Images contained in: ", dbc.Badge(cluster_name, color=color, text_color=text_color, className="ms-1")]
    return [], []

def get_text_color(r, g, b):
    # HSP (Highly Sensitive Poo) equation from http://alienryderflex.com/hsp.html
    hsp = math.sqrt(
        0.299 * (r * r) +
        0.587 * (g * g) +
        0.114 * (b * b)
    )

    if hsp>127.5:
        return 'black'
    
    return 'white'

##### END STEP 2 AND 3 #####

##### END STEP 4 #####

def collage(color_palette, base64_image):
    ima_size = 20

    final_width = 2100
    final_height = 2000

    image_type, image = base64_image.split(',')

    if 'svg' in image_type:
        svg_im = base64.b64decode(image)
        out = BytesIO()
        cairosvg.svg2png(bytestring=svg_im, write_to=out, output_width=final_width//ima_size, output_height=final_height//ima_size)
        template = Image.open(out)
    else:
        template = Image.open(BytesIO(base64.b64decode(image)))
        template = template.resize((final_width//ima_size, final_height//ima_size))

    new_im = Image.new('RGB', (final_width, final_height))

    blue_idx = 0
    yellow_idx = 0

    cluster_hex_colors = list(color_palette.keys())
    cluster_colors = [list(binascii.unhexlify(color[1:])) for color in color_palette.keys()]
    cluster_last_image_idx = [0 for images in color_palette.values()]

    for row in range(template.size[0]):
        for col in range(template.size[1]):
            rgb = template.getpixel((row, col))[:3]
            # Search closest cluster
            closest_idx = 0
            closest_distance = 9999999999999999
            for i, color in enumerate(cluster_colors):
                dist = math.sqrt((rgb[0]-color[0])**2 + (rgb[1]-color[1])**2 + (rgb[2]-color[2])**2)
                if dist < closest_distance:
                    closest_idx = i
                    closest_distance = dist

            color = cluster_hex_colors[closest_idx]
            n_images = len(color_palette[color])
            last_image_idx = cluster_last_image_idx[closest_idx]

            if last_image_idx == n_images - 1:
                shuffle(color_palette[color])
                base64_image = color_palette[color][0]
            else:
                base64_image = color_palette[color][0]
                cluster_last_image_idx[closest_idx] += 1

            image_type, image = base64_image.split(',')

            if 'svg' in image_type:
                svg_im = base64.b64decode(image)
                out = BytesIO()
                cairosvg.svg2png(bytestring=svg_im, write_to=out, output_width=final_width//ima_size, output_height=final_height//ima_size)
                im = Image.open(out)
            else:
                im = Image.open(BytesIO(base64.b64decode(image)))
            
            im = im.resize((ima_size, ima_size))
            new_im.paste(im, (row * ima_size, col * ima_size))

    log.info('Collage generated')

    return new_im

@app.callback(
    Output("reference-image-col", "children"), 
    Input('upload-reference-image', 'contents'),
    prevent_initial_call=True,
)
def show_reference_photo(contents):
    return html.Img(src=contents, width='100%', height='100%', className="reference-img"),

# @app.callback(
#     Output("placeholder", "children"), 
#     Output("collage-image-loading-col", "style"), 
#     Input("generate-collage-btn", "n_clicks"),
#     State("reference-image-col", "children"), 
#     prevent_initial_call=True, 
#     log=True
# )
# def generate_collage(n_clicks, reference_image, dash_logger: DashLogger):
#     if reference_image == None:
#         dash_logger.error("Please, select a reference image first")
#         return None, {'display': 'none'}
#     log.info('Showing loading message')
#     dash_logger.info("Generating the collage")
#     return ["fire"], { 'margin-top': '20px', 'margin-bottom': '50px' }

@app.callback(
    Output("collage-image-col", "children"), 
    Input("generate-collage-btn", "n_clicks"),
    State("reference-image-col", "children"), 
    State('color-palette', 'data'),
    prevent_initial_call=True,
)
def generate_collage(children, reference_image, color_palette):
    if children == None:
        return None
    log.info('Generating collage')
    return [html.Img(src=collage(color_palette, reference_image[0]['props']['src']), width='100%', height='100%', className="collage-img", style={'display': 'block', 'width': 'auto'}),]

##### END STEP 4 #####

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8080, use_reloader=False)
