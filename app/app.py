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
    external_stylesheets=[dbc.themes.LUMEN, dbc.icons.FONT_AWESOME],
    external_scripts = ['https://unpkg.com/js-image-zoom/js-image-zoom.js', 'https://cdn.jsdelivr.net/npm/js-image-zoom/js-image-zoom.min.js'],
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
                            width="4",
                            style={
                                'height': '430px'
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
                            width="4",
                            style={
                                'padding': '0'
                            }
                        ),
                    ],
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

    # color_palette = {'#3d3241': ['data:image/png;base64,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', 'data:image/jpeg;base64,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', 'data:image/jpeg;base64,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'], '#bea495': ['data:image/jpeg;base64,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'], '#f7b92c': ['data:image/jpeg;base64,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'], '#0b58a1': ['data:image/png;base64,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']}
    
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
    return html.Img(src=contents, height='100%', id="reference-img"),

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
    log=True
)
def generate_collage(n_clicks, reference_image, color_palette, dash_logger: DashLogger):
    if reference_image == None:
        dash_logger.error("Please choose a reference image")
        return None
    log.info('Generating collage')
    return [html.Img(src=collage(color_palette, reference_image[0]['props']['src']), width='100%', height='100%', id="collage-img", style={'display': 'block', 'width': 'auto'}),]

app.clientside_callback(
    """
    function(dummy, children) {
        var options = {
            height: 430,
        };
        var container = document.getElementById('collage-image-col');
        if (zoom != null) {
            zoom.kill();
        }
        zoom = new ImageZoom(container, options);
        console.log('Zoom added');
        return ['zoom'];
    }
    """,
    Output('placeholder', 'children'),
    Input('collage-image-col', 'children'),
    State('placeholder', 'children'),
    prevent_initial_call=True,
)

##### END STEP 4 #####

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8080, use_reloader=False)
