from dash import html, register_page, dcc, callback, Input, Output, State, callback
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import plotly.express as px

register_page(
    __name__,
    name='t-SNE',
    top_nav=True,
    path='/t-SNE'
)

def NamedSlider(name, short, min, max, step, val, marks=None):
    if marks:
        step = None
    else:
        marks = {i: i for i in range(min, max + 1, step)}

    return html.Div(
        style={"margin": "25px 5px 30px 0px"},
        children=[
            f"{name}:",
            html.Div(
                style={"margin-left": "5px"},
                children=[
                    dcc.Slider(
                        id=f"slider-{short}",
                        min=min,
                        max=max,
                        step=step,
                        value=val,
                    )
                ],
            ),
        ],
    )

def NamedSlider_100(name, short, min, max, step, val, marks=None):
    if marks is None:
        marks = {i: str(i) for i in range(min, max + 1, 500)}

    return html.Div(
        style={"margin": "25px 5px 30px 0px"},
        children=[
            f"{name}:",
            html.Div(
                style={"margin-left": "5px"},
                children=[
                    dcc.Slider(
                        id=f"slider-{short}",
                        min=min,
                        max=max,
                        marks=marks,
                        step=step,
                        value=val,
                    )
                ],
            ),
        ],
    )

def Card(children, **kwargs):
    return html.Section(children, className="card-style")

layout = html.Div([
  
    dcc.Graph(id='tsne-plot', style={"height": "98vh"}),
    html.Div(
                className="row background",
                style={"padding": "10px"},
                children=[
                    html.Div(
                        className="three columns",
                        children=[
                            Card(
                                [
                                   
                                    NamedSlider(
                                        name="Number Of Iterations",
                                        short="iterations",
                                        min=250,
                                        max=1000,
                                        step=None,
                                        val=500,
                                        marks={
                                            i: str(i) for i in [250, 500, 750, 1000]
                                        },
                                    ),
                                    NamedSlider(
                                        name="Perplexity",
                                        short="perplexity",
                                        min=3,
                                        max=10,
                                        step=None,
                                        val=5,
                                        marks={i: str(i) for i in [3, 10, 30, 50, 100]},
                                    ),
                                   
                                    NamedSlider(
                                        name="Learning Rate",
                                        short="learning-rate",
                                        min=1,
                                        max=10,
                                        step=None,
                                        val=5,
                                        marks={i: str(i) for i in [10, 50, 100, 200]},
                                    ),
                                     NamedSlider_100(
                                        name="Start word index",
                                        short="idx",
                                        min=0,
                                        max=7497,
                                        step=100,
                                        val=500
                                    
                                    ),
                                    ]

                            )]
                    )]

    )

])


@callback(
  
    Output('tsne-plot', 'figure'),  # Update the t-SNE plot figure
    [
        Input("slider-iterations", "value"),
        Input("slider-perplexity", "value"),
        Input("slider-learning-rate", "value"),
        Input("slider-idx", "value"),
    ],

)
def display_3d_scatter_plot(iterations,perplexity,learning_rate,idx):
    

    # Load the CSV file
    embedding_df = pd.read_csv('./data/financial_emb.csv')
    embedding_df.reset_index(drop=True,inplace=True)

    # Extract word embeddings
    words = embedding_df.iloc[int(idx):int(idx)+100, 0]
    embedding_matrix = embedding_df.iloc[int(idx):int(idx)+100, 1:].values

    # Apply t-SNE for dimensionality reduction
    tsne = TSNE(n_components=3, perplexity=perplexity, random_state=0,learning_rate=learning_rate,n_iter=iterations)  # Use n_components=3 for 3D visualization
    word_tsne = tsne.fit_transform(embedding_matrix)

    # Create a DataFrame with t-SNE coordinates and word information
    tsne_df = pd.DataFrame(word_tsne, columns=['Dimension 1', 'Dimension 2', 'Dimension 3'])
    tsne_df['Word'] = list(words)

    # Visualize the results using Plotly
    fig = px.scatter_3d(tsne_df, x='Dimension 1', y='Dimension 2', z='Dimension 3',
                        text='Word', title='3D t-SNE Visualization of Word Embeddings from financial reports',
                        color_discrete_sequence=px.colors.qualitative.Set1)
    fig.update_traces(marker=dict(size=3, opacity=0.7))
    fig.update_layout(hovermode='closest')
    fig.update_traces(textposition='top center')  # Display title on hover


    return fig
