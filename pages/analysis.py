from dash import html, dcc, register_page,Input, Output, State,callback, dash_table  # Import State here
import dash
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import io
import base64
import matplotlib as mpl
import spacy
import sqlite3
import dash_bootstrap_components as dbc
from spacy import displacy
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

register_page(
    __name__,
    name='analysis',
    top_nav=True,
    path='/analysis'
)


nlp = spacy.load("en_core_web_sm")

NER_ATTRS = ["text",  "start_char", "end_char", "label_",]
conn = sqlite3.connect('./notebooks/financial_reports.db')
query = "SELECT file_name FROM pdfs"
cursor = conn.cursor()
cursor.execute(query)
results = cursor.fetchall()
conn.close()
options_list = [{'label': title[0], 'value': title[0]} for title in results]


table = dash_table.DataTable(
    id="table",
    columns=[{"name": c, "id": c} for c in NER_ATTRS + ["description"]],
    filter_action="native",
    sort_action="native",
    page_size=10,
    style_table={"overflowX": "auto"},
)

hist_data=pd.read_csv('./data/2-gram.csv')  

def layout():
    layout = html.Div([
      html.Div([
            # Histogram plot
            dcc.Graph(id='histogram-graph', style={
                'flex': '1',
                'display': 'flex',
                'align-items': 'center',
                'justify-content': 'center',
                'margin': '5px',
                'background-color': '#FFFFFF',
                'border-radius': '5px',
                'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)',
            }),
            
            # Word cloud output
            html.Div(id='word-cloud-output', style={
                'flex': '1',
                'display': 'flex',
                'align-items': 'center',
                'justify-content': 'center',
                'margin': '5px',
                'background-color': '#FFFFFF',
                'border-radius': '5px',
                'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)',
            }),
        ], style={
            'display': 'flex',
            'align-items': 'stretch',  # Adjust alignment as needed
            'margin': '20px auto',
        }),
       
        
        html.Div(
            style={
                'display': 'flex',
                'justify-content': 'center',  # Align items to the right
                'align-items': 'center',  # Align items vertically
                'gap': '10px',  # Add some spacing between items
                'margin-top': '0px',  # Add top margin for spacing
            },
            children=[
                dcc.Dropdown(
                    id='max-words-dropdown',
                    options=[
                        {'label': '30', 'value': 30},
                        {'label': '100', 'value': 100},
                        {'label': '300', 'value': 300},
                        {'label': '500', 'value': 500},
                        {'label': '1000', 'value': 1000},
                        # Add more options as needed
                    ],
                    value="Select max words",  # Default value
                    placeholder="Select max words",
                    style={
                        'width': '170px',
                        'border': 'none',
                        'color': 'black',
                        'padding': '4px 8px',
                        'text-align': 'center',
                        'text-decoration': 'none',
                        'font-size': '14px',
                        'border-radius': '4px',
                        'cursor': 'pointer'
                    }
                ),
                html.Button(
                    'Generate Word Cloud',
                    id='generate-button',
                    style={
                        'background-color': '#C41E3A',
                        'border': 'none',
                        'color': 'white',
                        'padding': '8px 16px',
                        'text-align': 'center',
                        'text-decoration': 'none',
                        'font-size': '14px',
                        'border-radius': '4px',
                        'cursor': 'pointer'
                    }
                ),
            ]
        ),
        html.Div(style={'height': '40px'}),
        html.H4("Named Entity Recognition"),
        dcc.Dropdown(
                    id='report_name_dd',
                    options=options_list,
                    value=options_list[0]['value'],  # Default value
                    placeholder="Select report name",
                    style={
                        'width': '700px',
                        'border': 'none',
                        'color': 'black',
                        'padding': '4px 8px',
                        'text-align': 'center',
                        'text-decoration': 'none',
                        'font-size': '14px',
                        'border-radius': '4px',
                        'cursor': 'pointer'
                    }
                ),
        dbc.Card(
            dcc.Markdown(id="html", dangerously_allow_html=True),
            body=True,
            className="mb-5",
            style={
           
            "height": "300px"  # Adjust the height as desired
            },
        ),
        html.Div(table),
        html.Div([
        html.Button("Go Back", id="load-less-button",style={
                        'background-color': '#C41E3A',
                        'border': 'none',
                        'color': 'white',
                        'padding': '8px 16px',
                        'text-align': 'center',
                        'text-decoration': 'none',
                        'font-size': '14px',
                        'border-radius': '4px',
                        'cursor': 'pointer'
                    }),
        html.Button("Load More", id="load-more-button",style={
                        'background-color': '#C41E3A',
                        'border': 'none',
                        'color': 'white',
                        'padding': '8px 16px',
                        'text-align': 'center',
                        'text-decoration': 'none',
                        'font-size': '14px',
                        'border-radius': '4px',
                        'cursor': 'pointer'
                    })
        
            ], className="mb-3"),

            

    ], style={'display': 'flex', 'flex-direction': 'column'})  # Set the main layout container to flex column
    return layout


@callback(
    [Output('word-cloud-output', 'children'),
     Output("html", "children"),
     Output("table", "data"),
    ],
    [Input('generate-button', 'n_clicks'),
     Input("load-more-button", "n_clicks"),
     Input("load-less-button", "n_clicks"),
     Input("report_name_dd", "value")
     ],
    State('max-words-dropdown', 'value')  # Add this line
)
def update_word_cloud(n_clicks, load_more_clicks,load_less_clicks,report_name,max_words):  # Add max_words parameter
    if n_clicks is None:
        if load_more_clicks is None:
            load_more_clicks=0
        if load_less_clicks is None:
            load_less_clicks=0
  
        with open('./extracted_data/'+report_name+'.txt', 'r',encoding='utf-8') as file:
            file_contents = file.read()

        
        start_index=load_more_clicks-load_less_clicks
        
        #text = file_contents[start_index*1000:(start_index+1)*1000]

        words = file_contents.split()

        start_word_index = start_index * 140
        end_word_index = (start_index + 1) * 140
        selected_words = words[start_word_index:end_word_index]

        # Join the selected words back into a string
        selected_text = ' '.join(selected_words)
        text = selected_text.replace("\n", " ")

        doc = nlp(text)
        html = displacy.render(doc, style="ent", minify=True)

        table_data = [
            [str(getattr(ent, attr)) for attr in NER_ATTRS]
            for ent in doc.ents
            # if ent.label_ in label_select
        ]
        if table_data:
            dff = pd.DataFrame(table_data, columns=NER_ATTRS)
            dff["description"] = dff["label_"].apply(lambda x: spacy.explain(x))
        return generate_wordcloud(100),html, dff.to_dict("records")

    else:
        if load_more_clicks is None:
            load_more_clicks=0
        if load_less_clicks is None:
            load_less_clicks=0

        with open('./extracted_data/'+report_name+'.txt', 'r',encoding='utf-8') as file:
            file_contents = file.read()

        
        start_index=load_more_clicks-load_less_clicks
        
        text = file_contents[start_index*1000:(start_index+1)*1000]
        text = text.replace("\n", " ")

        doc = nlp(text)
        html = displacy.render(doc, style="ent", minify=True)

        table_data = [
            [str(getattr(ent, attr)) for attr in NER_ATTRS]
            for ent in doc.ents
            # if ent.label_ in label_select
        ]
        if table_data:
            dff = pd.DataFrame(table_data, columns=NER_ATTRS)
            dff["description"] = dff["label_"].apply(lambda x: spacy.explain(x))
        
            return generate_wordcloud(max_words),html, dff.to_dict("records")


@callback(
    Output('histogram-graph', 'figure'),
    [Input('generate-button', 'n_clicks')
    ]
)
def update_histogram(n_clicks):
    # Add your logic to update the histogram based on user interactions.
    # You can use the hist_data DataFrame to create the histogram.

    # Example code for updating the histogram:
    filtered_data = hist_data  # Add your filtering logic here
    
    fig = px.histogram(filtered_data, x='frequency', y='ngram')
    
    return fig

def generate_wordcloud(max_words):
    with open('C:/Users/jihed/OneDrive/Bureau/Financial reports analysis/financial_reports_analysis/pages/analysis.py', 'r',encoding='utf-8') as file:
            corpus=file.read()
    mpl.rcParams['font.size']=18              
    mpl.rcParams['savefig.dpi']=50             
    mpl.rcParams['figure.subplot.bottom']=.1 
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
                                background_color='white',
                                stopwords=stopwords,
                                max_words=max_words,
                                max_font_size=70, 
                                random_state=42
                                ).generate(str(corpus))

    plt.figure(figsize=(15, 8))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    plt.close()

    img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")


    
     
    return html.Img(src=f'data:image/png;base64,{img_base64}')




