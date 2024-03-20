from dash import html, register_page, dcc, callback, Input, Output, State, callback
import base64
import io
from main import extract_text_from_pdf_2
import sqlite3
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import dash_bootstrap_components as dbc
from main import chunking,get_vectorstore,get_conversation_chain
import torch
from xgboost import XGBClassifier


register_page(
    __name__,
    name='add_pdf',
    top_nav=True,
    path='/add_pdf'
)

def layout():
    layout = html.Div([
        html.H1("Add new report"),
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select a PDF File')
            ]),
            style={
                'width': '40%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'
            },
            multiple=False  # Allow only one file to be uploaded
        ),

        html.Div([
            dcc.Loading(
                id="loading-pdf-content",
                type="default",
                children=html.Div(id='output-pdf-content')),
        ], style={'float': 'left', 'width': '50%'}),  # Display output-pdf-content on the left

        html.Div([
            html.H1("Chat with this PDF"),
            dcc.Input(id="user-question", type="text", placeholder="Ask a question about your documents:", style={'width': '40%'}),
            html.Button("Process", id="process-button"),
            html.Div([
                html.Div(id="conversation-output", className="content"),
            ], className="content-container"),
        ], style={'float': 'right', 'width': '50%'}),  # Display chat on the right
    ])
    return layout


@callback(
    [Output('output-pdf-content', 'children'),
    Output('output-pdf-content', 'data'),
    ],
    [Input('upload-data', 'contents')],
    [State('upload-data', 'filename')
     ]
)
def update_uploaded_pdf(contents,filename):
    if contents is None:
        return '', None
    else:
        content_type, content_string = contents.split(',')
        decoded_content = io.BytesIO(base64.b64decode(content_string))
        name, extention = filename.split('.')
        data_f = extract_text_from_pdf_2(name, decoded_content=decoded_content)
        name, ext = filename.split('.')
        vectorizer =  joblib.load('./notebooks/vectorizer.pkl')
        xgb= joblib.load('./notebooks/XGBClassifier_model.pkl')
        topic=[]
        for index, row in data_f.iterrows():
            new_document = row['text']
            new_document_features = vectorizer.transform([new_document])
            prediction = xgb.predict(new_document_features)
            topic.append(prediction[0])
        data_f['topic']=topic
        data_f.to_csv('./data/'+name + '.csv', index=False)
        data_financial=data_f[data_f['topic']==0]
        aggregated_data = data_financial.groupby('file_name').agg({
            'text': ' '.join,
            'tokens': lambda x: [token for sublist in x for token in sublist],
            'clean_text': ' '.join,
        }).reset_index()

        data_financial = data_financial.drop(['text', 'tokens', 'clean_text'], axis=1)
        data_financial = data_financial.merge(aggregated_data, on='file_name')

        data_financial.drop_duplicates(subset=['file_name', 'text', 'clean_text'], inplace=True)
        data_financial.reset_index(drop=True, inplace=True)

        data_financial['bank']=data_f['bank'][0]
        data_financial['date']=data_f['date'][0]
        data_financial['title']=data_f['title'][0]

        data_financial.to_csv('./data/financial_' + name + '.csv', index=False)
   
        #to change
        conn = sqlite3.connect('./notebooks/financial_reports.db') 
        cursor = conn.cursor()

        for index, row in data_financial.iterrows():
            insert_query = "INSERT INTO pdfs(file_name, date, title, bank, text, clean_text, tokens) VALUES (?,?,?,?,?,?,?)"
            values = (row['file_name'], row['date'], row['title'], row['bank'], str(row['text']), str(row['clean_text']), str(row['tokens']),)
            cursor.execute(insert_query, values)

        conn.commit()
        conn.close()

        
        
        return (
            html.Div([
                html.H5(f'Content of {filename}'),
                html.Pre(data_f["text"])
            ]),
            data_f["text"]
        )

@callback(
Output("conversation-output", "children"),
[Input('output-pdf-content', 'data'),
Input("process-button", "n_clicks")],
State("user-question", "value")
)
def QA(inp_text,n_clicks,user_question):
    if not n_clicks or not user_question:
        return ""
    
    print("##################### user q",user_question)
                
    conversation_output = []
    text = ' '.join(inp_text)
    text_chunks = chunking(text)
    print(text_chunks)
    vectorstore = get_vectorstore(text_chunks[0])
    print('v: ',vectorstore)
    conversation_chain = get_conversation_chain(vectorstore)
    response = conversation_chain({'question': user_question})
    conversation_output = []
    for i, message in enumerate(response['chat_history']):
        if i % 2 == 0:
            conversation_output.append(html.P(f"User: {message.content}"))
        else:
            conversation_output.append(html.P(f"Bot: {message.content}"))
    save_conversation_to_file('conversation_output.txt', [message.content for message in response['chat_history']])
    return conversation_output


def save_conversation_to_file(filename, conversation_messages):
    with open(filename, 'w') as file:
        for message in conversation_messages:
            file.write(message + '\n')
