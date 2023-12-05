from dash import html, register_page, dcc, callback, Input, Output
import sqlite3
import pandas as pd
from dash.exceptions import PreventUpdate

register_page(
    __name__,
    name='Home',
    top_nav=True,
    path='/'
)

def layout():
    layout = html.Div([
        
        dcc.Interval(
            id='interval-component',
            interval=15 * 1000,  # in milliseconds
            n_intervals=0
        ),

       
        html.Div(id='database-content',style={
                'border': '1px solid #ddd',
                'border-radius': '5px',
                'padding': '20px',
                'margin': '20px auto',
                'width': '80%',
                'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)',
                'background-color': 'white'
            }), 

        html.Div(id='lda-content', style={
            'border': '1px solid #ddd',
            'border-radius': '5px',
            'padding': '20px',
            'margin': '20px auto',
            'width': '80%',  
            'box-shadow': '0 4px 8px 0 rgba(0, 0, 0, 0.2)',
            'background-color': 'white'
        }, children=[
            dcc.Markdown("### Topic Modeling Visualization"),
            html.Iframe(
                id='lda-visualization',
                style={'width': '100%', 'height': '800px'},
                srcDoc=''  # Leave srcDoc empty for now, it will be updated in the callback
            )
        ])
    ])

    
    return layout



@callback(
    [
        Output('database-content', 'children'),
        Output('lda-visualization', 'srcDoc')  # Add this output
    ],
    Input('interval-component', 'n_intervals')
)
def display_database_content(n_int):

    with open('./pyldavis_visualization.html', 'r') as f:
        pyldavis_html = f.read()


    conn = sqlite3.connect('./notebooks/financial_reports.db')
    query = "SELECT date,bank,title FROM pdfs"

    cursor = conn.cursor()

    # Execute a SELECT statement
    cursor.execute(query)

    # Fetch the results
    results = cursor.fetchall()

    # Get the column names from cursor.description
    column_names = [desc[0] for desc in cursor.description]

    # Close the connection
    conn.close()

    table_rows = []

    # Display the results
    for index,row in enumerate(results):
        row_cells = [html.Td(value) for value in row]
     
        table_row = html.Tr(row_cells )
        table_rows.append(table_row)
    


    table_style = {
        'width': '100%',
        'border-collapse': 'collapse',
        'border': '1px solid #ddd',
        'font-size': '14px'
    }

    th_style = {
        'background-color': '#f2f2f2',
        'text-align': 'left',
        'padding': '8px'
    }

    td_style = {
        'padding': '8px'
    }


 



    content = [
        html.H1("Banks' reports"),
        html.Table(
            # Header
            [html.Tr([html.Th(col,style=th_style) for col in column_names])] +
            # Data rows
            table_rows,
            style=table_style
        )
    ]

    

    
    return content,pyldavis_html

    