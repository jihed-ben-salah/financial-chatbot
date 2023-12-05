
import dash
from dash import html, dcc
import dash_bootstrap_components as dbc
from navbar import create_navbar

NAVBAR = create_navbar()
FA621 = "https://use.fontawesome.com/releases/v6.2.1/css/all.css"
APP_TITLE = "First Dash App"

app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[
        dbc.themes.LUX,  # Dash Themes CSS
        FA621,  # Font Awesome Icons CSS
    ],
    title=APP_TITLE,
    use_pages=True, 
)

app.index_string = f'''
<!DOCTYPE html>
<html>
    <head>
        {{%metas%}}
        <title>{APP_TITLE}</title>
        {{%favicon%}}
        {{%css%}}
    </head>
    <body>
        {{%app_entry%}}
        <footer>
            {{%config%}}
            {{%scripts%}}
            {{%renderer%}}
        </footer>
        
    </body>
</html>
'''

app.layout = dcc.Loading(  
    id='loading_page_content',
    children=[
        html.Div(
            [
                NAVBAR,
                dash.page_container
            ]
        )
    ],
    color='primary', 
    fullscreen=True  
)

server = app.server

if __name__ == "__main__":
    app.run_server(host="127.0.0.1", port=8088, debug=True)
