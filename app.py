from shiny import App
from ui import app_ui
from server import server_function

# Create and run the app
app = App(app_ui, server_function) 