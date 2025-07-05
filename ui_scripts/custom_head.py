from shiny import ui

custom_head = ui.tags.head(
    ui.tags.link(
        rel="stylesheet",
        href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    ),
    ui.tags.link(
        rel="stylesheet",
        href="custom.css"
    ),
    ui.tags.script(
        src="https://code.jquery.com/jquery-3.6.0.min.js"
    ),
    ui.tags.script(
        src="custom.js"
    )
)
