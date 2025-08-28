from shiny import ui

def nav_panel(title, *children, **kwargs):
    """Wrapper for ui.nav_panel to centralize future behaviour or styling."""
    return ui.nav_panel(title, *children, **kwargs)

def card_with_header(title, *children, class_="card p-3 mt-3"):
    """Create a card with a header and body."""
    return ui.card(ui.card_header(title), ui.card_body(*children))

def card_body(*children, class_="card p-3"):
    """Return a card body wrapper."""
    return ui.card_body(*children)

def action_button(input_id, label, icon_class=None, class_="btn-primary"):
    """Create an action button optionally with a font-awesome icon."""
    icon = ui.tags.i(class_=icon_class) if icon_class else None
    if icon:
        return ui.input_action_button(input_id, label, icon=icon, class_=class_)
    return ui.input_action_button(input_id, label, class_=class_)

def download_button(input_id, label, icon_class=None, class_="btn-info"):
    """Create a download button optionally with a font-awesome icon."""
    icon = ui.tags.i(class_=icon_class) if icon_class else None
    if icon:
        return ui.download_button(input_id, label, icon=icon, class_=class_)
    return ui.download_button(input_id, label, class_=class_)

def file_input(input_id, label, accept=None):
    """Wrapper for file input control."""
    return ui.input_file(input_id, label, accept=accept)

def get_custom_head():
    """Return common head tags used across the app (styles/scripts)."""
    return ui.tags.head(
        ui.tags.link(
            rel="stylesheet",
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
        ),
        ui.tags.link(rel="stylesheet", href="custom.css"),
        ui.tags.script(src="https://code.jquery.com/jquery-3.6.0.min.js"),
        ui.tags.script(src="custom.js"),
    )
