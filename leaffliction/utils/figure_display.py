"""Show matplotlib figures without crashing on a headless machine."""

import matplotlib
from matplotlib import pyplot

INTERACTIVE_BACKEND_NAMES = (
    "macosx",
    "qtagg",
    "qt5agg",
    "qtcairo",
    "tkagg",
    "tkcairo",
    "gtk3agg",
    "gtk4agg",
    "wxagg",
    "nbagg",
    "webagg",
)

HEADLESS_HINT_MESSAGE = (
    "No graphical display detected, the figure was not opened. "
    "Use the output option to write it to an image file instead."
)


def is_interactive_display_available():
    """Return True when the active backend can open a window."""
    active_backend_name = matplotlib.get_backend().lower()
    return active_backend_name in INTERACTIVE_BACKEND_NAMES


def display_figure_when_display_is_available(figure):
    """Open the figure in a window, or explain why it cannot be shown."""
    if is_interactive_display_available():
        pyplot.show()
        return True
    print(HEADLESS_HINT_MESSAGE)
    pyplot.close(figure)
    return False
