import logging

from .google import GoogleSaver, GoogleSheetInfo, SheetName
from .shifts import ShiftList, Shift
from .base import Sheets
from . import annealing, linear_prog

# Set up default logging configuration
logging.basicConfig(
    level=logging.INFO,  # Default logging level
    # format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",  # Log format
    format="%(levelname)s: %(message)s",  # Log format
)