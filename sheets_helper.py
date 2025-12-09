import os
import logging
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.oauth2.credentials import Credentials

logger = logging.getLogger("python_server")

# ----------------------------
# Configuration
# ----------------------------
SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]  # Read + Write
SPREADSHEET_ID = "1Ol-Zu6mQIku0PeI39sOrmq7NNjU7cSbEVY17BInAdf8"
TOKEN_PATH = "token.json"


# ----------------------------
# OAuth2 Flow
# ----------------------------
def get_credentials():
    """
    Handles OAuth2 login using client_id + client_secret from environment.
    Stores token.json locally so the login happens only once.
    """
    load_dotenv()
    CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

    if not CLIENT_ID or not CLIENT_SECRET:
        raise ValueError("Missing GOOGLE_CLIENT_ID or GOOGLE_CLIENT_SECRET in .env file")

    creds = None

    # Load saved credentials if available
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    # If no valid credentials, authenticate
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_config(
            {
                "installed": {
                    "client_id": CLIENT_ID,
                    "client_secret": CLIENT_SECRET,
                    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "redirect_uris": ["http://localhost:8080"],
                }
            },
            SCOPES,
        )

        creds = flow.run_local_server(
            port=8080,
            access_type='offline',
            prompt='consent'
        )

        # Save the token to avoid reauth next time
        with open(TOKEN_PATH, "w") as token_file:
            token_file.write(creds.to_json())

    return creds


# ----------------------------
# Sheet Helpers
# ----------------------------
def get_sheets_service():
    """Returns authenticated Google Sheets service."""
    creds = get_credentials()
    return build("sheets", "v4", credentials=creds)


def list_sheet_names(service, spreadsheet_id: str = SPREADSHEET_ID) -> List[str]:
    """Returns all sheet/tab names in the spreadsheet."""
    metadata = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    return [s["properties"]["title"] for s in metadata.get("sheets", [])]


def read_sheet_by_name(service, sheet_name: str, cell_range: str = "A:Z", 
                       spreadsheet_id: str = SPREADSHEET_ID) -> List[List[str]]:
    """Reads values from a sheet using its tab name."""
    range_ref = f"{sheet_name}!{cell_range}"
    result = (
        service.spreadsheets()
        .values()
        .get(spreadsheetId=spreadsheet_id, range=range_ref)
        .execute()
    )
    return result.get("values", [])


def update_cell(service, sheet_name: str, row_number: int, column: str, value: str,
                spreadsheet_id: str = SPREADSHEET_ID) -> dict:
    """
    Updates a single cell in the sheet.
    
    Args:
        service: Google Sheets service
        sheet_name: Name of the sheet/tab
        row_number: Row number (1-indexed, including header)
        column: Column letter (e.g., 'A', 'B', 'C')
        value: Value to set
        spreadsheet_id: Spreadsheet ID
    
    Returns:
        Update response from Sheets API
    """
    range_ref = f"{sheet_name}!{column}{row_number}"
    body = {"values": [[value]]}
    
    result = service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=range_ref,
        valueInputOption="RAW",
        body=body
    ).execute()
    
    logger.info(f"Updated {range_ref} = {value}")
    return result


def filter_status_rows(rows: List[List[str]]) -> List[Dict]:
    """
    Filters rows based on status column logic.
    Returns list of dicts with row data + row_number.
    
    Keep rows where status != 'uploaded' AND != 'staged-to-cloudinary'
    """
    if not rows:
        return []

    header = rows[0]
    excluded = {"uploaded", "staged-to-cloudinary"}

    if "status" not in header:
        raise ValueError("The sheet does not contain a 'status' column.")

    status_idx = header.index("status")

    filtered = []
    for idx, row in enumerate(rows[1:], start=2):  # Start at 2 (row 1 is header)
        if len(row) > status_idx and row[status_idx] not in excluded:
            # Create a dict with column names as keys
            row_dict = {header[i]: row[i] if i < len(row) else "" 
                       for i in range(len(header))}
            row_dict["row_number"] = idx  # Add row number (1-indexed)
            filtered.append(row_dict)

    return filtered


def get_column_letter(header: List[str], column_name: str) -> str:
    """
    Converts column name to letter (e.g., 'status' -> 'D').
    
    Args:
        header: List of column names from the sheet
        column_name: Name of the column to find
    
    Returns:
        Column letter (A, B, C, etc.)
    """
    if column_name not in header:
        raise ValueError(f"Column '{column_name}' not found in header")
    
    col_index = header.index(column_name)
    # Convert 0-based index to Excel column letter
    letter = ""
    col_index += 1  # Make 1-based
    while col_index > 0:
        col_index -= 1
        letter = chr(col_index % 26 + ord('A')) + letter
        col_index //= 26
    
    return letter
