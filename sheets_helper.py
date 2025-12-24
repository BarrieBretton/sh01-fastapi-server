"""
Google Sheets Helper with Auto-Refresh Token Management

This module provides a simple interface to Google Sheets with automatic
token refresh. Authenticate once, and it works forever (until revoked).

Setup:
1. Go to https://console.cloud.google.com/
2. Create a new project (or use existing)
3. Enable Google Sheets API
4. Create OAuth 2.0 credentials (Web app OR Desktop app)
   - For Web app: Add http://localhost:8080/ to authorized redirect URIs
5. Download credentials.json and place in project root
6. First run will authenticate (see Docker setup below for headless)
7. Token stored in token.json - keep this file safe!

Docker/Headless Setup:
1. Set SHEETS_HEADLESS=true
2. First run will print an auth URL
3. Open URL in browser on ANY machine
4. After authorizing, copy the redirect URL
5. Set SHEETS_AUTH_CODE with the full URL or just the 'code' parameter
6. Restart - token.json will be created and work forever

Environment Variables:
- SPREADSHEET_ID: Your Google Sheets ID (required)
- SHEETS_CREDENTIALS_PATH: Path to credentials.json (default: ./credentials.json)
- SHEETS_TOKEN_PATH: Path to token.json (default: ./token.json)
- SHEETS_AUTH_PORT: Port for OAuth redirect (default: 8080)
- SHEETS_HEADLESS: Set to 'true' for Docker/headless (default: false)
- SHEETS_AUTH_CODE: Authorization code or full redirect URL (headless only)
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

import ssl
import certifi
# Set the SSL context to use certifi's certificates
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
ssl_context = ssl.create_default_context(cafile=certifi.where())

from google.oauth2 import service_account

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# Configure logging
logger = logging.getLogger(__name__)

# Scopes required for reading and writing to Google Sheets
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']

load_dotenv(".env")

# Get configuration from environment
print('env:: ', os.environ)

SPREADSHEET_ID = os.getenv('SPREADSHEET_ID')
CREDENTIALS_PATH = Path(os.getenv('SHEETS_CREDENTIALS_PATH', 'credentials.json'))
TOKEN_PATH = Path(os.getenv('SHEETS_TOKEN_PATH', 'token.json'))
AUTH_PORT = int(os.getenv('SHEETS_AUTH_PORT', '8080'))  # Default to 8080 for web apps
HEADLESS_MODE = os.getenv('SHEETS_HEADLESS', 'false').lower() == 'true'  # For Docker/headless

if not SPREADSHEET_ID:
    logger.warning("SPREADSHEET_ID not set in environment variables")


def detect_client_type(credentials_path: Path) -> str:
    """
    Detect if credentials.json is for web app or desktop/installed app.
    
    Web app format:
    {
        "web": {
            "client_id": "...",
            "client_secret": "...",
            "redirect_uris": ["https://..."],
            ...
        }
    }
    
    Desktop/Installed app format:
    {
        "installed": {
            "client_id": "...",
            "client_secret": "...",
            "redirect_uris": ["http://localhost", ...],
            ...
        }
    }
    
    Returns:
        'web' or 'installed'
    """
    try:
        with open(credentials_path) as f:
            creds_data = json.load(f)
            
        # Check if it's a web application
        if 'web' in creds_data:
            logger.info("Detected web application credentials")
            return 'web'
        elif 'installed' in creds_data:
            logger.info("Detected installed/desktop application credentials")
            return 'installed'
        else:
            logger.warning("Could not determine client type from keys: %s. Defaulting to installed", list(creds_data.keys()))
            return 'installed'
    except Exception as e:
        logger.warning("Error reading credentials file: %s. Defaulting to installed", e)
        return 'installed'


# ─────────────────────────────────────────────────────────────────────────────
# 1. CREDENTIALS – Service Account OR OAuth2 (user token (both supported)
# ─────────────────────────────────────────────────────────────────────────────
def get_credentials():
    """
    Returns authenticated Google credentials.
    Priority order (highest first):
      1. GOOGLE_SERVICE_ACCOUNT  → service account (recommended for servers)
      2. GOOGLE_TOKEN_JSON            → OAuth2 user token (with refresh (for personal sheets))
      3. service_account.json file    → fallback for local dev only
      4. token.json file              → fallback for local dev only
    """

    # 1. Service Account from env var (most secure & recommended)
    sa_json = os.getenv("GOOGLE_SERVICE_ACCOUNT")
    if sa_json:
        try:
            info = json.loads(sa_json)
            return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        except Exception as e:
            raise RuntimeError(f"Invalid GOOGLE_SERVICE_ACCOUNT: {e}")

    # 2. OAuth2 user token with refresh from env var
    token_json = os.getenv("GOOGLE_TOKEN_JSON")
    if token_json:
        try:
            creds_info = json.loads(token_json)
            creds = Credentials.from_authorized_user_info(creds_info, SCOPES)
            # Refresh if expired
            if creds.expired and creds.refresh_token:
                creds.refresh(Request())
                logger.info("Refreshed expired Google OAuth2 token from GOOGLE_TOKEN_JSON")
            return creds
        except Exception as e:
            raise RuntimeError(f"Invalid GOOGLE_TOKEN_JSON: {e}")

    # 3. Fallback: service_account.json file (local dev only)
    if os.path.exists("service_account.json"):
        logger.warning("Using local service_account.json (only for development)")
        return service_account.Credentials.from_service_account_file("service_account.json", scopes=SCOPES)

    # 4. Fallback: token.json file (local dev only)
    if os.path.exists("token.json"):
        logger.warning("Using local token.json (only for development)")
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
            # Save refreshed token back to file for next local run
            with open("token.json", "w") as f:
                f.write(creds.to_json())
        return creds

    raise RuntimeError(
        "No Google credentials found!\n"
        "Set either:\n"
        "  • GOOGLE_SERVICE_ACCOUNT (recommended), or\n"
        "  • GOOGLE_TOKEN_JSON\n"
        "or place service_account.json / token.json in project root for local dev."
    )

def get_sheets_service():
    """
    Get an authenticated Google Sheets service instance using Service Account.
    """
    try:
        creds = get_credentials()
        service = build('sheets', 'v4', credentials=creds)
        return service
    except Exception as e:
        logger.exception("Failed to create Sheets service")
        raise RuntimeError(f"Could not authenticate with Google Sheets: {e}")


def list_sheet_names(service, spreadsheet_id: str = None) -> List[str]:
    """
    Get all sheet names in a spreadsheet.
    
    Args:
        service: Google Sheets service instance
        spreadsheet_id: Spreadsheet ID (uses env var if not provided)
        
    Returns:
        List of sheet names
    """
    spreadsheet_id = spreadsheet_id or SPREADSHEET_ID
    if not spreadsheet_id:
        raise ValueError("spreadsheet_id required (or set SPREADSHEET_ID env var)")
    
    try:
        sheet_metadata = service.spreadsheets().get(
            spreadsheetId=spreadsheet_id
        ).execute()
        
        sheets = sheet_metadata.get('sheets', [])
        return [sheet['properties']['title'] for sheet in sheets]
    
    except HttpError as e:
        logger.error("Failed to list sheet names: %s", e)
        raise


def read_sheet_by_name(
    service,
    sheet_name: str,
    spreadsheet_id: str = None,
    value_render_option: str = 'UNFORMATTED_VALUE'
) -> List[List[Any]]:
    """
    Read all data from a sheet by name.
    
    Args:
        service: Google Sheets service instance
        sheet_name: Name of the sheet tab
        spreadsheet_id: Spreadsheet ID (uses env var if not provided)
        value_render_option: How to render values (FORMATTED_VALUE, UNFORMATTED_VALUE, FORMULA)
        
    Returns:
        List of rows, where each row is a list of cell values
    """
    spreadsheet_id = spreadsheet_id or SPREADSHEET_ID
    if not spreadsheet_id:
        raise ValueError("spreadsheet_id required")
    
    try:
        result = service.spreadsheets().values().get(
            spreadsheetId=spreadsheet_id,
            range=f"'{sheet_name}'",  # Single quotes handle spaces in names
            valueRenderOption=value_render_option
        ).execute()
        
        rows = result.get('values', [])
        logger.info("Read %d rows from sheet '%s'", len(rows), sheet_name)
        return rows
    
    except HttpError as e:
        logger.error("Failed to read sheet '%s': %s", sheet_name, e)
        raise


def filter_status_rows(
    rows: List[List[Any]],
    status_values: List[str] = None,
    exclude_status: bool = False  # NEW: When True, exclude these statuses instead of including
) -> List[Dict[str, Any]]:
    """
    Filter rows by status column and return as dictionaries.
    
    Assumes first row is header with columns including 'status'.
    
    Args:
        rows: Raw rows from read_sheet_by_name
        status_values: List of status values to filter
        exclude_status: If True, exclude rows with these statuses (default: False = include)
        
    Returns:
        List of dicts with column names as keys, plus 'row_number' key
    """
    if not rows:
        return []
    
    status_values = status_values or ['staged']
    header = rows[0]
    
    # Find status column index
    try:
        status_idx = header.index('status')
    except ValueError:
        logger.warning("No 'status' column found in header")
        return []
    
    filtered = []
    for i, row in enumerate(rows[1:], start=2):  # Start at row 2 (Excel row numbering)
        # Handle rows shorter than header
        while len(row) < len(header):
            row.append('')
        
        row_dict = {header[j]: row[j] for j in range(len(header))}
        row_dict['row_number'] = i  # Add Excel row number
        
        current_status = row_dict.get('status', '')
        
        if exclude_status:
            # Exclude rows with status in status_values
            if current_status not in status_values:
                filtered.append(row_dict)
        else:
            # Include rows with status in status_values
            if current_status in status_values:
                filtered.append(row_dict)
    
    if exclude_status:
        logger.info("Filtered %d rows excluding status in %s", len(filtered), status_values)
    else:
        logger.info("Filtered %d rows with status in %s", len(filtered), status_values)
    
    return filtered

def update_cell(
    service,
    sheet_name: str,
    row_number: int,
    column_letter: str,
    value: Any,
    spreadsheet_id: str = None
) -> None:
    """
    Update a single cell by row number and column letter.
    
    Args:
        service: Google Sheets service instance
        sheet_name: Name of the sheet tab
        row_number: Row number (1-indexed, Excel-style)
        column_letter: Column letter (A, B, C, etc.)
        value: Value to write
        spreadsheet_id: Spreadsheet ID (uses env var if not provided)
    """
    spreadsheet_id = spreadsheet_id or SPREADSHEET_ID
    if not spreadsheet_id:
        raise ValueError("spreadsheet_id required")
    
    cell_range = f"'{sheet_name}'!{column_letter}{row_number}"
    
    try:
        body = {'values': [[value]]}
        service.spreadsheets().values().update(
            spreadsheetId=spreadsheet_id,
            range=cell_range,
            valueInputOption='RAW',
            body=body
        ).execute()
        
        logger.debug("Updated cell %s to '%s'", cell_range, value)
    
    except HttpError as e:
        logger.error("Failed to update cell %s: %s", cell_range, e)
        raise


def get_column_letter(header: List[str], column_name: str) -> str:
    """
    Convert column name to Excel-style column letter (A, B, C, ..., Z, AA, AB, ...).
    
    Args:
        header: List of column names from first row
        column_name: Name of column to find
        
    Returns:
        Column letter (e.g., 'A', 'B', 'AA')
        
    Raises:
        ValueError: If column not found
    """
    try:
        col_index = header.index(column_name)
    except ValueError:
        raise ValueError(f"Column '{column_name}' not found in header")
    
    # Convert 0-indexed to Excel column letter
    letter = ''
    col_index += 1  # Convert to 1-indexed
    
    while col_index > 0:
        col_index -= 1
        letter = chr(col_index % 26 + ord('A')) + letter
        col_index //= 26
    
    return letter


def batch_update_cells(
    service,
    sheet_name: str,
    updates: List[Dict[str, Any]],
    spreadsheet_id: str = None
) -> None:
    """
    Update multiple cells in a single batch request.
    
    More efficient than multiple update_cell calls.
    
    Args:
        service: Google Sheets service instance
        sheet_name: Name of the sheet tab
        updates: List of dicts with keys: row_number, column_letter, value
        spreadsheet_id: Spreadsheet ID (uses env var if not provided)
        
    Example:
        updates = [
            {'row_number': 2, 'column_letter': 'A', 'value': 'done'},
            {'row_number': 2, 'column_letter': 'B', 'value': 'https://...'},
        ]
    """
    spreadsheet_id = spreadsheet_id or SPREADSHEET_ID
    if not spreadsheet_id:
        raise ValueError("spreadsheet_id required")
    
    if not updates:
        return
    
    data = []
    for update in updates:
        cell_range = f"'{sheet_name}'!{update['column_letter']}{update['row_number']}"
        data.append({
            'range': cell_range,
            'values': [[update['value']]]
        })
    
    try:
        body = {'data': data, 'valueInputOption': 'RAW'}
        service.spreadsheets().values().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body=body
        ).execute()
        
        logger.info("Batch updated %d cells in sheet '%s'", len(updates), sheet_name)
    
    except HttpError as e:
        logger.error("Batch update failed: %s", e)
        raise


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*60)
    print("Google Sheets Helper - Authentication Test")
    print("="*60 + "\n")
    
    try:
        # Test authentication
        print("→ Testing authentication...")
        service = get_sheets_service()
        print("✓ Authentication successful!\n")
        
        # Test listing sheets (if SPREADSHEET_ID is set)
        if SPREADSHEET_ID:
            print(f"→ Reading spreadsheet: {SPREADSHEET_ID}")
            sheets = list_sheet_names(service)
            print(f"✓ Found {len(sheets)} sheets: {sheets}\n")
            
            # Test reading first sheet
            if sheets:
                print(f"→ Reading sheet '{sheets[0]}'...")
                rows = read_sheet_by_name(service, sheets[0])
                print(f"✓ Read {len(rows)} rows from '{sheets[0]}'")
                if rows:
                    print(f"  Header: {rows[0][:5]}...")  # Show first 5 columns
                    if len(rows) > 1:
                        print(f"  First row: {rows[1][:5]}...")
        else:
            print("⚠ Set SPREADSHEET_ID environment variable to test reading/writing")
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60 + "\n")
    
    except Exception as e:
        print(f"\n✗ Error: {e}\n")
        import traceback
        traceback.print_exc()
        raise

