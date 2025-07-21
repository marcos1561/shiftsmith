from enum import Enum
from pathlib import Path
import os.path
import pandas as pd

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class SheetName(Enum):
    TARGET_WORKLOAD = "target_workload"
    AVAILABILITY = "availability"
    PREFERENCE = "preference"

class GoogleSheetInfo:
    def __init__(self, sheet_id, sheet_range, sheet_name: SheetName):
        self.id = sheet_id
        self.range = sheet_range
        self.sheet_name = sheet_name

class GoogleSaver:
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

    def __init__(self, credentials_path, save_path, sheet_info_arr:list[GoogleSheetInfo]):
        self.save_path = Path(save_path)
        self.credentials_path = Path(credentials_path)
        self.sheet_info_arr = sheet_info_arr

        self.creds = self.get_credentials()

    def get_credentials(self):
        creds = None
        SCOPES = GoogleSaver.SCOPES

        # The file token.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists(self.credentials_path.parent / "token.json"):
            creds = Credentials.from_authorized_user_file("token.json", SCOPES)
        
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_path, SCOPES
                )
                creds = flow.run_local_server(port=0)
        
            # Save the credentials for the next run
            with open(self.credentials_path.parent / "token.json", "w") as token:
                token.write(creds.to_json())

        return creds

    def read_sheet(self, sheet_info: GoogleSheetInfo):
        service = build("sheets", "v4", credentials=self.creds)

        # Call the Sheets API
        sheet = service.spreadsheets()
        result = (
            sheet.values()
            .get(spreadsheetId=sheet_info.id, range=sheet_info.range)
            .execute()
        )
        values = result.get("values", [])

        num_cols = len(values[0])

        for row in values[1:]:
            row: list
            if len(row) < num_cols:
                row.extend([""] * (num_cols - len(row)))
        return values

    def save_sheet(self, sheet_info: GoogleSheetInfo, post_process_func=None):
        values = self.read_sheet(sheet_info)
        
        df = pd.DataFrame(values[1:], columns=values[0])
        
        if post_process_func:
            post_process_func(df, sheet_info)
        
        df.to_csv(self.save_path / f"{sheet_info.sheet_name.value}.csv", index=None)

    def update(self, post_process_func=None):
        for sheet_info in self.sheet_info_arr:
            self.save_sheet(sheet_info, post_process_func)