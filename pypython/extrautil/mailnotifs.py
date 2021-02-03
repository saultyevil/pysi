#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tools to send email notifications to update users on the current progress or
some long program. Requires access to the mcrtpythonupdates@gmail.com Gmail API.
Currently access to this API is limited.
"""

import pickle
import os
import base64
from typing import Union
from email.mime.text import MIMEText
from googleapiclient.discovery import build, Resource
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request


def create_email_message(
    sender: str, to: str, subject: str, message: str
) -> dict:
    """Create an email message and encode it.

    Parameters
    ----------
    sender: str
        The email address to send the notification from.
    to: str
        The email address to create the message for.
    subject: str
        The subject of the email notification.
    message: str
        The main body of the email.

    Returns
    -------
    message: dict
        The created email message"""

    message = MIMEText(message)
    message["to"] = to
    message["from"] = sender
    message["subject"] = subject
    message = {"raw": base64.urlsafe_b64encode(message.as_string().encode()).decode()}

    return message


def send_email_message(
    service: Resource, msg: dict, user: str
) -> Union[None, dict]:
    """Send an email message using the API service.

    Parameters
    ----------
    service: discovery.Resource
        The service API.
    msg: dict
        The message to be sent
    user: str
        The user id for the service

    Returns
    -------
    msg: dict
        The email message sent"""

    try:
        msg = service.users().messages().send(userId=user, body=msg).execute()
        return msg
    except Exception as e:
        print("Unable to send email message")
        # print(e)

    return


def send_notification(
    to: str, subject: str, notification: str, sender: str = "mcrtpythonupdates@gmail.com"
) -> dict:
    """Send a notification email to the user. Requires access to the Gmail API.

    Parameters
    ----------
    to: str
        The email address to create the message for.
    subject: str
        The subject of the email notification.
    notification: str
        The main body of the email.
    sender: str [optional]
        The email address to send the notification from.

    Returns
    -------
    message: dict
        The message sent"""

    credentials = None
    scope = ["https://www.googleapis.com/auth/gmail.compose"]
    message = create_email_message(sender, to, subject, notification)

    # Store the users access in token.pickle. It is created automatically when
    # the authorization flow completes for the first time.

    token_path = os.path.expanduser("~/.mcrtpythonupdate_token.pickle")

    if os.path.exists(token_path):
        with open(token_path, "rb") as token:
            credentials = pickle.load(token)

    # If there are no (valid) credentials available, let the user log in

    try:
        if not credentials or not credentials.valid:

            if credentials and credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", scope)
                credentials = flow.run_local_server(port=0)

            # Save the credentials for next time

            with open(token_path, "wb") as token:
                pickle.dump(credentials, token)

        service = build('gmail', 'v1', credentials=credentials)
        message = send_email_message(service, message, sender)
        return message
    except Exception as e:
        print("Cant send email for some reason, token probably expired")
        return {}
