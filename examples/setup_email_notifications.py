#!/usr/bin/env python3

from socket import gethostname
from pyPython import mailNotifications

print("Setting up email notifications by sending a message...")

message = mailNotifications.send_notification(
    "ejp1n17@soton.ac.uk", "Token creation success", 
    "The computer {} has been setup to send email notifications".format(gethostname())
)

print(message)
print("Success, I hope!")
