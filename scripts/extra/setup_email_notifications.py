#!/usr/bin/env python3

from socket import gethostname

from pypython.util import mailnotifs

print("Setting up email notifications by sending a message...")

message = mailnotifs.send_notification(
    "ejp1n17@soton.ac.uk", "Token creation success",
    "The computer {} has been setup to send email notifications".format(gethostname()))

print(message)
print("Success, I hope!")
