import requests
import json
import datetime
from datetime import datetime
from datetime import timezone

import dateutil

from disnake.ext import commands


class F1(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.slash_command(description="Next F1 Race")
    async def f1(
        self,
        inter,
        requested_timezone=commands.Param(
            default="Europe/London",
            autocomplete=[
                "Europe/London",
                "America/Los_Angeles",
                "America/Chicago",
                "Australia/Adelaide",
                "Europe/Bucharest",
            ],
        ),
    ):
        embedf1 = self.f1_calendar(requested_timezone)
        await inter.response.send_message(embed=embedf1)

    @staticmethod
    def f1_calendar(requested_timezone):
        calendar = requests.get("https://f1calendar.com/api/year/2023")
        f1data = json.loads(calendar.text)
        usertimezone = requested_timezone

        # loops through the list of races and finds the next grand prix in the future
        today = datetime.now(timezone.utc)
        next_race = None
        for race in f1data["races"]:
            gp_time = dateutil.parser.parse(race["sessions"]["gp"])
            if gp_time > today:
                next_race = race
                break
