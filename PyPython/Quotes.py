#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains a bunch of random quotes in a list, as well as a function
which can pick a random quote.
"""


from numpy.random import randint


QUOTES = [
    # IRC
    "I have put you on a permanent ignore, public and private. I have found you disturbing, rude and generally not"
    "worth talking to. According to the channels you hang on, it strengtens the effect of wanting to put you on ignore"
    "because of my lack of interest in you as a person. This message is not meant to be rude to you, just to inform"
    "you that i won't see anything of what you type from now on.",
    # Ghastly Eyrie
    "From this Ghastly Eyrie I can see to the ends of the world, and from this vantage point I declare with utter"
    "certainty that this one is in the bag!",
    # Joker
    "We live in a society.",
    # RMS and /g/ technology
    "I tried to look at that page but saw only inane comments.",
    # Cat
    "If you were my cat, would I be able to neuter you?",
    # Marley #1
    "If you want to listen to Yung Lean then the leather club is two doors down",
    # Adam #1 
    "GGG: It's 4d chess.",
    # Adam #2
    "GGG: ive had a couple mouth fulls of my own fresh piss",
    # Adam #4
    "GGG: I will give you one word as a clue edward\n     COUP",
    # Adam #6
    "GGG: i can get 12 cans of carling for 8 leffe",
    # Jack # 1
    "Citrus Lime: my zesty nature keeps me celibate",
]


def random_quote():
    """
    Print a random quote to the screen.

    Returns
    -------
    quote: str
        The random quote.
    """

    nquotes = len(QUOTES)
    quote = QUOTES[randint(0, nquotes)]
    print(quote)
    print()

    return quote
