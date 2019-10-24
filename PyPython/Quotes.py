#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains a bunch of random quotes in a list, as well as a function
which can pick a random quote.
"""


from numpy.random import randint


QUOTES = [
    "I have put you on a permanent ignore, public and private. I have found you disturbing, rude and generally not "
    "worth talking to. According to the channels you hang on, it strengtens the effect of wanting to put you on ignore "
    "because of my lack of interest in you as a person. This message is not meant to be rude to you, just to inform "
    "you that i won't see anything of what you type from now on.",
    "From this Ghastly Eyrie I can see to the ends of the world, and from this vantage point I declare with utter "
    "certainty that this one is in the bag!",
    "We live in a society.",
    "I tried to look at that page but saw only inane comments.",
    "I'd just like to interject for a moment. What you're referring to as Linux, is in fact, GNU/Linux, or as I've "
    "recently taken to calling it, GNU plus Linux. Linux is not an operating system unto itself, but rather another "
    "free component of a fully functioning GNU system made useful by the GNU corelibs, shell utilities and vital "
    "system components comprising a full OS as defined by POSIX. Many computer users run a modified version of the "
    "GNU system every day, without realizing it. Through a peculiar turn of events, the version of GNU which is "
    "widely used today is often called 'Linux', and many of its users are not aware that it is basically the GNU "
    "system, developed by the GNU Project. There really is a Linux, and these people are using it, but it is just a "
    "part of the system they use. Linux is the kernel: the program in the system that allocates the machine's "
    "resources to the other programs that you run. The kernel is an essential part of an operating system, but useless "
    "by itself; it can only function in the context of a complete operating system. Linux is normally used in "
    "combination with the GNU operating system: the whole system is basically GNU with Linux added, or GNU/Linux. "
    "All the so-called 'Linux' distributions are really distributions of GNU/Linux.",
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

    return quote
