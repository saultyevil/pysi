#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This file contains a bunch of random quotes in a list, as well as a function
which can pick a random quote.
"""


from numpy.random import randint


QUOTES = [
    # IRC
    "I have put you on a permanent ignore, public and private. I have found you disturbing, rude and generally not "
    "worth talking to. According to the channels you hang on, it strengtens the effect of wanting to put you on ignore "
    "because of my lack of interest in you as a person. This message is not meant to be rude to you, just to inform "
    "you that i won't see anything of what you type from now on.",
    # Ghastly Eyrie
    "From this Ghastly Eyrie I can see to the ends of the world, and from this vantage point I declare with utter "
    "certainty that this one is in the bag!",
    # Joker
    "We live in a society.",
    # RMS and /g/ technology
    "I tried to look at that page but saw only inane comments.",
    # GNU
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
    # Systemd
    "First off, systemd is not an init system, it has an init system as part of the systemd suite. systemd is a "
    "project to build a standardised lowlevel userland for Linux. The project is pretty comprehensive and it delivers "
    "a lot of functionality under one umbrella. It does away with a lot of older, often undermaintained software "
    "packages, which were traditionally used to assemble a low level userland. Which is where the contention comes "
    "from, as a system suite systemd is restrictive for Unix virtuosi who are used to tailor a system with wit, "
    "ingenuity, a lick and a prayer and a couple dozen of unrelated packages. systemd makes such knowledge useless. "
    "The faction that thinks that systemd is Linux's Hiroshima, finds all the added functionality bloat, unnecessary "
    "and dangerous, as it is all under development in one project. All the systemd jokes stem from the "
    "comprehensiveness as a low level system suite. People against it love to joke that one day systemd will write"
    " its own kernel. There is a lot of FUD and hate going around. Some arguments do have merit, a lot of eggs in one "
    "basket is certainly true, but as with all things in life, it depends which tradeoff you prefer. Do you want a "
    "suite of well designed software, working closely together, so that system management is streamlined or do you "
    "want the complete freedom to tailor your own low level system with a lot of time tested, interchangeable "
    "components. I have no desire to be a low level system designer, so I prefer systemd. I don't hate traditional "
    "init systems though. If a Linux system has one and I need to work with it, I'm still happy it boots and starts "
    "the necessary services.",
    # PNG
    "Seeing the difference now isn't the reason to save photos as PNG. PNG uses lossless compression, while JPEG is "
    "'lossy'. What this means is that for each year the JPEG photographs sit on your hard drive, they will lose, on "
    "average, roughly 200 bits, assuming you have SATA - it's about 120 bits on IDE, but only 5...10 bits on SCSI, due "
    "to rotational velocidensity. You don't want to know how much worse it is on CD-ROM or other optical media. I "
    "started collecting JPEG pictures in about 2001, and if I try to view any of the photos I downloaded back then, "
    "even the stuff which was saved at 100% quality and 4:4:4 chroma subsampling, they just look like crap. The shadow "
    "detail is terrible, the highlights...well don't get me started. Some of those photos have degraded at a rate of "
    "100 or even 150 bits/year. PNG pictures from the same period still look great, even if they weren't stored "
    "correctly, in a cool, dry place. Seriously, stick to PNG, you may not be able to see the difference now, but in a "
    "year or two, you'll be glad you did.",
    # OSX Lion
    "listen y'all are just the many hatas u just jealous cause you dont got the best computer in the world or the best "
    "operating system OSX LION. when windows is out mac will take over control and what can you do? whimper and cry on "
    "your couch if you have one U STUPID FREAKING NOOBS MACS ARE THE BEST AND YOU ARE JUST JEALOUS. think what you "
    "want to say but dont say it because some of us actually use OSX the best operating system in the world to this "
    "very moment i started a blog on this topic and guess what? it was so popular they over ran the servers (by the "
    "way macs are the best) and if you disagree please PM me cause i would love to prove you wrong (not that you "
    "already are) and you need to go back to the third grade cause you guys cant spell so just read this carefully "
    "and you will see the difference between me and you adults and children (im a teen but considered an adult) so "
    "please just shut the freak up and say what you want to but macs are still the best",
    # Beats
    "Dr Dre Beats are the way to go. The ultra static dynamic magnetism within the upper echelon of the headphone has a"
    " tendency to help sharpen and brighten the lower range of the midrange, allowing both the warm sound of dynamic "
    "headphones and more precise tones of electrostatics to come in shape. The treble is where the clarity and "
    "calamity shines, you can literally touch the body of the treble while listening, thanks to thermodynamic "
    "activated noise cancelling outside noise won't affect the treble quality which is an issue in most headphones. "
    "We're not just talking about your screaming mother, but outside frequencies you cannot hear still decrease the "
    "sound quality. Thanks to the thermodynamics the treble truly shines. Finally the mids, the perfectly crafted V "
    "shaped sound signature making the headphone more fun, without reducing neutrality like the Denon's or Ultrasones. "
    "Dr Dre knows your brain has trouble registering abundant mid frequencies that are too forward and solidified, "
    "thus the mids remain where they should be. Slightly recessed with a many musical tones entwined with the sparkly "
    "high yet not buried by the mind crushing bass. Dr Dre knows bass better than anyone and boy has he shown it. The "
    "Beats have a bass with the lush sonics only apparent in subwoofers while keeping the analytical veil of the "
    "higher end AKG line, it's not a question of either or with Beats. You get the perfect cold sound of the AKG's "
    "with the planarmagnetically enhanced sub frequencies you normally can't find in headphones. The bass dances "
    "around while the sonics caress the mids bringing both forward, yet without recessing the beautifully scent "
    "filled treble. The bass doesn't only vibrate your body, it vibrates your subconscious and soul. Only thanks to "
    "the psychologically altered rotational velocidensities engineered into the sub frequencies. The Beats are a "
    "complete package and you would be stupid to get anything else.",
    # Programming
    "You talk so much about programming, but you cant talk about hacking because you cant actually do anything. I run "
    "a hacked network of computers that I programmed to click on googles ads in my secret website. I even write my own "
    "viruses to make people get hacked into my network. I work at home and have a bunch of screens showing me what "
    "people on my network are doing on their screens. I can even set it so that i can see the code of their computers. "
    "can you guys do any of that? I dont think so. I bet you dont know where all the websites real hackers hang out "
    "are either? if you name them, I just might tell them that marshviperX sent you.",
    # Baka Mitai
    "Baka mitai Kodomo na no ne. Yume wo otte kidzutsuite .Uso ga heta na kuse ni waraenai egao wo miseta. I love you "
    "mo roku ni iwanai. Kuchibeta de honma ni bukiyou. Na no ni na no ni doushite sayonara ha ieta no. Dame da ne. "
    "Dame yo dame na no yo. Anta ga suki de sukisugite. Dore dake tsuyoi osake demo. Yugamanai omoide ga baka mitai. "
    "Baka mitai hontou baka ne. Anta shinjiru bakari de. Tsuyoi onna no furi setsunasa no yokaze abiru. Hitori ni "
    "natte sannen ga sugi. Machinami sae mo kawarimashita. Na no ni na no ni doushite miren dake okizari. Honma ni roku"
    " na otoko ya nai. Soroi no yubiwa hazushimasu. Zamaa miro seisei suru wa. Ii kagen mattete mo baka mitai. Dame da"
    " ne Dame yo dame na no yo. Anta ga sukide sukisugite. Dore dake tsuyoi osake demo. Yugamanai omoide ga baka "
    "mitai. Honma ni roku na otoko ya nai. Soroi no yubiwa hazushimasu. Zamaa miro seisei suru wa. Nan na yo kono "
    "namida baka mitai",
    # Judgement
    "Reeru kara hazureta furyouhin no norainu sa. Da kedo kantan ni wa teeru wa furanai ze. Iesuman ni narisobireta. "
    "Waru ni nokosareta no to iu na no Justice. Wow, Breakin’ the law. Breakin’ the world　Kowase. Kirisake "
    "Tenderness. Wow, Breakin’ the rule. Roppouzensho ja shibarenai hanran bunshi sa. Furiageta nigirikobushi ga "
    "oretachi no JUDGEMENT. Ai saae mo shiranai furyouhin no norainu sa. da kedo chii ya kane ja edzuke wa dekinai "
    "ze. Kizutsuite kizutsuite subete ushinacchimattemo. somacchanaranee sore koso ga JUSTICE……　JUSTICE!. Wow, "
    "Breakin’ the law. Breakin’ the world　Kowase. Kirisake Tenderness. Wow, Breakin’ the rule. Roppouzensho ja "
    "shibarenai hanran bunshi sa. Furiagero nigirikobushi wo oretachi wa…… Sou sa oretachi ga JUDGEMENT. JUDGEMENT……",
    # Cat
    "If you were my cat, would I be able to neuter you?",
    # Marley #1
    "If you want to listen to Yung Lean then the leather club is two doors down",
    # Adam #1 
    "It's 4d chess.",
    # Adam #2
    "GGGG: ive had a couple mouth fulls of my own fresh piss",
    # Adam #3
    "GGGG: why do slightly above knee length skirts and dresses make me want to do a rape",
    # Adam #4
    "GGGG: I will give you one word as a clue edward\n      COUP",
    # Adam #5
    "GGGG: youre saying no to a girl but yes to a dick, even if it is plastic",
    # Adam #6
    "GGGG: i can get 12 cans of carling for 8 leffe",
    # Adam #7
    "GGGG: yeah but youre just odd not homo",
    # Adam #8
    "GGGG: you know those africans that carry shit around their heads\n      why dont they have swole necks"
    # Sean Connery
    "I am Sean Connery and I want a massage",
    # Marley # 1
    "Hypnotized: If you're gonna dress like a woman, at least get your dick out"
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
