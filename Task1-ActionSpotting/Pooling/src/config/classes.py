import torch

# EVENT_DICTIONARY_V1 = {"soccer-ball": 0, "soccer-ball-own": 0, "r-card": 1, "y-card": 1, "yr-card": 1,
#                                  "substitution-in": 2}
EVENT_DICTIONARY_V1 = {"soccer": 0, "card": 1, "subs": 2}
EVENT_DICTIONARY_V2 = {"Penalty":0,"Kick-off":1,"Goal":2,"Substitution":3,"Offside":4,"Shots on target":5,
                                "Shots off target":6,"Clearance":7,"Ball out of play":8,"Throw-in":9,"Foul":10,
                                "Indirect free-kick":11,"Direct free-kick":12,"Corner":13,"Yellow card":14
                                ,"Red card":15,"Yellow->red card":16}
