import torch


# Event name to label index fororor SoccerNet-V2
EVENT_DICTIONARY_V2 = {"Penalty":0,"Kick-off":1,"Goal":2,"Substitution":3,"Offside":4,"Shots on target":5,
                                "Shots off target":6,"Clearance":7,"Ball out of play":8,"Throw-in":9,"Foul":10,
                                "Indirect free-kick":11,"Direct free-kick":12,"Corner":13,"Yellow card":14
                                ,"Red card":15,"Yellow->red card":16}
INVERSE_EVENT_DICTIONARY_V2 = {0:"Penalty",1:"Kick-off",2:"Goal",3:"Substitution",4:"Offside",5:"Shots on target",
                                6:"Shots off target",7:"Clearance",8:"Ball out of play",9:"Throw-in",10:"Foul",
                                11:"Indirect free-kick",12:"Direct free-kick",13:"Corner",14:"Yellow card"
                                ,15:"Red card",16:"Yellow->red card"}

# Separation for Only patterned classes 
EVENT_DICTIONARY_V2_VISUAL = {"Penalty":0,"Kick-off":1,"Throw-in":2,"Direct free-kick":3,"Corner":4,"Yellow card":5,"Red card":6,"Yellow->red card":7}

INVERSE_EVENT_DICTIONARY_V2_VISUAL = {0:"Penalty",1:"Kick-off",2:"Throw-in",3:"Direct free-kick",4:"Corner",5:"Yellow card",6:"Red card",7:"Yellow->red card"}

# Only fuzzy classes 
EVENT_DICTIONARY_V2_NONVISUAL = {"Goal":0,"Substitution":1,"Offside":2,"Shots on target":3,"Shots off target":4,"Clearance":5,"Ball out of play":6,"Foul":7,"Indirect free-kick":8}

INVERSE_EVENT_DICTIONARY_V2_NONVISUAL = {0:"Goal",1:"Substitution",2:"Offside",3:"Shots on target",4:"Shots off target",5:"Clearance",6:"Ball out of play",7:"Foul",8:"Indirect free-kick"}

# Values of the K parameters (in seconds) in the context-aware loss
K_V2 = torch.FloatTensor([[-100, -98, -20, -40, -96, -5, -8, -93, -99, -31, -75, -10, -97, -75, -20, -84, -18], [-50, -49, -10, -20, -48, -3, -4, -46, -50, -15, -37, -5, -49, -38, -10, -42, -9], [50, 49, 60, 10, 48, 3, 4, 46, 50, 15, 37, 5, 49, 38, 10, 42, 9], [100, 98, 90, 20, 96, 5, 8, 93, 99, 31, 75, 10, 97, 75, 20, 84, 18]]).cuda()

K_V2_VISUAL = torch.FloatTensor([[-100, -98, -31, -97, -75, -20, -84, -18], [-50, -49, -15, -49, -38, -10, -42, -9], [50, 49, 15, 49, 38, 10, 42, 9], [100, 98, 31, 97, 75, 20, 84, 18]]).cuda()

K_V2_NONVISUAL = torch.FloatTensor([[-20, -40, -96, -5, -8, -93, -99, -75, -10], [-10, -20, -48, -3, -4, -46, -50, -37, -5], [60, 10, 48, 3, 4, 46, 50, 37, 5], [90, 20, 96, 5, 8, 93, 99, 75, 10]]).cuda()
