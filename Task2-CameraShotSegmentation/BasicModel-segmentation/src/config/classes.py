import torch

EVENT_DICTIONARY_V1 = {"soccer-ball": 0, "soccer-ball-own": 0, "r-card": 1, "y-card": 1, "yr-card": 1,
                                 "substitution-in": 2}

EVENT_DICTIONARY_V2 = {"Penalty":0,"Kick-off":1,"Goal":2,"Substitution":3,"Offside":4,"Shots on target":5,
                                "Shots off target":6,"Clearance":7,"Ball out of play":8,"Throw-in":9,"Foul":10,
                                "Indirect free-kick":11,"Direct free-kick":12,"Corner":13,"Yellow card":14
                                ,"Red card":15,"Yellow->red card":16}
K_V1 = torch.FloatTensor([[-20,-20,-40],[-10,-10,-20],[60,10,10],[90,20,20]]).cuda()
K_V2 = torch.FloatTensor([[-20,-20,-20,-40,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20,-20],[-10,-10,-10,-20,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10,-10],[10,10,60,10,10,10,10,10,10,10,10,10,10,10,10,10,10],[20,20,90,20,20,20,20,20,20,20,20,20,20,20,20,20,20]]).cuda()
Camera_Change_DICTIONARY = {"abrupt": 1, "logo": 1, "smooth": 1}
# Camera_Type_DICTIONARY = {"Main camera center":0,"Close-up player or field referee":1,"Main camera left":2,"Main camera right":3,"Goal line technology camera":4,"Main behind the goal": 5,"Spider camera":6,
#                                 "Close-up side staff":7,"Close-up corner":8,"Close-up behind the goal":9,"Inside the goal":10,"Public":11}
Camera_Type_DICTIONARY = {"Main camera center":0,"Close-up player or field referee":1,"Main camera left":2,"Main camera right":3,"Goal line technology camera":4,"Main behind the goal": 5,"Spider camera":6,
                                "Close-up side staff":7,"Close-up corner":8,"Close-up behind the goal":9,"Inside the goal":10,"Public":11,"other":12,"I don't know":12}

# Weight_camera_type= torch.tensor([0.00376631, 0.00101809, 0.04597685, 0.05156424, 0.33772375, 0.04499323, 0.14527238, 0.00628741, 0.05253881, 0.02011916,0.26395199, 0.02678777], dtype=torch.float).cuda()
Weight_camera_type= torch.tensor([0.00143, 0.0003865, 0.0174, 0.0196, 0.128, 0.00171, 0.0552, 0.00239, 0.02, 0.00765,0.1001, 0.0101,0.594,0.0253], dtype=torch.float).cuda()
                                
