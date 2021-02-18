

class Event:

	def __init__(self, label=None, half=None, time=None, team=None, position= None, visibility=None):

		self.label = label
		self.half = half
		self.time = time
		self.team = team
		self.position = position
		self.visibility = visibility

	def to_text(self):
		return self.time + " || " + self.label + " - " + self.team  + " - " + str(self.half) + " - " + str(self.visibility)

	def __lt__(self, other):
		self.position < other.position

def ms_to_time(position):
	minutes = int(position//1000)//60
	seconds = int(position//1000)%60
	return str(minutes).zfill(2) + ":" + str(seconds).zfill(2)