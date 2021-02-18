

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


class Camera:

	def __init__(self, label=None, half=None, time=None, change_type=None, position= None, replay=None, link=None):

		self.label = label
		self.half = half
		self.time = time
		self.change_type = change_type
		self.position = position
		self.replay = replay
		self.link = link

	def to_text(self):
		if self.link is None:
			return self.time + " || " + self.label + " - " + self.change_type  + " - " + str(self.half) + " - " + str(self.replay)
		else:
			return self.time + " || " + self.label + " - " + self.change_type  + " - " + str(self.half) + " - " + str(self.replay) + "\n<--" + self.link.to_text()

	def __lt__(self, other):
		self.position < other.position


def ms_to_time(position):
	minutes = int(position//1000)//60
	seconds = int(position//1000)%60
	return str(minutes).zfill(2) + ":" + str(seconds).zfill(2)

