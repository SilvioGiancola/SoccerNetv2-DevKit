from utils.event_class import Event, Camera
import json
import os

class ListManager:

	def __init__(self):

		self.event_list = list()

	def create_list_from_json(self, path, half, camera=True):

		self.event_list.clear()
		if camera:
			self.event_list = self.read_json_camera(path, half)
		else:
			self.event_list = self.read_json_event(path, half)
		self.sort_list()

	def create_text_list(self):

		list_text = list()
		for event in self.event_list:
			list_text.append(event.to_text())

		return list_text

	def delete_event(self, index):

		self.event_list.pop(index)
		self.sort_list()

	def add_event(self, event):

		self.event_list.append(event)
		self.sort_list()

	def sort_list(self):

		position = list()
		for event in self.event_list:
			position.append(event.position)

		self.event_list = [x for _,x in sorted(zip(position,self.event_list))]

		self.event_list.reverse()

	def read_json_event(self, path, half):

		event_list = list()
		if os.path.isfile(path):
			with open(path) as file:
				data = json.load(file)["annotations"]
				for event in data:
					tmp_half = int(event["gameTime"][0])
					if tmp_half == half:
						tmp_time = event["gameTime"][4:]
						tmp_position = int(event["position"])
						tmp_label = event["label"]
						tmp_team = event["team"]
						tmp_visibility = event["visibility"]
						event_list.append(Event(tmp_label, tmp_half, tmp_time, tmp_team, tmp_position, tmp_visibility))
		return event_list

	def read_json_camera(self, path, half):

		event_list = list()
		if os.path.isfile(path):
			with open(path) as file:
				data = json.load(file)["annotations"]
				for event in data:
					tmp_half = int(event["gameTime"][0])
					if tmp_half == half:
						tmp_time = event["gameTime"][4:]
						tmp_position = int(event["position"])
						tmp_label = event["label"]
						tmp_change_type = event["change_type"]
						tmp_replay = event["replay"]
						tmp_link = None
						if "link" in event:
							tmp_link = Event(event["link"]["label"], event["link"]["half"], event["link"]["time"], event["link"]["team"], event["link"]["position"], event["link"]["visibility"] )
						event_list.append(Camera(tmp_label, tmp_half, tmp_time, tmp_change_type, tmp_position, tmp_replay, tmp_link))
		return event_list

	def save_file(self, path, half):

		final_list = list()

		if half == 1:
			list_other_half = self.read_json_camera(path,2)
			final_list = self.event_list[::-1] + list_other_half
		else:
			list_other_half = self.read_json_camera(path,1)
			final_list = list_other_half + self.event_list[::-1]


		annotations_dictionary = list()
		for event in final_list:
			tmp_dict = dict()
			tmp_dict["gameTime"] = str(event.half) + " - " + str(event.time)
			tmp_dict["label"] = str(event.label)
			tmp_dict["change_type"] = str(event.change_type)
			tmp_dict["replay"] = str(event.replay)
			tmp_dict["position"] = str(event.position)
			if event.link is not None:
				tmp_dict["link"] = dict()
				tmp_dict["link"]["label"] = str(event.link.label)
				tmp_dict["link"]["half"] = str(event.link.half)
				tmp_dict["link"]["time"] = str(event.link.time)
				tmp_dict["link"]["team"] = str(event.link.team)
				tmp_dict["link"]["position"] = str(event.link.position)
				tmp_dict["link"]["visibility"] = str(event.link.visibility)
			annotations_dictionary.append(tmp_dict)

		data = None
		with open(os.path.dirname(path) + "/Labels-v2.json", 'r') as original_file:
			data = json.load(original_file)
		data["annotations"] = annotations_dictionary

		path_to_save = os.path.dirname(path) + "/Labels-cameras.json"
		with open(path_to_save, "w") as save_file:
			json_data = json.dump(data,save_file, indent=4, sort_keys=True)
