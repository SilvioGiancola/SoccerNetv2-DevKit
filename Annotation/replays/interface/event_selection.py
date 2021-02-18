from PyQt5.QtWidgets import QMainWindow, QWidget, QGridLayout, QListWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QPalette
from PyQt5.QtCore import Qt
from utils.event_class import Event, Camera, ms_to_time
from interface.loop_player import LoopPlayer
from utils.list_management import ListManager
import os

class EventSelectionWindow(QMainWindow):
	def __init__(self, main_window):
		super().__init__()

		self.main_window = main_window

		# Defining some variables of the window
		self.title_window = "Event Selection"

		# Setting the window appropriately
		self.setWindowTitle(self.title_window)
		self.set_position()

		self.palette_main_window = self.palette()
		self.palette_main_window.setColor(QPalette.Window, Qt.black)

		# Initiate the sub-widgets
		self.init_window()

	def init_window(self):

		# Read the available labels
		self.labels = list()
		with open('../config/classes.txt') as file:
			for cnt, line in enumerate(file):
				self.labels.append(line.rstrip())

		# Read the available second labels
		self.second_labels = list()
		with open('../config/second_classes.txt') as file:
			for cnt, line in enumerate(file):
				self.second_labels.append(line.rstrip())

		# Read the available third labels
		self.third_labels = list()
		with open('../config/third_classes.txt') as file:
			for cnt, line in enumerate(file):
				self.third_labels.append(line.rstrip())

		self.list_widget = QListWidget()

		for item_nbr, element in enumerate(self.labels):
			self.list_widget.insertItem(item_nbr,element)

		self.list_widget_second = QListWidget()

		for item_nbr, element in enumerate(self.second_labels):
			self.list_widget_second.insertItem(item_nbr,element)

		self.list_widget_third = QListWidget()

		for item_nbr, element in enumerate(self.third_labels):
			self.list_widget_third.insertItem(item_nbr,element)

		#TBD
		self.path_video_first_half = None
		self.path_video_second_half = None 
		self.path_labels_event = None
		self.list_events = list()

		self.list_widget_fourth = None

		self.loop_player = LoopPlayer(self.main_window)
		self.loop_duration = 10
		self.video_display = QWidget(self)
		self.video_display.setLayout(self.loop_player.layout)
		self.video_display.setFixedWidth(398)
		self.video_display.setFixedHeight(224)

		# Layout the different widgets
		central_display = QWidget(self)
		self.setCentralWidget(central_display)
		final_layout = QHBoxLayout()
		final_layout.addWidget(self.list_widget)
		final_layout.addWidget(self.list_widget_second)
		final_layout.addWidget(self.list_widget_third)
		central_display.setLayout(final_layout)

		self.to_second = False
		self.to_third = False
		self.to_fourth = False
		self.first_label = None
		self.second_label = None
		self.third_label = None
		self.fourth_label = None


	def init_replay_loop(self):

		self.path_video_first_half = os.path.dirname(self.main_window.media_player.path_label) + "/1.mkv"
		self.path_video_second_half = os.path.dirname(self.main_window.media_player.path_label) + "/2.mkv"
		self.path_labels_event = os.path.dirname(self.main_window.media_player.path_label) + "/Labels-v2.json"

		self.event_list_manager_first_half = ListManager()
		self.event_list_manager_second_half = ListManager()

		self.event_list_manager_first_half.event_list = self.event_list_manager_first_half.read_json_event(self.path_labels_event, 1)
		self.event_list_manager_second_half.event_list = self.event_list_manager_second_half.read_json_event(self.path_labels_event, 2)
		self.event_list_manager_first_half.sort_list()
		self.event_list_manager_second_half.sort_list()

		self.tmp_list_event = list()
		self.tmp_list_filepath = list()


		self.list_widget_fourth = QListWidget()
		self.list_widget_fourth.clicked.connect(self.doubleClicked)
		self.list_widget_fourth.itemDoubleClicked.connect(self.doubleClicked)
		self.list_widget_fourth.currentItemChanged.connect(self.doubleClicked)
		self.list_widget_fourth.setFixedWidth(398)

		# Layout the different widgets
		central_display = QWidget(self)
		self.setCentralWidget(central_display)

		final_layout = QVBoxLayout()
		horizontal_layout = QHBoxLayout()
		horizontal_layout.addWidget(self.list_widget)
		horizontal_layout.addWidget(self.list_widget_second)
		horizontal_layout.addWidget(self.list_widget_third)
		horizontal_layout.addWidget(self.list_widget_fourth)
		final_layout.addLayout(horizontal_layout)
		final_layout.addWidget(self.video_display)
		central_display.setLayout(final_layout)

	def update_replay_list(self, position, half):
		self.list_widget_fourth.clear()

		self.tmp_list_event = list()
		self.tmp_list_filepath = list()

		path_dir = os.path.dirname(self.main_window.media_player.path_label)

		if half == 1:
			for item_nbr, element in enumerate(self.event_list_manager_first_half.event_list):
				if element.position <= position:
					self.list_widget_fourth.insertItem(item_nbr,element.to_text())
					self.tmp_list_event.append(element)
					self.tmp_list_filepath.append(path_dir + "/1.mkv")

		if half == 2:
			counter = 0
			for item_nbr, element in enumerate(self.event_list_manager_second_half.event_list):
				if element.position <= position:
					self.list_widget_fourth.insertItem(item_nbr,element.to_text())
					self.tmp_list_event.append(element)
					self.tmp_list_filepath.append(path_dir + "/2.mkv")
					counter += 1 
			for item_nbr, element in enumerate(self.event_list_manager_first_half.event_list):
				self.list_widget_fourth.insertItem(item_nbr+counter,element.to_text())
				self.tmp_list_event.append(element)
				self.tmp_list_filepath.append(path_dir + "/1.mkv")

	def doubleClicked(self, qmodelindex):

		if self.list_widget_fourth.currentRow() >= 0:

			video_path = self.tmp_list_filepath[self.list_widget_fourth.currentRow()]

			position = self.tmp_list_event[self.list_widget_fourth.currentRow()].position

			position_start = position-self.loop_duration//2*1000
			position_end = position+self.loop_duration//2*1000

			self.loop_player.open_file(video_path, position_start, position_end)
		else:
			self.loop_player.media_player.pause()

	def set_position(self):
		self.xpos_window = self.main_window.pos().x()+self.main_window.frameGeometry().width()//4
		self.ypos_window = self.main_window.pos().y()+self.main_window.frameGeometry().height()//4
		self.width_window = self.main_window.frameGeometry().width()//1.5
		self.height_window = self.main_window.frameGeometry().height()//1.2
		self.setGeometry(self.xpos_window, self.ypos_window, self.width_window, self.height_window)

	def keyPressEvent(self, event):

		if event.key() == Qt.Key_Return:
			if not self.to_second and not self.to_third and not self.to_fourth:
				self.first_label = self.list_widget.currentItem().text()
				self.list_widget_second.setFocus()
				self.to_second=True
			elif self.to_second:
				self.second_label = self.list_widget_second.currentItem().text()
				self.to_second=False
				self.to_third=True
				self.list_widget_third.setFocus()
			elif self.to_third:
				self.third_label = self.list_widget_third.currentItem().text()
				if self.third_label != "replay":
					position = self.main_window.media_player.media_player.position()
					self.main_window.list_manager.add_event(Camera(self.first_label,self.main_window.half,ms_to_time(position),self.second_label, position, self.third_label ))
					self.main_window.list_display.display_list(self.main_window.list_manager.create_text_list())
					self.first_label = None
					self.second_label = None
					self.to_third=False
					# Save
					path_label = self.main_window.media_player.get_last_label_file()
					self.main_window.list_manager.save_file(path_label, self.main_window.half)
					self.hide()
					self.list_widget_second.setCurrentRow(-1)
					self.list_widget_third.setCurrentRow(-1)
					self.list_widget_fourth.setCurrentRow(-1)
					self.main_window.setFocus()
				else:
					self.to_third=False
					self.to_fourth=True
					self.list_widget_fourth.setFocus()
			elif self.to_fourth:
				self.fourth_label = self.list_widget_fourth.currentRow()
				position = self.main_window.media_player.media_player.position()
				self.main_window.list_manager.add_event(Camera(self.first_label,self.main_window.half,ms_to_time(position),self.second_label, position, self.third_label, self.tmp_list_event[self.fourth_label] ))
				self.main_window.list_display.display_list(self.main_window.list_manager.create_text_list())
				self.first_label = None
				self.second_label = None
				self.third_label = None
				self.fourth_label = None
				self.to_third=False
				self.to_fourth=False
				# Save
				path_label = self.main_window.media_player.get_last_label_file()
				self.main_window.list_manager.save_file(path_label, self.main_window.half)
				self.hide()
				self.list_widget_second.setCurrentRow(-1)
				self.list_widget_third.setCurrentRow(-1)
				self.list_widget_fourth.setCurrentRow(-1)
				self.main_window.setFocus()



		if event.key() == Qt.Key_Escape:
			self.to_second=False
			self.to_third=False
			self.to_fourth=False
			self.first_label = None	
			self.second_label = None
			self.third_label = None
			self.list_widget_second.setCurrentRow(-1)
			self.list_widget_third.setCurrentRow(-1)
			self.list_widget_fourth.setCurrentRow(-1)
			self.hide()
			self.main_window.setFocus()