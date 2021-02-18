from PyQt5.QtWidgets import QMainWindow, QWidget, QGridLayout, QListWidget, QHBoxLayout
from PyQt5.QtGui import QPalette
from PyQt5.QtCore import Qt
from utils.event_class import Event, ms_to_time

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
		self.list_widget.clicked.connect(self.clicked)

		for item_nbr, element in enumerate(self.labels):
			self.list_widget.insertItem(item_nbr,element)

		self.list_widget_second = QListWidget()
		self.list_widget_second.clicked.connect(self.clicked)

		for item_nbr, element in enumerate(self.second_labels):
			self.list_widget_second.insertItem(item_nbr,element)

		self.list_widget_third = QListWidget()
		self.list_widget_third.clicked.connect(self.clicked)

		for item_nbr, element in enumerate(self.third_labels):
			self.list_widget_third.insertItem(item_nbr,element)

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
		self.first_label = None
		self.second_label = None

	def clicked(self, qmodelindex):
		print("clicked")

	def set_position(self):
		self.xpos_window = self.main_window.pos().x()+self.main_window.frameGeometry().width()//4
		self.ypos_window = self.main_window.pos().y()+self.main_window.frameGeometry().height()//4
		self.width_window = self.main_window.frameGeometry().width()//2
		self.height_window = self.main_window.frameGeometry().height()//2
		self.setGeometry(self.xpos_window, self.ypos_window, self.width_window, self.height_window)

	def keyPressEvent(self, event):

		if event.key() == Qt.Key_Return:
			if not self.to_second and not self.to_third:
				self.first_label = self.list_widget.currentItem().text()
				self.list_widget_second.setFocus()
				self.to_second=True
			elif self.to_second:
				self.second_label = self.list_widget_second.currentItem().text()
				self.to_second=False
				self.to_third=True
				self.list_widget_third.setFocus()
			elif self.to_third:
				position = self.main_window.media_player.media_player.position()
				self.main_window.list_manager.add_event(Event(self.first_label,self.main_window.half,ms_to_time(position),self.second_label, position, self.list_widget_third.currentItem().text()))
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
				self.main_window.setFocus()



		if event.key() == Qt.Key_Escape:
			self.to_second=False
			self.to_third=False
			self.first_label = None	
			self.second_label = None
			self.list_widget_second.setCurrentRow(-1)
			self.list_widget_third.setCurrentRow(-1)
			self.hide()
			self.main_window.setFocus()