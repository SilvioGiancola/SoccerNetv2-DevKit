from PyQt5.QtWidgets import QWidget, QPushButton, QStyle, QSlider, QHBoxLayout, QVBoxLayout, QFileDialog, QGridLayout, QListWidget
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl

class ListDisplay(QWidget):

	def __init__(self, main_window):
		super().__init__()

		self.max_width = 300
		self.setMaximumWidth(self.max_width)

		self.main_window = main_window

		self.layout = QGridLayout()
		self.setLayout(self.layout)

		self.list_widget = QListWidget()
		self.list_widget.clicked.connect(self.clicked)
		self.list_widget.itemDoubleClicked.connect(self.doubleClicked)

		self.layout.addWidget(self.list_widget)

	def clicked(self, qmodelindex):
		item = self.list_widget.currentItem()

	def doubleClicked(self, item):
		row = self.list_widget.currentRow()
		position = self.main_window.list_manager.event_list[row].position
		if self.main_window.media_player.play_button.isEnabled():
			self.main_window.media_player.set_position(position)
		self.main_window.setFocus()


	def display_list(self, list_to_display):
		self.list_widget.clear()
		for item_nbr, element in enumerate(list_to_display):
			self.list_widget.insertItem(item_nbr,element)