# Adapted from https://codeloop.org/python-how-to-create-media-player-in-pyqt5/
import os
from PyQt5.QtWidgets import QWidget, QPushButton, QStyle, QSlider, QHBoxLayout, QVBoxLayout, QFileDialog
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtCore import Qt, QUrl

class LoopPlayer(QWidget):

	def __init__(self, main_window):

		# Defining the elements of the media player
		super().__init__()

		self.main_window = main_window

		# Media Player
		self.media_player = QMediaPlayer(None, QMediaPlayer.VideoSurface)

		# Video Widget
		self.video_widget = QVideoWidget()

		# Button to open a new file
		self.open_file_button = QPushButton('Open video')
		self.open_file_button.clicked.connect(self.open_file)

		# Button for playing the video
		self.play_button = QPushButton()
		self.play_button.setEnabled(False)
		self.play_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
		self.play_button.clicked.connect(self.play_video)

		# Button for the slider
		self.slider = QSlider(Qt.Horizontal)
		self.slider.setRange(0,0)
		self.slider.sliderMoved.connect(self.set_position)

		#create hbox layout
		hboxLayout = QHBoxLayout()
		hboxLayout.setContentsMargins(0,0,0,0)

		#set widgets to the hbox layout
		hboxLayout.addWidget(self.open_file_button)
		hboxLayout.addWidget(self.play_button)
		hboxLayout.addWidget(self.slider)

		#create vbox layout
		self.layout = QVBoxLayout()
		self.layout.addWidget(self.video_widget)
		self.layout.addLayout(hboxLayout)

		self.media_player.setVideoOutput(self.video_widget)


		# Media player signals
		self.media_player.stateChanged.connect(self.mediastate_changed)
		self.media_player.positionChanged.connect(self.position_changed)
		self.media_player.durationChanged.connect(self.duration_changed)

		self.path_label = None

		self.position_start = 0
		self.position_end = 1


	def open_file(self, filename, position_start, position_end):

		if filename != '':
			self.media_player.setMedia(QMediaContent(QUrl.fromLocalFile(filename)))
			self.play_button.setEnabled(True)
			filpath = os.path.basename(filename)
			self.play_video()
			self.position_start = position_start
			self.position_end = position_end
			self.set_position(self.position_start)


	def play_video(self):
		if self.media_player.state() == QMediaPlayer.PlayingState:
			self.media_player.pause()

		else:
			self.media_player.play()


	def mediastate_changed(self, state):
		if self.media_player.state() == QMediaPlayer.PlayingState:
			self.play_button.setIcon(
				self.style().standardIcon(QStyle.SP_MediaPause)

			)

		else:
			self.play_button.setIcon(
				self.style().standardIcon(QStyle.SP_MediaPlay)

			)

	def position_changed(self, position):
		if position > self.position_end:
			self.set_position(self.position_start)
		self.slider.setValue(position)


	def duration_changed(self, duration):
		self.slider.setRange(0, duration)


	def set_position(self, position):
		self.media_player.setPosition(position)


	def handle_errors(self):
		self.play_button.setEnabled(False)
		print("Error: " + self.media_player.errorString())
