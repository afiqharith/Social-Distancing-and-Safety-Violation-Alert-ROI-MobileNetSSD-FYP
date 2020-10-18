class Config:

	def get(VIDEONAME):

		if VIDEONAME == 'TownCentre.mp4':
			distance = 60
			threshold = 0.2
		
		if VIDEONAME == 'PETS2009.mp4':
			distance = 90
			threshold = 0.2	

		if VIDEONAME == 'VIRAT.mp4':
			distance = 55
			threshold = 0.25

		return threshold, distance


