#Data Preprocessing
import cv2
class Box(object):
	def __init__(self, x, y, w, h, label):
		self.label = label;
		self.x = x;
		self.y = y;
		self.h = h;
		self.w = w;

	def overlap(b1, b2):
		x1max = b1.x+0.5*b1.w
		x2max = b2.x+0.5*b2.w
		x1min = b1.x-0.5*b1.w
		x2min = b2.x-0.5*b2.w
		y1max = b1.y+0.5*b1.h
		y2max = b2.y+0.5*b2.h
		right = min(x1max, x2max)
		left = max(x1min, x2min)
		top = min(y1max, y1max)
		bot = max(y1min, y2min)

		if (right-left>0) and (top-bot>0):
			return (right-left)*(top-bot)
		else:
			return 0

def preprocessing(infos):
	CELL_SIZE = 64
	NUM_CELL = 49
	NUM_CLASSES = 20
	x = []
	y = []
	w = []
	h = []
#	cell_x = []
#	cell_y = []
	p = []
	objects = sorted(['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor'])
	#bound_areas = []
 	#intersect_areas= []
	#total_area = []
	c = [1]*len(infos)
	objs = []
	images = []
	for img in infos:
		image = cv2.imread(img['file'])
		image = cv2.resize(image, (448, 448))
		images += [image]

		h_org = img['height']
		w_org = img['width']

		x_center = []
		y_center = []
		width = []
		height = []
		#box = []
		#overlaps = [] #intersecting areas
		#areas = [] #
		#total_area = 0
		p_cond = []
		label = []
#		cell_num_x = []
#		cell_num_y = []

		for obj in img['object']:

			x_start = obj['xmin']/w_org*448
			x_end = obj['xmax']/w_org*448
			y_start = obj['ymin']/h_org*448
			y_end = obj['ymax']/h_org*448

			x_temp = int((x_start+x_end)/2)
			y_temp = int((y_start+y_end)/2)
			w_temp = int(x_end - x_start)
			h_temp = int(y_end - y_start)


#			cell_num_x.append(range(x_start,x_end+1))
#			cell_num_y.append(range(y_start,y_end+1))

#			x_temp = 0.5*(obj['xmax']+obj['xmin'])
#			y_temp = 0.5*(obj['ymax']+obj['ymin'])
#			w_temp = (obj['xmax']-obj['xmin'])
#			h_temp = (obj['ymax']-obj['ymin'])
			label_temp = obj['name']
			label.append(label_temp)
			#box += Box(x_temp, y_temp, w_temp, h_temp, label_temp)
			#area_temp = w_temp*h_temp #bounding box area
#			x_temp = (x_temp % CELL_SIZE)/CELL_SIZE
#			y_temp = (y_temp % CELL_SIZE)/CELL_SIZE
#			w_temp /= 448
#			h_temp /= 448
			x_center.append(x_temp)
			y_center.append(y_temp)
			width.append(w_temp)
			height.append(h_temp)
			temp = [0]*NUM_CLASSES
		
			for index in range(0,len(objects)):
				if label_temp == objects[index]:
					temp[index] = 1
			p_cond.append(temp)					

		x.append(x_center)
		y.append(y_center)
		w.append(width)
		h.append(height)
		p.append(p_cond)
#		cell_x.append(cell_num_x)
#		cell_y.append(cell_num_y)
		#box = []
		#overlaps = [] #intersecting areas
		#areas = [] #
		#total_area = 0

		#bound_areas += [areas]
		#intersect_areas += [overlaps]
		#total_area += [np.sum(areas) - np.sum(overlaps)] #total area of the bounding boxes
		#objs += [labels]

	return image, x, y, w, h, p, c

def sum_overlap(boxes):
	for i in len(boxes):
		for j in len(box):
				if i<j:
					overlaps += [overlap(box[i], box[j])]