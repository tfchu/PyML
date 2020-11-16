'''
json
{'image': file_name, 'label': label}
'''
# face detection with mtcnn on a photograph
import json, os
import matplotlib
from matplotlib import pyplot
from matplotlib.widgets import RadioButtons, Button
from mtcnn.mtcnn import MTCNN

class Tag_Face():
    def __init__(self, dirpath):
        self.facelist = []
        self.dirpath = dirpath
        self.file_counter = 0
        # radio button
        self.rvalue = 0             # default
        self.roptions = ('Tony', 'Alice', 'Derek', 'Darren', 'Others')
        self.rdict = {'Tony': 0, 'Alice': 1, 'Derek': 2, 'Darren': 3, 'Others': 4}   # radio button dict
        # DNN
        # detect faces in the image, e.g. 
        # [{'box': [389, 378, 223, 288], 'confidence': 0.9999948740005493, 'keypoints': {'left_eye': (469, 493), 'right_eye': (569, 497), 'nose': (521, 550), 'mouth_left': (467, 587), 'mouth_right': (555, 593)}}]
        self.detector = MTCNN()

    def tag_faces(self):
        for file in os.listdir(self.dirpath):
            if file.endswith('jpg'):
                self.pixels = pyplot.imread(os.path.join(self.dirpath, file))
                self.faces = self.detector.detect_faces(self.pixels)
                self._tag_single_face()

    def _tag_single_face(self):
        def move_figure(f, x, y):
            """Move figure's upper left corner to pixel (x, y)"""
            backend = matplotlib.get_backend()
            if backend == 'TkAgg':
                f.canvas.manager.window.wm_geometry("+%d+%d" % (x, y))
            elif backend == 'WXAgg':
                f.canvas.manager.window.SetPosition((x, y))
            else:
                # This works for QT and GTK
                # You can also use window.setGeometry
                f.canvas.manager.window.move(x, y)
        def set_entry(label):
            self.rvalue = self.rdict[label]
        def save_entry(event):
            # image file (feature) and label
            filename = str(self.file_counter) + '.jpg'
            fdict = {
                'image': filename, 
                'label': self.rvalue
            }
            self.facelist.append(fdict)
            # extract ROI and save as jpg
            extent = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
            self.fig.savefig(filename, bbox_inches=extent)
            pyplot.close()
            # reset radio button to default
            self.rvalue = 0
            # increase counter
            self.file_counter += 1

        for face in self.faces:
            # get face location
            x1, y1, width, height = face['box']
            x2, y2 = x1 + width, y1 + height
            # plot face
            self.fig, self.ax = pyplot.subplots()
            self.ax.axis('off')						                # after subplot to remove axis
            self.ax.imshow(self.pixels[y1:y2, x1:x2])		        # display in pyplot
            # add radio button
            pyplot.subplots_adjust(left=0.2)
            rax = pyplot.axes([0.1, 0.5, 0.15, 0.15], facecolor='lightgoldenrodyellow')
            radio = RadioButtons(rax, self.roptions)
            radio.on_clicked(set_entry)
            # add confirm button
            bax = pyplot.axes([0.1, 0.40, 0.15, 0.075], facecolor='lightgoldenrodyellow')
            but = Button(bax, 'Confirm')
            but.on_clicked(save_entry)
            # reposition figure window
            move_figure(self.fig, 100, 100)
            # show image
            pyplot.show()

    def save(self):
        with open('face_tags.json', 'w') as tagf:
            tagf.write(json.dumps(self.facelist, indent=4))
            print('json saved')

tf = Tag_Face('../images')
tf.tag_faces()
tf.save()