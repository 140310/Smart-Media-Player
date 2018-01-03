
#from pyannote.core import Annotation
import pyannote.core.json

from structure.video import Video
from face import Face
from pyannote.video import FaceTracking

from pandas import read_table

from six.moves import zip
import numpy as np
import cv2

import dlib


MIN_OVERLAP_RATIO = 0.5
MIN_CONFIDENCE = 10.
MAX_GAP = 1.

FACE_TEMPLATE = ('{t:.3f} {identifier:d} '
                 '{left:.3f} {top:.3f} {right:.3f} {bottom:.3f} '
                 '{status:s}\n')


def getFaceGenerator(tracking, frame_width, frame_height, double=True):
    """Parse precomputed face file and generate timestamped faces"""

    # load tracking file and sort it by timestamp
    names = ['t', 'track', 'left', 'top', 'right', 'bottom', 'status']
    dtype = {'left': np.float32, 'top': np.float32,
             'right': np.float32, 'bottom': np.float32}
    tracking = read_table(tracking, delim_whitespace=True, header=None,
                          names=names, dtype=dtype)
    tracking = tracking.sort_values('t')

    # t is the time sent by the frame generator
    t = yield

    rectangle = dlib.drectangle if double else dlib.rectangle

    faces = []
    currentT = None

    for _, (T, identifier, left, top, right, bottom, status) in tracking.iterrows():

        left = int(left * frame_width)
        right = int(right * frame_width)
        top = int(top * frame_height)
        bottom = int(bottom * frame_height)

        face = rectangle(left, top, right, bottom)

        # load all faces from current frame and only those faces
        if T == currentT or currentT is None:
            faces.append((identifier, face, status))
            currentT = T
            continue

        # once all faces at current time are loaded
        # wait until t reaches current time
        # then returns all faces at once

        while True:

            # wait...
            if currentT > t:
                t = yield t, []
                continue

            # return all faces at once
            t = yield currentT, faces

            # reset current time and corresponding faces
            faces = [(identifier, face, status)]
            currentT = T
            break

    while True:
        t = yield t, []


def getLandmarkGenerator(shape, frame_width, frame_height):
    """Parse precomputed shape file and generate timestamped shapes"""

    # load landmarks file
    shape = read_table(shape, delim_whitespace=True, header=None)

    # deduce number of landmarks from file dimension
    _, d = shape.shape
    n_points = (d - 2) / 2

    # t is the time sent by the frame generator
    t = yield

    shapes = []
    currentT = None

    for _, row in shape.iterrows():

        T = float(row[0])
        identifier = int(row[1])
        landmarks = np.float32(list(pairwise(
            [coordinate for coordinate in row[2:]])))
        landmarks[:, 0] = np.round(landmarks[:, 0] * frame_width)
        landmarks[:, 1] = np.round(landmarks[:, 1] * frame_height)

        # load all shapes from current frame
        # and only those shapes
        if T == currentT or currentT is None:
            shapes.append((identifier, landmarks))
            currentT = T
            continue

        # once all shapes at current time are loaded
        # wait until t reaches current time
        # then returns all shapes at once

        while True:

            # wait...
            if currentT > t:
                t = yield t, []
                continue

            # return all shapes at once
            t = yield currentT, shapes

            # reset current time and corresponding shapes
            shapes = [(identifier, landmarks)]
            currentT = T
            break

    while True:
        t = yield t, []

        

def track(video, shot, output,
          detect_min_size=0.0,
          detect_every=0.0,
          track_min_overlap_ratio=MIN_OVERLAP_RATIO,
          track_min_confidence=MIN_CONFIDENCE,
          track_max_gap=MAX_GAP):
    """Tracking by detection"""

    tracking = FaceTracking(detect_min_size=detect_min_size,
                            detect_every=detect_every,
                            track_min_overlap_ratio=track_min_overlap_ratio,
                            track_min_confidence=track_min_confidence,
                            track_max_gap=track_max_gap)

    with open(shot, 'r') as fp:
        shot = pyannote.core.json.load(fp)

    #if isinstance(shot, Annotation):
    #    print "Annotation called"
    #    shot = shot.get_timeline()

    with open(output, 'w') as foutput:

        for identifier, track in enumerate(tracking(video, shot)):

            for t, (left, top, right, bottom), status in track:

                foutput.write(FACE_TEMPLATE.format(
                    t=t, identifier=identifier, status=status,
                    left=left, right=right, top=top, bottom=bottom))

            foutput.flush()

def extract(video, landmark_model, embedding_model, tracking, landmark_output, embedding_output):
    """Facial features detection"""

    # face generator
    frame_width, frame_height = video.frame_size
    faceGenerator = getFaceGenerator(tracking,
                                     frame_width, frame_height,
                                     double=False)
    faceGenerator.send(None)

    face = Face(landmark_model,embedding_model)

    with open(landmark_output, 'w') as flandmark, \
        open(embedding_output, 'w') as fembedding:

        for timestamp, rgb in video:

            # get all detected faces at this time
            T, faces = faceGenerator.send(timestamp)
            # not that T might be differ slightly from t
            # due to different steps in frame iteration

            for identifier, bounding_box, _ in faces:

                landmarks = face.get_landmarks(rgb, bounding_box)
                embedding = face.get_embedding(rgb, landmarks)

                flandmark.write('{t:.3f} {identifier:d}'.format(
                    t=T, identifier=identifier))
                for p in landmarks.parts():
                   x, y = p.x, p.y
                   flandmark.write(' {x:.5f} {y:.5f}'.format(x=x / frame_width,
                                                           y=y / frame_height))
                flandmark.write('\n')

                fembedding.write('{t:.3f} {identifier:d}'.format(
                    t=T, identifier=identifier))
                for x in embedding:
                    fembedding.write(' {x:.5f}'.format(x=x))
                fembedding.write('\n')

            flandmark.flush()
            fembedding.flush()


