
# coding: utf-8

# In[1]:


#get_ipython().magic(u'load_ext autoreload')
#get_ipython().magic(u'autoreload 2')

import pyannote.core.json
from structure.video import Video
from structure.shot import Shot
import faceController as fc
from face.clustering import FaceClustering
from pyannote.core import Timeline
import numpy as np
from PIL import Image
from scipy import misc
from pandas import read_table


# In[2]:


def do_shot(video, output, height=50, window=2.0, threshold=1.0):
    shots = Shot(video, height=height, context=window, threshold=threshold)
    shots = Timeline(shots)
    with open(output, 'w') as fp:
        pyannote.core.json.dump(shots, fp)




# In[3]:

filename = "thief.mp4"
shotOutput ="./extra/"+filename.split(".")[0]+"Shots.json"
landmark_model = "./models/shape_predictor_68_face_landmarks.dat"
embedding_model = "./models/dlib_face_recognition_resnet_model_v1.dat"
landmarks = "./extra/"+filename.split(".")[0]+"Landmarks.txt"
tracking =  "./extra/"+filename.split(".")[0]+"Tracking.txt"
embeddings = "./extra/"+filename.split(".")[0]+"Embedding.txt"

# In[4]:

video = Video("./video/"+filename)
do_shot(video, shotOutput)
video = Video("./video/"+filename)
fc.track(video, shotOutput, tracking,detect_every=0.5)
fc.extract(video, landmark_model,embedding_model, tracking, landmarks,embeddings)
clustering = FaceClustering(threshold=0.6)
face_tracks, embeddings = clustering.model.preprocess(embeddings)
result = clustering(face_tracks, features=embeddings)




# In[15]:


track=np.loadtxt(tracking,delimiter=" ",dtype=str)
tempArray=[]
for segment, track_id, cluster in result.itertracks(yield_label=True):
    tempArray.append([segment,track_id, cluster])
tempArray=np.array(tempArray)


# In[16]:


video = Video("./video/"+filename)


# In[17]:


actorBasedShotArray=[]
for u in np.unique(tempArray[:,2]):
    actorBasedShotArray.append(tempArray[tempArray[:,2]==u])


# In[18]:


names = ['t', 'track', 'left', 'top', 'right', 'bottom', 'status']
dtype = {'left': np.float32, 'top': np.float32,
         'right': np.float32, 'bottom': np.float32}
tracking1 = read_table(tracking, delim_whitespace=True, header=None,
                      names=names, dtype=dtype)
tracking1 = tracking1.sort_values('t')
del tracking1["status"]
del tracking1["track"]
tracking1=np.array(tracking1)


# In[19]:



uniqueActors=len(actorBasedShotArray)
frame_width, frame_height = video.frame_size
arraytosave=[]
for i in range(uniqueActors):
    startTime=actorBasedShotArray[i][0][0].start
    image=video._get_frame(startTime)
    faceDimension=tracking1[tracking1[:,0]==startTime][0]

    left = int(faceDimension[1] * frame_width)
    right = int(faceDimension[3] * frame_width)
    top = int(faceDimension[2] * frame_height)
    bottom = int(faceDimension[4] * frame_height)
    Image.fromarray(misc.imresize(image[top:bottom,left:right,:], (180, 180), interp='bilinear'),'RGB').save("./output/actor"+filename.split(".")[0]+str(i)+".jpg")
    arraytosave.append([])
    arraytosave[-1].append("actor"+filename.split(".")[0]+str(i))
    for j in actorBasedShotArray[i]:
        arraytosave[-1].append(str(j[0].start)+"*"+str(j[0].end))


# In[20]:


outputFile=open("./output/"+filename.split(".")[0]+"output.txt",'w')
for item in arraytosave:
    print item
    outputFile.write("%s\n"%item)

