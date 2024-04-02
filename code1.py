import os
import pickle

actors = os.listdir('data1')

# print(actors)

filenames = []

for actor in actors:
    for file in os.listdir(os.path.join('data1', actor)):
        filenames.append(os.path.join('data1', actor, file))

# print(filenames)

pickle.dump(filenames, open('filenames.pkl', 'wb'))
