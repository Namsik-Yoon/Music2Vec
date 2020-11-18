import os
import tarfile
if 'data' not in os.listdir():
    os.mkdir('data')
    os.mkdir('data/genres')
if 'blues' not in os.listdir('data/genres'):
    my_tar = tarfile.open('genres.tar.gz')
    my_tar.extractall('data/') # specify which folder to extract to
    my_tar.close()
print('complete unzip')