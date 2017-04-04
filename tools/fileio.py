import pickle

def store(object,filename):
    with open(filename,'wb') as my_file:
        pickle.dump(object,my_file)

def load(filename):
    with open(filename,'rb') as my_file:
        object = pickle.load(my_file)
    return object