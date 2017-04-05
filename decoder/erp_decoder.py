class ERPDecoder(object):

    def add_letters(self):
        pass

    def new_trial(self):
        pass

    def add_iteration(self,x,s):
        pass

    def predict_last_trial(self):
        pass

    def predict_all_trials(self):
        pass


class SupervisedERPDecoder(ERPDecoder):
    def train(self,x,y):
        raise NotImplementedError("Subclass responsability")


class AdaptiveERPDecoder(ERPDecoder):
    def update_decoder(self):
        raise NotImplementedError("Subclass responsability")


class UnsupervisedERPDecoder(AdaptiveERPDecoder):
    def __init__(self):
        pass



class LDAERPDecoder(SupervisedERPDecoder):
    pass
