from allennlp.predictors.predictor import Predictor

class TemotoUMRF:
    """A class that leverages the AllenNLP SRL model to build Temoto Universal Meaning Representation Format (tUMRF) objects"""
    def __init__(self, openIE_model_path):
        self.ros_topic_name = ""
        self.openIE_model_path = openIE_model_path
        self.openIE_model = Predictor.from_path(self.openIE_model_path)

    def predict_descriptors(self, input_sentence):
        desc = self.openIE_model.predict(input_sentence)
        return desc 

    def create_tumrf(self, verb, word_list):
        print(verb)
        tags = verb["tags"]
        # if we cared about agents/causers we would parse for-ARG0 too
        # see English Propbank annotation guidelines for descriptions of all tags
        arg1s = [tags.index(i) for i in tags if "ARG1" in i] # only reports objects of transitive verbs
        verb = [tags.index(i) for i in tags if "-V" in i]

        arg0s = [tags.index(i) for i in tags if "ARG0" in i]
        # print(arg0s) 
        # arg0 is usually the agent that is causing change in the world; however, for the "move"
        # action, turn, rotate, and left were issues within our known vocabulary. This catch is for those cases.
        if arg0s and word_list[arg0s[0]] == "turn" or arg0s and word_list[arg0s[0]] == "rotate":
                verb.append(arg0s[0])
                verb.pop(0)
                arg1s.append(arg0s[0]+1)

        # TODO: insert NLTK wordnet search for synonyms to match similar words to finite action list

        tumrf = {"notation": "TeMoto SFT", "action_identifier": word_list[verb[0]]}
        
        if arg1s:
            tumrf.update({"input_parameters": word_list[arg1s[0]]})
 
        print(tumrf)

    def create_tumrfs(self, desc):
        # 1) parse the incoming tagged words
        # 2) separate into 
        #    i) input params
        #    ii) output params
        #    iii) suffix - num which differentiates multiple instances of actions (?)
        #    iv) effect - does action terminate after main execution is finished
        #    v) parents
        #    vi) children
        for verb in desc["verbs"]:
            self.create_tumrf(verb, desc["words"])
