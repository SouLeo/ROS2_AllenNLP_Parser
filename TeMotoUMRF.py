from allennlp.predictors.semantic_role_labeler import SemanticRoleLabelerPredictor

class TemotoUMRF:
    """A class that leverages the AllenNLP SRL model to build Temoto Universal Meaning Representation Format (tUMRF) objects"""
    def __init__(self, srl_model_path):
        self.ros_topic_name = ""
        self.srl_model_path = srl_model_path
        self.srl_model = SemanticRoleLabelerPredictor.from_path(self.srl_model_path)

    def predict_descriptors(self, input_sentence):
        desc = self.srl_model.predict(input_sentence)
        return desc 

    def create_tumrf(self, verb, word_list):
        print(verb)
        tags = verb["tags"]
        # if we cared about agents/causers we would parse for-ARG0 too
        # see English Propbank annotation guidelines for descriptions of all tags
        args = [tags.index(i) for i in tags if "ARG1" in i] # only reports objects of transitive verbs
        verb = [tags.index(i) for i in tags if "-V" in i]

        tumrf = {"notation": "TeMoto SFT", "action_identifier": word_list[verb[0]]}

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
