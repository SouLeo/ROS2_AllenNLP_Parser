import json

from allennlp.predictors.semantic_role_labeler import SemanticRoleLabelerPredictor
from word2number import w2n

class TemotoUMRF:
    """A class that leverages the AllenNLP SRL model to build Temoto Universal Meaning Representation Format (tUMRF) objects"""
    def __init__(self, srl_model_path):
        self.ros_topic_name = ""
        self.srl_model_path = srl_model_path
        self.srl_model = SemanticRoleLabelerPredictor.from_path(self.srl_model_path)
        self.is_dir_left = False

    def predict_descriptors(self, input_sentence):
        # Hack below because left is not recognized as a direction
        # in allennlp, but right is :)
        sentence_tokens = input_sentence.split()
        if 'left' in input_sentence:
            # print('left detected')
            self.is_dir_left = True
            input_sentence = input_sentence.replace('left', 'right')
            # print(input_sentence)
        desc = self.srl_model.predict(input_sentence)
        return desc 

    def find_arg_extent(self, tags):
        arg_extent = []
        for i in range(len(tags)):
            if "ARGM-EXT" in tags[i]:
                arg_extent.append(i)
        # print(arg_extent)
        return arg_extent

    def find_arg_direction(self, tags):
        arg_dir = []
        for i in range(len(tags)):
            if "ARGM-DIR" in tags[i]:
                arg_dir.append(i)
        # print(arg_dir)
        return arg_dir
    
    def find_arg_mnr(self, tags):
        arg_mnr = []
        for i in range(len(tags)):
            if "ARGM-MNR" in tags[i]:
                arg_mnr.append(i)
        # print(arg_mnr)
        return arg_mnr

    def create_tumrf(self, verb, word_list):
        # TODO: Make this damn function less loooooong
        # print(verb)
        tags = verb["tags"]
        # print(tags)
        verb_token = verb["verb"]
        # print(verb)
        # desc = verb["description"]
        # print(desc)
        arg_extent = self.find_arg_extent(tags)
        arg_direction = self.find_arg_direction(tags)
        arg_manner = self.find_arg_mnr(tags)  

        if verb_token:
            verb_pvf = {'verb':{'pvf_type':'string', 'pvf_value':verb_token}}
            # print(verb_pvf)

        if arg_direction:
            # create a direction pvf
            # print("direction label")
            # print(arg_direction)
            
            for i in range(len(arg_direction)):
                direction = word_list[arg_direction[i]] 
                # print(direction) 
                # Hack continuation for left turns
                if direction == 'right' and self.is_dir_left:
                    # fill in left as direction instead of right
                    direction = 'left'
                    self.is_dir_left = False
                dir_pvf = {'direction':{'pvf_type':'string', 'pvf_value':direction}}
                # print(dir_pvf)

        if arg_extent:
            # create num val
            # print("extent label")
            # print(verb)
            is_integer = []
            arg_extent_tokens = []
            for i in range(len(arg_extent)):
                arg_extent_tokens.append(word_list[arg_extent[i]])
                try:
                    if isinstance(w2n.word_to_num(word_list[arg_extent[i]]),int):
                        is_integer.append(i)
                except ValueError:
                    pass
            
            if len(is_integer) == 1:
                num_alpha = w2n.word_to_num(arg_extent_tokens[is_integer[0]])

            else:
                str_concat = ''
                for i in range(len(is_integer)):
                    str_concat = str_concat + ' ' + arg_extent_tokens[is_integer[i]]  
                num_alpha = w2n.word_to_num(str_concat)
            # print(num_alpha)
            # TODO: Tack on unit of measurement as param
            dis_ext_pvf = {'distance':{'pvf_type':'number', 'pvf_value':num_alpha}} 
#
#           TODO: CREATE MULTIWORD CLUSTERING
#            
        
        if arg_manner:
            # print("manner label")
            # print(arg_manner)
            # print(verb)
            is_integer = []
            arg_manner_tokens = []
            for i in range(len(arg_manner)):
                arg_manner_tokens.append(word_list[arg_manner[i]])
                try:
                    if isinstance(w2n.word_to_num(word_list[arg_manner[i]]),int):
                        is_integer.append(i)
                except ValueError:
                    pass
            
            if len(is_integer) == 1:
                num_alpha = w2n.word_to_num(arg_manner_tokens[is_integer[0]])

            else:
                str_concat = ''
                for i in range(len(is_integer)):
                    str_concat = str_concat + ' ' + arg_manner_tokens[is_integer[i]]  
                num_alpha = w2n.word_to_num(str_concat)
            # print(num_alpha)
            # TODO: Tack on unit of measurement as param
            dis_mnr_pvf = {'distance':{'pvf_type':'number', 'pvf_value':num_alpha}} 
            # create num val
        
        input_param_f = {} 
        if verb_token:
            input_param_f.update(verb_pvf)
            # print(verb_pvf)
        if arg_direction:
            input_param_f.update(dir_pvf)
            # print(dir_pvf)
        if arg_extent:
            input_param_f.update(dis_ext_pvf)
            # print(dis_ext_pvf)
        if arg_manner:
            input_param_f.update(dis_mnr_pvf)
            # print(dis_mnr_pvf)
        temoto_umrf = {'effect':'synchronous', 'input_parameters':input_param_f}
        tumrf_json = json.dumps(temoto_umrf)
        print(tumrf_json)
        return tumrf_json

    def create_tumrfs(self, desc):
        # 1) parse the incoming tagged words
        # 2) separate into 
        #    i) input params
        #    ii) output params
        #    iii) suffix - num which differentiates multiple instances of actions (?)
        #    iv) effect - does action terminate after main execution is finished
        #    v) parents
        #    vi) children
        tumrf_jsons = []
        for verb in desc["verbs"]:
            tumrf_jsons.append(self.create_tumrf(verb, desc["words"]))
        return tumrf_jsons
