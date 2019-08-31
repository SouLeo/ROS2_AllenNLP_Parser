from allennlp.predictors.semantic_role_labeler import SemanticRoleLabelerPredictor

#predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz")
print("Loading AllenNLP SRL Model into memory")
predictor = SemanticRoleLabelerPredictor.from_path("~/Downloads/bert-base-srl-2019.06.17.tar.gz")
print("Loaded AllenNLP SRL Model")
ex1 = predictor.predict(sentence="please move that table")
ex2 = predictor.predict(sentence="follow me turtlebot")
ex3 = predictor.predict(sentence="speed up by 2")
ex4 = predictor.predict(sentence="open the door and display the hand on the screen")
ex5 = predictor.predict(sentence="open the rviz program and display the hand on the screen")
# Let's mess with example 1

# 1) for each example, iterate through the "verbs" keyword: ex1["verbs"][i]
# 2) create a string match for ARG descriptor and associated word 
# 3) create a class/dict for tUMRF to be filled with verb and descriptors
# 4) convert python class/dict to JSON format
# 5) create the ROS2.0 package then add the JSON string publisher (to communicate with ROS1)
print(ex1["verbs"][0]["verb"])
print(ex1["verbs"][0]["description"])

# Create a tUMRF builder class
# 1) Load the SRL model into class as param & ROSTOPIC (ROS2)
# 2) Have member function that creates tUMRF dict
# 3) Have member function that fills tUMRF dict (pass in public dialogue/string)
# 4) Have member function that converts tUMRF dict -> JSON
# 5) Have member var that publishes JSON to ROSTOPIC (ROS2)
