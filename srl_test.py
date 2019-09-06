#!/usr/bin/python3
from TeMotoUMRF import TemotoUMRF
from allennlp.predictors.semantic_role_labeler import SemanticRoleLabelerPredictor
import rospy
from std_msgs.msg import String

#openIE_model_path = "https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz"
#print("Loading AllenNLP SRL Model into memory")
#predictor = SemanticRoleLabelerPredictor.from_path("~/Downloads/bert-base-srl-2019.06.17.tar.gz")
#print("Loaded AllenNLP SRL Model")
#ex1 = predictor.predict(sentence="please move that table")
#ex2 = predictor.predict(sentence="follow me turtlebot")
#ex3 = predictor.predict(sentence="speed up by 2")
#ex4 = predictor.predict(sentence="open the door and display the hand on the screen")
#ex5 = predictor.predict(sentence="open the rviz program and display the hand on the screen")
# Let's mess with example 1

# 1) for each example, iterate through the "verbs" keyword: ex1["verbs"][i]
# 2) create a string match for ARG descriptor and associated word 
# 3) create a class/dict for tUMRF to be filled with verb and descriptors
# 4) convert python class/dict to JSON format
# 5) create the ROS2.0 package then add the JSON string publisher (to communicate with ROS1)
#print(ex1["verbs"][0]["verb"])
#print(ex1["verbs"][0]["description"])

# Create a tUMRF builder class
# 1) Load the SRL model into class as param & ROSTOPIC (ROS2)
# 2) Have member function that creates tUMRF dict
# 3) Have member function that fills tUMRF dict (pass in public dialogue/string)
# 4) Have member function that converts tUMRF dict -> JSON
# 5) Have member var that publishes JSON to ROSTOPIC (ROS2)
#desc = tUMRF.predict_descriptors("move backwards and then turn ninety degrees clockwise")
#desc = tUMRF.predict_descriptors("follow me at 5 meters per second")
#desc = tUMRF.predict_descriptors("find the cup and pick it up.")

#TODO: 1) Have a listener function for incoming strings
#      2) Probably have de-noising strats to allennlp doesn't have a goddamn stroke

def talker():
    pub = rospy.Publisher('tumrf_json', String, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)

    print('starting test execution')
    srl_model_path = "~/Downloads/bert-base-srl-2019.06.17.tar.gz"
    tUMRF = TemotoUMRF(srl_model_path)

    while not rospy.is_shutdown():
        desc = tUMRF.predict_descriptors("henry drive forward")
        tumrf_jsons = tUMRF.create_tumrfs(desc)
        pub.publish(tumrf_jsons[0])
        print(tumrf_jsons[0])

        desc = tUMRF.predict_descriptors("henry drive ahead")
        tumrf_jsons = tUMRF.create_tumrfs(desc)
        pub.publish(tumrf_jsons[0])
        print(tumrf_jsons[0])
        
        desc = tUMRF.predict_descriptors("henry drive backward")
        tumrf_jsons = tUMRF.create_tumrfs(desc)
        pub.publish(tumrf_jsons[0])
        print(tumrf_jsons[0])
        
        desc = tUMRF.predict_descriptors("henry move forward ten meters")
        tumrf_jsons = tUMRF.create_tumrfs(desc)
        pub.publish(tumrf_jsons[0])
        print(tumrf_jsons[0])
        
        desc = tUMRF.predict_descriptors("henry move backward five meters")
        tumrf_jsons = tUMRF.create_tumrfs(desc)
        pub.publish(tumrf_jsons[0])
        print(tumrf_jsons[0])
        
        desc = tUMRF.predict_descriptors("henry turn left twenty five degrees")
        tumrf_jsons = tUMRF.create_tumrfs(desc)
        pub.publish(tumrf_jsons[0])
        print(tumrf_jsons[0])
        
        desc = tUMRF.predict_descriptors("henry turn right twenty five degrees")
        tumrf_jsons = tUMRF.create_tumrfs(desc)
        pub.publish(tumrf_jsons[0])
        print(tumrf_jsons[0])
        
        desc = tUMRF.predict_descriptors("henry rotate left twenty five degrees")
        tumrf_jsons = tUMRF.create_tumrfs(desc)
        pub.publish(tumrf_jsons[0])
        print(tumrf_jsons[0])
        
        desc = tUMRF.predict_descriptors("henry rotate right twenty five degrees")
        tumrf_jsons = tUMRF.create_tumrfs(desc)
        pub.publish(tumrf_jsons[0])
        print(tumrf_jsons[0])
        
        desc = tUMRF.predict_descriptors("stop")
        tumrf_jsons = tUMRF.create_tumrfs(desc)
        pub.publish(tumrf_jsons[0])
        print(tumrf_jsons[0])

if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
