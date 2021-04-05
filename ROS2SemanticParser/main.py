from time import sleep
import rclpy
from std_msgs.msg import String
from CopyNet import predict_continuous
from DataPreProcessing import frame_parse, json_lint, assemble_graph, FIFOBuffer

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    nl_input = "go to the door and turn left"
    parsed = frame_parse(nl_input)

    rclpy.init()
    node = rclpy.create_node('semantic_parser')
    pub = node.create_publisher(String, 'talker', 10)
    msg = String()

    while rclpy.ok():

        decoded_output = []
        for x in parsed:
            output = predict_continuous(x)
            linted_output = json_lint(output)
            decoded_output.append(linted_output)
        graph = assemble_graph(decoded_output)
        msg.data = str(graph)
        pub.publish(msg)
        sleep(0.5)

