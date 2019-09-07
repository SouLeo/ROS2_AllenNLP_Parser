import http.server as hserver 
import socketserver
from TeMotoUMRF import TemotoUMRF 
from allennlp.predictors.semantic_role_labeler import SemanticRoleLabelerPredictor

#TODO: 1) Have a listener function for incoming strings
#      2) Probably have de-noising strats to allennlp doesn't have a goddamn stroke

class MyAss(socketserver.BaseRequestHandler):

    def set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()


    def do_GET(self):
        self.set_headers()
        print("ass")


    def handle(self):
        data = self.request.recv(1024)
        print(data)
        self.request.send(data)
        return


def main(args=None):
    
    print('setting up server')
    PORT = 80
    # httpd = socketserver.TCPServer(("", PORT), MyAss)
    httpd = hserver.HTTPServer(('', PORT), MyAss)
    httpd.serve_forever()
    
    print('starting test execution')
    srl_model_path = "https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz"
    tUMRF = TemotoUMRF(srl_model_path)

    desc = tUMRF.predict_descriptors("henry drive forward")
    tumrf_jsons = tUMRF.create_tumrfs(desc)
    print(tumrf_jsons[0])

    desc = tUMRF.predict_descriptors("henry drive ahead")
    tumrf_jsons = tUMRF.create_tumrfs(desc)
    print(tumrf_jsons[0])
    
    desc = tUMRF.predict_descriptors("henry drive backward")
    tumrf_jsons = tUMRF.create_tumrfs(desc)
    print(tumrf_jsons[0])
    
    desc = tUMRF.predict_descriptors("henry move forward ten meters")
    tumrf_jsons = tUMRF.create_tumrfs(desc)
    print(tumrf_jsons[0])
    
    desc = tUMRF.predict_descriptors("henry move backward five meters")
    tumrf_jsons = tUMRF.create_tumrfs(desc)
    print(tumrf_jsons[0])
    
    desc = tUMRF.predict_descriptors("henry turn left twenty five degrees")
    tumrf_jsons = tUMRF.create_tumrfs(desc)
    print(tumrf_jsons[0])
    
    desc = tUMRF.predict_descriptors("henry turn right twenty five degrees")
    tumrf_jsons = tUMRF.create_tumrfs(desc)
    print(tumrf_jsons[0])
    
    desc = tUMRF.predict_descriptors("henry rotate left twenty five degrees")
    tumrf_jsons = tUMRF.create_tumrfs(desc)
    print(tumrf_jsons[0])
    
    desc = tUMRF.predict_descriptors("henry rotate right twenty five degrees")
    tumrf_jsons = tUMRF.create_tumrfs(desc)
    print(tumrf_jsons[0])
    
    desc = tUMRF.predict_descriptors("stop")
    tumrf_jsons = tUMRF.create_tumrfs(desc)
    print(tumrf_jsons[0])


if __name__ == '__main__':
    main()
