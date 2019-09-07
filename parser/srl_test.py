import inspect
import http.server as hserver 
import socketserver
from TeMotoUMRF import TemotoUMRF 
from allennlp.predictors.semantic_role_labeler import SemanticRoleLabelerPredictor

#TODO: 1) Have a listener function for incoming strings
#      2) Probably have de-noising strats to allennlp doesn't have a goddamn stroke

class MyAss(hserver.BaseHTTPRequestHandler):
    
    srl_model_path = "https://s3-us-west-2.amazonaws.com/allennlp/models/bert-base-srl-2019.06.17.tar.gz"
    tUMRF = TemotoUMRF(srl_model_path)

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()


    def do_GET(self):
        self._set_headers()
        self.wfile.write(b'<html><body><h1>GET!</h1></body></html>')
        print("ass")


    def do_HEAD(self):
        self._set_headers()


    def do_POST(self):
        content_len = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_len)
        input_sentence = post_data.decode("utf-8")
        desc = MyAss.tUMRF.predict_descriptors(input_sentence)
        tumrf_jsons = MyAss.tUMRF.create_tumrfs(desc)
        self._set_headers()
        self.wfile.write(tumrf_jsons[0].encode("utf-8"))


#    def handle(self):
#        data = self.request.recv(1024)
#        print(data)
#        self.request.send(data)
#        return
#

def main(args=None):
    print('starting test execution')
    
    print('setting up server')
    PORT = 80
    # httpd = socketserver.TCPServer(("", PORT), MyAss)
    httpd = hserver.HTTPServer(('', PORT), MyAss)
    print('server ready')
    httpd.serve_forever()
    

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
