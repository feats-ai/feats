from RPC import Server
from move_cnc import move


def startSrv(ip, port):
	srv = Server(str(ip), port)
	srv.regFunction(move)
	srv.run()


startSrv("192.168.0.123", 8080)
