'''
This module contains helper classes to perform Remote procedure calls to control the CNC milling machine.
(Thanks to https://medium.com/@taraszhere/coding-remote-procedure-call-rpc-with-python-3b14a7d00ac8!!)

Example function call:
    Server:
    >>> def square(a):
    >>>     return a*a
    >>> srv = Server()
    >>> srv.regFunction(square) #register square function to server
    >>> srv.run() #run server
    
    Client:
    >>> clt = Client()
    >>> client.connect()
    >>> client.square(5) # remote procedure call 
    

- Server(): Server class accepting method calls.
- Client(): Client class connecting to Server and calling Server methods.
'''
from __future__ import print_function
import socket
import json
import sys 

#if sys.version_info[0] == 2:
#    VER = 2
#elif sys.version_info[0] == 3:
#    VER = 3
#else: 
#    raise Exception("Unknown python version...")

SIZE = 1024


class Server():
    '''
    Server class opening a socket and waiting for a client connection.
    Client can call registered methods, registered with regFunction() before running the server. 

    Attribute: 
        - host (string): Address where to open the socket
        - port (int): Port where to open the socket
        
    Methods:
        - regFunction(x): recister a function x
        - run()         : runs the server
    '''
    
    def __init__(self, host:str='localhost', port:int=8080):
        self.address = (host, port)
        self._methods = {}

    def regFunction(self, function):
        '''
        register a method to the server, that can be called then by a client.
        
        Args:
            - function (function): function object to be registered 
        '''
        try:
            self._methods.update({function.__name__ : function})
        except:
            raise Exception('A non method object was passed to regFunction()!')

    def run(self):
        '''
        Runs the server and waits for connection on given adress and port.
        '''
        srv_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv_sock.bind(self.address)
        srv_sock.listen()

        print('Server {} ist gestartet ...'.format(self.address))

        try:
            connection, client_address = srv_sock.accept()
        except KeyboardInterrupt:
            print('KeyboardInterrupt ...')

        print('Verbindung hergestellt mit ({},{})'.format(client_address[0], client_address[1]))

        try:
            while True:
                try:
                    functionName, args, kwargs = json.loads(connection.recv(SIZE).decode())
                except:
                    print('! Client {} disconnected...a'.format(client_address))
                    break

                print('Request: {}({}) , kwargs: {}'.format(functionName, args, kwargs))
                
                try:
                    response = self._methods[functionName](*args, **kwargs)
                except Exception as e:
                    connection.sendall(json.dumps('Function not registered: ' + str(e)).encode()) 
                else:
                    connection.sendall(json.dumps(response).encode())

        finally:
            connection.close()


class Client:
    '''
    Client class connecting to a Server Socket and calling registered Server functions
    
    Args: 
        - host (string): IPv4 adress of the host
        - port (int): Port to connect to
     
    Methods:
        - connect(): Connects to the given Address and Port
        - disconnect(): Disconnect from a running connection
        
    '''
	
    def __init__(self, host:str='localhost', port:int=8080) -> None:
        self._sock = None
        self._address = (host, port)
	
    def connect(self):
        try:
            self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._sock.connect(self._address)
        except EOFError as e:
            print(e)
            raise Exception('Client was not able to connect.')
		
    def disconnect(self):
        try:
            self._sock.close()
        except:
            pass

    def __getattr__(self, _name:str):
        def exec(*args, **kwargs):
            self._sock.sendall(json.dumps((_name, args, kwargs)).encode())
            response = json.loads(self._sock.recv(SIZE).decode())
            return response

        return exec

    def __del__(self):
        try:
            self._sock.close()
        except:
            pass
