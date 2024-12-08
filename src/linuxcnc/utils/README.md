## RPC.py 
Module for Server and Client class for Remote Procedure Calls in python<br>
(Thanks to: https://medium.com/@taraszhere/coding-remote-procedure-call-rpc-with-python-3b14a7d00ac8)

### Example

Server:
```python
from RPC import Server

srv = Srever() # init with address and port e.g. Server('192.168.0.1', 8080)

#define function to register
def square(a):
  return a*a

srv.regFunction(square) #register function
srv.run() #run server
```
Client:
```python
from RPC import Client

client = Client() #init with address and port e.g. Client('192.168.0.1', 8080)
client.connect() # connect to given address and port

result = client.square(5) # call server method
print(result)
```
Output:
```plaintext
25
```
