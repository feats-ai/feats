## RPC.py 
Module for Server and Client class for implementing Remote Procedure Calls in python<br>
(Thanks to: https://medium.com/@taraszhere/coding-remote-procedure-call-rpc-with-python-3b14a7d00ac8)

Server:
```python
from RPC import Server

srv = Srever() # init with address and port

#define function to register
def square(a):
  return a*a

srv.regFunction(square) #register function
srv.run() #run server
```
Client:
```
from RPC import Client

```
