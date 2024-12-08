#How to use RPCs 
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
