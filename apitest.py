import requests
print(requests.post("http://0.0.0.0:10000",json=({"query":"Provide a brief post independence history of Ghana"})).json())# Getting an error that the address is not valid within this context