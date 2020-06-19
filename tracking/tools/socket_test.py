import socket, sys

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except socket.error as msg:
    sys.stderr.write("[ERROR] %s\n" % msg[1])
    sys.exit(1)

try:
    sock.connect(('192.168.0.11', 9001))
except socket.error as msg:
    sys.stderr.write("[ERROR] %s\n" % msg[1])
    exit(1)

sock.send("Hello I'm Client.\r\n")
print (sock.recv(1024))
sock.close()