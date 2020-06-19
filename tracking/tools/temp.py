import serial
import binascii
string = 'VT=0 G '
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=0.5)
#print(ser.name)  # 列印裝置名稱
#print(ser.port)  # 列印裝置名
#ser.open()
if not ser.isOpen():
    ser.open()

hex_b = ''.join(hex(ord(c))[2:] for c in string)
print(hex_b)
print(bytes.fromhex(hex_b))
ser.write(bytes.fromhex(hex_b))
#ser.write(b'\x80\x47\x4F\x53\x55\x42\x33\x20')