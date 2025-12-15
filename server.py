'''
# server.py (노트북에서 실행)
import socket

HOST = '0.0.0.0'  # 모든 인터페이스에서 수신
PORT = 9999       # 포트 번호 (자유롭게 설정 가능)

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)
print("[서버] 연결 대기 중...")

conn, addr = server.accept()
print(f"[서버] 연결됨: {addr}")

data = conn.recv(1024)
print("[서버] 받은 메시지:", data.decode())

conn.close()
'''
import socket
import cv2
import numpy as np
import struct

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('0.0.0.0', 9999))
server.listen(1)
conn, _ = server.accept()

while True:
    # 4바이트 크기 정보 수신
    size_data = conn.recv(4)
    if not size_data:
        break
    size = struct.unpack(">L", size_data)[0]

    # 프레임 데이터 수신
    data = b""
    while len(data) < size:
        packet = conn.recv(size - len(data))
        if not packet:
            break
        data += packet

    # 디코드 및 출력
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    cv2.imshow('Raspberry Pi Stream', frame)
    if cv2.waitKey(1) == ord('q'):
        break

conn.close()
