

#################
# background threaed with ZMQ
import threading
import zmq
import time

def subscriber(stop_event):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)

    socket.connect("tcp://127.0.0.1:5555")
    socket.setsockopt_string(zmq.SUBSCRIBE, "gpu ")

    # Allow periodic checks for shutdown.
    socket.setsockopt(zmq.RCVTIMEO, 500)

    try:
        while not stop_event.is_set():
            try:
                message = socket.recv_string()
                print(f"Received: {message}")
            except zmq.Again:
                continue
    finally:
        socket.close()
        context.term()


stop_event = threading.Event()

thread = threading.Thread(
    target=subscriber,
    args=(stop_event,),
    daemon=True
)

thread.start()

try:
    for i in range(10):
        print(f"Main thread working: {i}")
        time.sleep(1)
finally:
    stop_event.set()
    thread.join()
