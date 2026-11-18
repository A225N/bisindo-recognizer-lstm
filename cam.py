import cv2
import time

cap = cv2.VideoCapture(2)

print("Webcam successfully started. Press 'q' to exit.")

# Countdown parameters
countdown_duration = 3
start_time = time.time()

while True:
    ret, frame = cap.read()

    # Calculate remaining time
    elapsed = time.time() - start_time
    remaining = countdown_duration - int(elapsed)

    if remaining > 0:
        mins, secs = divmod(remaining, 60)
        timer_text = f"{mins:02d}:{secs:02d}"
    else:
        timer_text = "00:00"

    # Display overlay on video frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, timer_text, (10, 450), font, 2, (255, 255, 255), 3, cv2.LINE_AA)

    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()