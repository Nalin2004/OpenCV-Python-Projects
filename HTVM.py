import cv2
import mediapipe as mp
import math
import pyautogui
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
screen_width, screen_height = pyautogui.size()

last_click_time = 0

mp_draw = mp.solutions.drawing_utils
camera = cv2.VideoCapture(0)

while True:

    success, frame = camera.read()

    if not success:
        break

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )
    results = hands.process(
        rgb_frame
    )

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            thumb_x, thumb_y = 0, 0
            index_x, index_y = 0, 0

            for id, landmark in enumerate(
                    hand_landmarks.landmark):

                height, width, channels = frame.shape

                cx = int(
                    landmark.x * width
                )

                cy = int(
                    landmark.y * height
                )

                if id == 4:

                    thumb_x = cx
                    thumb_y = cy

                    cv2.circle(
                        frame,
                        (cx, cy),
                        15,
                        (255, 0, 0),
                        -1
                    )

                if id == 8:

                    index_x = cx
                    index_y = cy

                    mouse_x = screen_width / width * cx
                    mouse_y = screen_height / height * cy

                    pyautogui.moveTo(
                        mouse_x,
                        mouse_y,
                        duration=0.05
                    )

                    cv2.circle(
                        frame,
                        (cx, cy),
                        15,
                        (0, 0, 255),
                        -1
                    )

            distance = math.sqrt(
                (index_x - thumb_x) ** 2 +
                (index_y - thumb_y) ** 2
            )

            if distance < 40:

                gesture = "CLICK"
                line_color = (0, 255, 0)
                current_time = time.time()

                if current_time - last_click_time > 0.5:

                    pyautogui.click()

                    last_click_time = current_time

            else:

                gesture = "OPEN"
                line_color = (255, 0, 0)

            cv2.line(
                frame,
                (thumb_x, thumb_y),
                (index_x, index_y),
                line_color,
                3
            )

            cv2.putText(
                frame,
                f"Distance: {int(distance)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                gesture,
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                line_color,
                2
            )

    cv2.imshow(
        "Hand Tracking",
        frame
    )

    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()