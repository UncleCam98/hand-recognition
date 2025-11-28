import hand_recognition as hr

detector = hr.HandRecognizer()
detector.run()
try:
    while True:
        gesture = detector.get_gesture()
        position = detector.get_position()

        if gesture == "FIST" or gesture == "PALM":
            print("Gesture:", gesture)
            print("Position:", position)
            print("---------")

except KeyboardInterrupt:
    detector.stop()
