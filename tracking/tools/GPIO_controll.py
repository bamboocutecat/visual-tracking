import pigpio
import time

PWM_LED_PIN = 18
PWM_FREQ = 800

pi = pigpio.pi()

try:
    print('按下 Ctrl-C 可停止程式')
    while True:
        for i in range(0, 30, 1):
            pi.hardware_PWM(PWM_LED_PIN, PWM_FREQ, i*10000)
            time.sleep(0.1)
        for i in range(29, -1, -1):
            pi.hardware_PWM(PWM_LED_PIN, PWM_FREQ, i*10000)
            time.sleep(0.1)
except KeyboardInterrupt:
    print('關閉程式')
finally:
    pi.set_mode(PWM_LED_PIN, pigpio.INPUT)