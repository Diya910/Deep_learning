import cv2
import numpy as np
import mediapipe as mp
import pygame
import sys
import time

# Initialize pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 1180, 720
SIDE_PANEL_WIDTH = 300
win = pygame.display.set_mode((WIDTH + SIDE_PANEL_WIDTH, HEIGHT))
pygame.display.set_caption("AI Whiteboard")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Mediapipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Drawing variables
drawing = False
last_point = None
color = BLACK
brush_size = 5

# Main loop
running = True
cap = cv2.VideoCapture(0)

# Clear screen function
def clear_screen():
    win.fill(WHITE)
    pygame.draw.rect(win, BLACK, (WIDTH, 0, SIDE_PANEL_WIDTH, HEIGHT))

clear_screen()

# Draw buttons on side panel
def draw_buttons():
    pygame.draw.rect(win, RED, (WIDTH + 10, 10, 100, 50))
    pygame.draw.rect(win, GREEN, (WIDTH + 10, 70, 100, 50))
    pygame.draw.rect(win, BLUE, (WIDTH + 10, 130, 100, 50))
    pygame.draw.rect(win, WHITE, (WIDTH + 10, 190, 100, 50))

    font = pygame.font.SysFont(None, 24)
    win.blit(font.render('Red', True, WHITE), (WIDTH + 35, 25))
    win.blit(font.render('Green', True, BLACK), (WIDTH + 25, 85))
    win.blit(font.render('Blue', True, WHITE), (WIDTH + 35, 145))
    win.blit(font.render('Clear', True, BLACK), (WIDTH + 30, 205))

draw_buttons()

def detect_full_palm(hand_landmarks):
    # Heuristic to detect if the palm is fully open based on landmark spread
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
    distance = np.sqrt((thumb_tip.x - pinky_tip.x) ** 2 + (thumb_tip.y - pinky_tip.y) ** 2)
    return distance > 0.4  # You might need to adjust this threshold

# Rate limiting for drawing
last_draw_time = 0
draw_delay = 0.2  # secondsw

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if WIDTH + 10 <= mx <= WIDTH + 110:
                if 10 <= my <= 60:
                    color = RED
                elif 70 <= my <= 120:
                    color = GREEN
                elif 130 <= my <= 180:
                    color = BLUE
                elif 190 <= my <= 240:
                    clear_screen()
                    draw_buttons()

    success, img = cap.read()
    if not success:
        continue

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * WIDTH)
            y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * HEIGHT)

            if detect_full_palm(hand_landmarks):
                clear_screen()
                draw_buttons()
            else:
                current_time = time.time()
                if current_time - last_draw_time > draw_delay:
                    if last_point:
                        pygame.draw.line(win, color, last_point, (x, y), brush_size)
                    last_point = (x, y)
                    last_draw_time = current_time
    else:
        last_point = None

    # Show webcam feed in the side panel
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.rot90(img)
    img = pygame.surfarray.make_surface(img)
    img = pygame.transform.scale(img, (SIDE_PANEL_WIDTH, HEIGHT))
    win.blit(img, (WIDTH, 0))

    pygame.display.flip()

cap.release()
cv2.destroyAllWindows()
pygame.quit()