import cv2
import mediapipe as mp
import math
import random
import numpy as np

cap = cv2.VideoCapture(0)
neg = cv2.imread('./img/negative.png', cv2.IMREAD_UNCHANGED)
# 블록을 지정 그림으로 설정하기 위해 이미지 읽음+마스킹을 위한 알파채널

cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

### 주요 변수 설정 ###
compareIndex = [[18, 4], [6, 8], [10, 12], [14, 16], [18, 20]]
#엄지,검지,중지,약지,소지에 대한 마디 값 2개/접힘과 안접힘을 검사하기 위한 (5,2)배열
#ex)손 마디중 14가 16보다 길다면 해당 손가락이 접힌 것
open = [False, False, False, False, False]  # 손가락의 접힘/펴짐에 대한 배열(접히면 False)
auto_flag = False               #게임을 오토로 실행할지/말지
target = (450, 240, 70, 70)     #마우스의 위치 값
block_arr = np.zeros((5, 3))    #블록에 대한 x, y, w&h(높이,너비 동일)
block_arr = block_arr.astype(int) ##arr를 int형으로(zeros는 실수 디폴트)
ms = 100                        #100==10초
count = 0                       #블록 arr의 행 지정 변수
score = 0                       #점수 변수

##mediapipe의 hands API 세팅##
mpHands = mp.solutions.hands
my_hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils #손과 연결 선을 그리는 것

#손바닥(x1,y1)에서 손가락(x2,y2)까지의 거리를 구하는 함수
#점과 점 사이의 거리 구하는 공식
def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)** 2) + math.sqrt((y1 - y2)** 2)

# 새 블록을 추가하는 함수
def insert_block(count, block_arr):
    # 위치, 너비 모두 랜덤
    x = (random.randint(0, 640))
    y = (random.randint(0, 480))
    wh = (random.randint(30, 80))
    # 화면 밖을 넘어가지 않도록 설정
    if x + wh > 640:
        x -= wh
    if y + wh > 480:
        y -= wh
    block_arr[count][0] = x
    block_arr[count][1] = y
    block_arr[count][2] = wh


# 모든 블록을 그리는 함수
def draw_block(frame, block_arr):
    for i in range(0, 5):
        x, y, wh = block_arr[i]
        if (block_arr[i][2] == 0):  # 너비가 0이라면 그려지지 않는 블록
            continue
        # 화면과 그림을 합성하기 위해 크기만큼 resize
        block = cv2.resize(neg[:, :, :], dsize=(wh, wh))
        # copyTo함수로 마스킹 부분에 블록을 합성함
        cv2.copyTo(block[:, :, :3], block[:, :, 3], frame[y:y + wh, x:x + wh])


# 블록이 터치 되었을 때 제거하고 점수를 증가시키는 함수
def check_touch_block(rc, rec_arr):
    global score,count
    # for문으로 모든 블록을 확인하며 터치된 블록의 행을 확인
    for i in range(0, 5):
        x, y, wh = rec_arr[i]
        if (rc[0]) >= x and (rc[1]) >= y and (rc[0]) <= x + wh and (rc[1]) <= y + wh:
            rec_arr[i] = (0, 0, 0)
            score = score + 100  # 점수가 100점씩 증가
            insert_block(count, block_arr)  # 닿은 블록을 지우고 새로 생성하는 함수
            count = (count + 1) % 5  # count를 0~5사이값으로 유지

##1.게임 시작 화면##
while True:
    ret, frame = cap.read()
    h, w, c = frame.shape

    #영상을 hands API에 전달-손바닥과 21개의 손가락 점에 대한 결과 반환
    results = my_hands.process(frame)
    if results.multi_hand_landmarks: #전달된 결과가 있을 때
        for handLms in results.multi_hand_landmarks:
            for i in range(0, 5):  ##모든 손가락에 대한 상태 확인
                #한 손가락에 대해 마디 길이를 dist 함수로 비교하여 펴짐/접힘에 대한 True/False 저장
                open[i] = dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[compareIndex[i][0]].x,
                               handLms.landmark[compareIndex[i][0]].y) < dist(handLms.landmark[0].x,
                                                                              handLms.landmark[0].y,
                                                                              handLms.landmark[compareIndex[i][1]].x,
                                                                              handLms.landmark[compareIndex[i][1]].y)
                # print(open) ##영상에 손가락 선과 마디 원을 그리는 부분
                mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

    ##게임 메인 화면##
    cv2.putText(frame, "1.Start  2.Auto  3.End", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    # 추적 위치(사용자 마우스) 표시
    cv2.putText(frame, "Put", (target[0] + 10, target[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.rectangle(frame, target, (0, 255, 255), 2)
    # 게임 시작-손가락 1
    if open[0] == False and open[1] == True and open[2] == False and open[3] == False and open[4] == False:
        break
    # 게임 오토 시작-손가락 2
    elif open[0] == False and open[1] == True and open[2] == True and open[3] == False and open[4] == False:
        auto_flag = True
        break
    # 게임 종료-손가락 3
    elif open[0] == True and open[1] == True and open[2] == True and open[3] == False and open[4] == False:
        exit(0)

    cv2.imshow('StartScreen', frame)
    cv2.waitKey(20)
##게임 시작 화면 종료##
cv2.destroyWindow('StartScreen')


##2.게임 화면##
ret, frame = cap.read()
##추적할 객체(사용자 마우스) 설정
tracker = cv2.TrackerCSRT_create()  # 객체 생성
tracker.init(frame, target)  # 추적 위치를 target값으로

while True:
    ret, frame = cap.read()
    if not ret:
        print('Frame read failed!')

    # 새로운 블록이 3초 간격으로 생성
    if ms % 30 == 0:
        insert_block(count, block_arr)  # 새로운 랜덤위치에 블록 생성
        count = (count + 1) % 5  # 5행 배열을 사용하여 화면에 블록 5개만 존재하도록 함

    # ms가 0 == 게임 종료//최종 점수를 표시함
    if ms <= 0:  # rec=검정 배경화면 putText=점수 표시
        cv2.rectangle(frame, (int(cap_w / 4) - 30, int(cap_h / 2) - 50, 410, 80), (0, 0, 0), -1)
        cv2.putText(frame, "Your score is " + str(score), (int(cap_w / 4) - 10, int(cap_h / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.imshow('GameScreen', frame)
        if cv2.waitKey(0) == 27:  # 종료 전 ESC 입력을 기다림
            break

    draw_block(frame, block_arr)  # 배열에 저장된 블록을 그리는 함수

    # 오토 모드-프레임의 특징점을 찾아 마우스를 자동으로 변경
    if auto_flag:
        feature = cv2.ORB_create(7) #최대 7개의 특징점을 찾음
        kp1, desc = feature.detectAndCompute(frame, mask=None)
        for kp in kp1:
            x, y = kp.pt    #특징점의 좌표가 저장된 변수
            rc = (int(x), int(y), 70, 70)
            cv2.rectangle(frame, rc, (255, 255, 255), 2)
            # 사용자 마우스가 블록에 닿았을때
            if check_touch_block(rc, block_arr):
                print('touch')
    # 사용자 모드-마우스를 추적함
    else:
        ret, rc = tracker.update(frame)  # 사용자 마우스를 추적하여 업데이트함
        rc = tuple([int(_) for _ in rc])  # 실수 좌표값을 정수 튜플로 변환함
        cv2.rectangle(frame, rc, (0, 0, 255), 2)  # 사용자 마우스의 범위를 표시

        # 사용자 마우스가 블록에 닿았을때
        if check_touch_block(rc, block_arr):
            print('touch')

    ##점수와 제한시간 표시##
    time = int(ms / 10)
    if time < 10:  # 10보다 작은 값은 0을 붙여 위치를 맞춤-
        time = "0" + str(time)
    # 각 글자의 위치는 임의로 설정
    cv2.putText(frame, "Score: " + str(score), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, "00 : " + str(time), (480, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow('GameScreen', frame)

    if cv2.waitKey(30) == 27:
        break
    print(ms)
    ms = ms - 1  # 제한 시간 감소

cap.release()
cv2.destroyAllWindows()
