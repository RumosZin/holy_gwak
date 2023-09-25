import cv2
import mediapipe as mp
import numpy as np
import time, os

actions = ['circular', 'horizontal', 'vertical'] # idx 0, 1, 2 매칭
seq_length = 30 # window
secs_for_action = 30 # action recording time

# MediaPipe holistic model
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
holistic = mp.solutions.holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

cap = cv2.VideoCapture(0) # openCV's web cam initialize

created_time = int(time.time())
os.makedirs('dataset', exist_ok=True) # make directory to save dataset

while cap.isOpened():   # 영상이 틀어져 있는 동안 반복
    for idx, action in enumerate(actions):  # action 종류마다 for문 전체 실행
        data = []   # dataset에 넣을 list

        ret, img = cap.read()   # frame 읽어오기?

        img = cv2.flip(img, 1)  # 좌우 반전 -> 셀카모드같이

        # 3초동안 액션 준비시간 주기
        cv2.putText(img, f'Waiting for collecting {action.upper()} action...', org=(10, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        cv2.imshow('img', img)
        cv2.waitKey(3000)

        start_time = time.time()    # 30초 세기

        # face -> 눈 주위 가로선 landmarks idx
        # hand -> 중지 너클 idx
        face_landmark_indices = [127, 34, 143, 35, 226, 130, 33, 7, 163, 144, 145, 153, 154, 155, 133, 243, 244, 245, 122, 6, 351, 465, 464, 463, 362, 382, 381, 380, 374, 373, 390, 249, 263, 369, 446, 265, 372, 264, 356]
        hand_landmark_index = [9]

        while time.time() - start_time < secs_for_action:   # 30초 동안
            ret, img = cap.read()   # 프레임 불러와서

            img = cv2.flip(img, 1)  # 영상반전 해주고
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 홀리스터는 RGB로 바꿔야함
            result = holistic.process(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  

            if result.left_hand_landmarks is not None and result.face_landmarks is not None:    # 오른손(left지만 영상반전) 인식 + 얼굴인식이 될때만 데이터셋 생성
                
                face_point = np.zeros((468, 4)) # 모든 face_point의 landmark와 그에 따른 좌표 넣은 array -> 468행(landmarks), 4열(x,y,z,visibility)
                hand_joint = np.zeros((21, 4))  # hand는 21개 landmark

                for j, lm in enumerate(result.face_landmarks.landmark): # 바로 전에 만든 배열에 값 삽입 과정
                    face_point[j] = [lm.x, lm.y, lm.z, lm.visibility] # see or not - visibility

                for k, lm2 in enumerate(result.left_hand_landmarks.landmark):
                    hand_joint[k] = [lm2.x, lm2.y, lm2.z, lm2.visibility]

                # Compute angles between points
                v1 = face_point[face_landmark_indices, :3]  # v1좌표 -> face에서 필요한 landmark만 가져오기 + visibility 슬라이싱
                v2 = hand_joint[hand_landmark_index, :3]    # v2좌표
                v = v1 - v2 # 벡터 -> 2차원 배열 생성 [39][3]
#                # Normalize v
#                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]    # 벡터 정규화 (크기 1)
#
#                # Get angle using arcos of dot product
#                angle = np.arccos(np.einsum('nt,nt->n', # 각 계산 -> 이거 슬라이싱으로 코드 줄여볼라했는데 왜 안됨?
#                    v[[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37],:], 
#                    v[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38],:]))
#
#                angle = np.degrees(angle) # Convert radian to degree 단위 변환
#
#                angle_label = np.array([angle], dtype=np.float32)
#                angle_label = np.append(angle_label, idx) # labeling 0, 1, 2

                v_label = np.array(v, dtype=np.float32)
                v_label = np.append(v_label, idx)

 #               d = np.concatenate([face_point.flatten(), hand_joint.flatten(), angle_label])   # face_point 2차원 array, hand_joint 2차원 array, 라벨링 1차원 array로 만들고 concatenate
                d = np.concatenate([face_point.flatten(), hand_joint.flatten(), v_label])   # face_point 2차원 array, hand_joint 2차원 array, 라벨링 1차원 array로 만들고 concatenate

                data.append(d)  # 아까 만든 데이터 배열에 넣기 -> data = [[data1,label], [data2,label], ... [datan,label]] 2차원 배열

                # 1. Draw face landmarks 드로잉용
                mp_drawing.draw_landmarks(img, result.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                            )
                # 2. Left Hand landmarks
                mp_drawing.draw_landmarks(img, result.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                            )

            cv2.imshow('img', img)
            if cv2.waitKey(1) == ord('q'):
                break


        data = np.array(data)
        print(action, data.shape)
        np.save(os.path.join('dataset', f'raw_{action}_{created_time}'), data)

        # Create sequence data
        full_seq_data = []
        for seq in range(len(data) - seq_length):
            full_seq_data.append(data[seq:seq + seq_length])

        full_seq_data = np.array(full_seq_data)
        print(action, full_seq_data.shape)
        np.save(os.path.join('dataset', f'seq_{action}_{created_time}'), full_seq_data)
    break
