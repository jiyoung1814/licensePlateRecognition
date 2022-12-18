import cv2
import numpy as np #수치계산
import matplotlib.pyplot as plt #시각화
import pytesseract #글자 인식
import easyocr
plt.style.use('dark_background')

#read input image
# img_ori = cv2.imread('img/test.png',cv2.IMREAD_COLOR)
# img_ori = cv2.imread('img/test2.png',cv2.IMREAD_COLOR)
# img_ori = cv2.imread('img/test3.png',cv2.IMREAD_COLOR)
img_ori = cv2.imread('img/img1.jpg',cv2.IMREAD_COLOR)
# img_ori = cv2.imread('img/test8.jpg',cv2.IMREAD_COLOR)

# plt.figure(figsize=(6,5))
# plt.imshow(img_ori)
# plt.show()


#미디언 블러링
# img_ori = cv2.medianBlur(img_ori, 3);
# plt.imshow(img_ori, cmap='gray')
# plt.show()


height, width, channel = img_ori.shape  #높이,너비,채널의 값

# plt.figure(figsize=(6,5))
# plt.imshow(img_ori, cmap='gray') #gray로 설정하여 출력
# print(height, width, channel)


#Convert Image to Grayscale
gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY) #gray scale로 변경
# plt.figure(figsize=(6,5))
# plt.imshow(gray, cmap='gray')
# plt.show()

#Adaptive Thresholding
# img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0) #가우시안 블러 -> 노이즈 제거
# plt.figure(figsize=(6,5))
# plt.imshow(img_blurred, cmap='gray')

img_blur_thresh = cv2.adaptiveThreshold(  #threshold 값을 기준으로 정하고 이보다 낮은 값은 0, 높은 값은 255로 변환
    gray,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)
# plt.figure(figsize=(6,5))
# plt.title('Blur and Threshold')
# plt.imshow(img_blur_thresh, cmap='gray')
# plt.show()

#Find Contours
#Contours: 동일한 색 또는 동일한 강도를 가지고 있는 영역의 경계선을 연결한 선
#findContours(): 검은색 바탕에서 흰색 대상을 찾음

# 윤곽선 찾기
contours, _ = cv2.findContours(
    img_blur_thresh,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
)
# 빈 이미지 생성
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

# 윤곽선을 그려줌
cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255,255,255))

# plt.figure(figsize=(6, 5))
# plt.imshow(temp_result)
# plt.show()


#Prepare Data
## 윤곽선을 사각형 모양으로 그리기 위한 함수
# contours: 윤곽선 목록
#cv2.boundingRect(): 윤곽선을 감싸는 사각형을 구한다.

temp_result = np.zeros((height, width, channel), dtype=np.uint8)
contours_dict = []
for contour in contours:
    # 주어진 점을 감싸는 최소 크기 사각형(바운딩 박스)를 반환합니다.
    x, y, w, h = cv2.boundingRect(contour)
    # 생성한 이미지에 사각형을 그림
    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)

    # print(w*h);

    # 사각형 정보를 넣어줌
    # cx: x좌표의 중심, cy: y 좌표의 중심
    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    })
#
plt.figure(figsize=(6,5))
plt.imshow(temp_result, cmap='gray')
# plt.show()


#Select Candidates by Char Size
# 사각형 중 유효한 사각형을 추출
#contours_dict: 사각형 목록
#possible_contours: 유효한 사각형 목록

# # iot 제작 번호판
# # 사각형의 최소 넓이
# MIN_AREA = 3000
# MAX_AREA = 10000
# # 사각형의 최소 폭, 높이
# MIN_WIDTH, MIN_HEIGHT = 20, 30
# # 사각형의 최소, 최대 가로 세로 비율
# MIN_RATIO, MAX_RATIO = 0.5, 1.0

#라즈베리카메라 자동차 번호판
MIN_AREA = 1000   # 번호판 윤곽선 최소 범위 지정
MAX_AREA = 100000
MIN_WIDTH, MIN_HEIGHT = 10, 10         # 최소 너비 높이 범위 지정
MIN_RATIO, MAX_RATIO = 0.25, 1.0     # 최소 비율 범위 지정

possible_contours = []

# 유효한 사각형에 부여되는 index
cnt = 0

for d in contours_dict:
    area = d['w'] * d['h']   # 넓이
    ratio = d['w'] / d['h']  # 비율

    #라즈베리카메라 자동차 번호판
    if area > MIN_AREA and area <MAX_AREA\
            and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        print("{}".format(cnt)+": " +"{}".format(area)+"==> w:"+"{}".format(d['w'])+", h: "+"{}".format(d['h'])+", ratio: "+"{}".format(ratio))
        cnt += 1
        possible_contours.append(d)
        # if cnt==3: break

    # #iot 제작 번호판
    # if area > MIN_AREA and area < MAX_AREA\
    #         and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
    #         and MIN_RATIO < ratio < MAX_RATIO:
    #     d['idx'] = cnt
    #     print("{}".format(cnt)+": " +"{}".format(area)+"==> w:"+"{}".format(d['w'])+", h: "+"{}".format(d['h'])+", ratio: "+"{}".format(ratio))
    #     cnt += 1
    #     possible_contours.append(d)
    #     # if cnt==10: break

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for d in possible_contours:
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                  thickness=2)
plt.figure(figsize=(6, 5))
plt.imshow(temp_result, cmap='gray')
# plt.show()


#Select Candidates by Arrangement of Contours
##더 유효한 사각형 추출

#조건
# 1. 번호판 Contours(윤각선)의 width와 height의 비율은 모두 동일하거나 비슷하다.
# 2. 번호판 Contours 사이의 간격은 일정하다.
# 3. 최소 3개 이상 Contours가 인접해 있어야한다. (대한민국 기준)

# # iot 제작 자동차 번호판
# # MAX_DIAG_MULTIPLYER = 5 # 사각형의 대각선 길이의 5배가 최대 간격
# MAX_DISTANT_DIFF = 200
# MAX_ANGLE_DIFF = 2   # 사각형의 최대 중심 최대 각도  (1번째 contour와 2번째 contour 의 각도)
# MAX_AREA_DIFF = 0.5     # 사각형의 최대 면적 차이
# MAX_WIDTH_DIFF = 0.3    # 사각형의 최대 넓이 차이
# MAX_HEIGHT_DIFF = 0.03   # 사각형의 최대 높이 차이
# MIN_N_MATCHED = 3       # 사각형의 그룹의 최소 갯수

#라즈베리 자동차 번호판
MAX_DIAG_MULTIPLYER = 5 # 사각형의 대각선 길이의 5배가 최대 간격
MAX_DISTANT_DIFF = 80
MAX_ANGLE_DIFF = 20   # 사각형의 최대 중심 최대 각도  (1번째 contour와 2번째 contour 의 각도)
MAX_AREA_DIFF = 1     # 사각형의 최대 면적 차이
MAX_WIDTH_DIFF = 1   # 사각형의 최대 넓이 차이
MAX_HEIGHT_DIFF = 1   # 사각형의 최대 높이 차이
MIN_N_MATCHED = 3       # 사각형의 그룹의 최소 갯수


def find_chars(contour_list):
    matched_result_idx = []


#ex) 첫번째 contour와 두번째 contour를 비교 -> 두번째 contour와 세번째 contour를 비교
    cnt = 0
    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            # 각을 구하기 위한 중심 거리 계산
            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            # d1의 대각선 길이
            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            # 중심 간격
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))

            # 각 계산
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))

            # 면적 비율
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            # 폭의 비율
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            # 높이의 비율
            height_diff = abs(d1['h'] - d2['h']) / d1['h']


            # print("{}".format(d1['idx'])+", {}: ".format(d2['idx'])+"대각선의 길이{}".format(diagonal_length1)+", 거리{}".format(distance)+", 각도{}".format(angle_diff)+", 면적{}".format(area_diff)+", 너비{}".format(width_diff)+", 높이{}".format(height_diff))
            # cnt +=1

            # # 실제 자동차 번호판 조건 확인
            # if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            #         and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            #         and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
            #     matched_contours_idx.append(d2['idx'])

            # iot,라즈베리카메라 제작 번호판 조건 확인
            if distance < MAX_DISTANT_DIFF \
                    and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                    and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])
            # print("d2: ")
            # print(matched_contours_idx)

        matched_contours_idx.append(d1['idx'])
        # print("d1: ")
        # print(matched_contours_idx)

        # rect group이 기준 이하(3개 이하)면 결과에 포함하지 않음
        if len(matched_contours_idx) < MIN_N_MATCHED:  #최소 3개 이상 Contours가 인접
            continue

        # 모든 조건에 맞는 사각형 결과에 포함
        matched_result_idx.append(matched_contours_idx)

        # 매칭이 안된 것끼리 다시 진행
        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        #np.take() => 인덱스를 이용해서 어레이의 요소를 가져옵니다.
        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

        recursive_contour_list = find_chars(unmatched_contour)

        # recursive 결과 취합
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    # print(matched_result_idx)

    return matched_result_idx


result_idx = find_chars(possible_contours)

matched_result = []
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for r in matched_result:
    for d in r:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=2)

plt.figure(figsize=(6, 5))
plt.imshow(temp_result, cmap='gray')
# plt.show();

#Rotate Plate Images
PLATE_WIDTH_PADDING = 1.3  # 1.3
PLATE_HEIGHT_PADDING = 1.5  # 1.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

plate_imgs = []
plate_infos = []

n = 0

for i, matched_chars in enumerate(matched_result):
    sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

    # 사각형 사이의? 센터 좌표 구하기
    plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
    plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

    plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING

    sum_height = 0
    for d in sorted_chars:
        sum_height += d['h']

    plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)

    # 번호판의 기울어진 각도를 구하기 (삼각함수 이용)
    triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']
    triangle_hypotenus = np.linalg.norm(
        np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) -
        np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    )


    angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))

    # print('width: {}, height: {}, angle: {}'.format(plate_width, plate_height, angle))

    # cv2.getRotationMatrix2D() 로테이션매트릭스를 구한다.
    rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)

    # cv2.warpAffine() 이미지를 변현한다.
    img_rotated = cv2.warpAffine(img_blur_thresh, M=rotation_matrix, dsize=(width, height))

    # cv2.getRectSubPix() 회전된 이미지에서 원하는 부분만을 잘라낸다.
    img_cropped = cv2.getRectSubPix(
        img_rotated,
        patchSize=(int(plate_width), int(plate_height)),
        center=(int(plate_cx), int(plate_cy))
    )
    # plt.figure(figsize=(6, 5))
    # plt.imshow(img_cropped, cmap='gray')


    # print('MIN_PLATE_RATIO: {}, '.format((img_cropped.shape[1] / img_cropped.shape[0])))

    # if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO or img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO > MAX_PLATE_RATIO:
    #     continue

    plate_imgs.append(img_cropped)
    plate_infos.append({
        'x': int(plate_cx - plate_width / 2),
        'y': int(plate_cy - plate_height / 2),
        'w': int(plate_width),
        'h': int(plate_height)
    })

    # cv2.imwrite(save_file, img)

    # plt.subplot(len(matched_result), 1, i + 1)
    # plt.imshow(img_cropped, cmap='gray')
    # plt.show()

#
# # #Another Thresholding
# longest_idx, longest_text = -1, 0
#
#
# for i, plate_img in enumerate(plate_imgs):
#     plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
#     _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#
#     # plt.figure(figsize=(6, 5))
#     # plt.imshow(plate_img, cmap='gray')
#
#     # find contours again (same as above)
#     contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)
#
#     plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
#     plate_max_x, plate_max_y = 0, 0
#
#     for contour in contours:
#         x, y, w, h = cv2.boundingRect(contour)
#
#         area = w * h
#         ratio = w / h
#
#         if area > MIN_AREA \
#                 and w > MIN_WIDTH and h > MIN_HEIGHT \
#                 and MIN_RATIO < ratio < MAX_RATIO:
#             if x < plate_min_x:
#                 plate_min_x = x
#             if y < plate_min_y:
#                 plate_min_y = y
#             if x + w > plate_max_x:
#                 plate_max_x = x + w
#             if y + h > plate_max_y:
#                 plate_max_y = y + h
#     # print(img_result)
#     img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
#
# # print(img_result)

#이미지 저장 코드
# idx = 0
# for img in plate_imgs:
#     plt.figure(figsize=(3, 2))
#     cv2.imwrite("car{}.png".format(idx), img);
#     idx += 1;


trainedData = 'tesseract/kor.traineddata'
pytesseract.pytesseract.tesseract_cmd ='C:/Program Files/Tesseract-OCR/tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd ='tesseract/tesseract-ocr-w64-setup-v5.2.0.20220712.exe'

# img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
# _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT,
#                                 value=(0, 0, 0))

# chars = pytesseract.image_to_string(plate_imgs[0], lang='eng')
# chars = pytesseract.image_to_string(img_result, lang='kor', config='--psm 7 --oem 0')

# for img in plate_imgs:
#     plt.figure(figsize=(3, 2))
#     plt.imshow(cv2.threshold(img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU), cmap='gray')


plate_chars = []
for img in plate_imgs:
    plt.figure(figsize=(3, 2))
    plt.imshow(img, cmap='gray')
    reader = easyocr.Reader(['ko', 'en'], gpu=False)  # need to run only once to load model into memory
    result = reader.readtext(img)
    plate_chars.append(result[0][-2]);
    # print(result[0][-2])


#     #pytesseract
#     char = pytesseract.image_to_string(img, lang='eng')
#     char = char.replace('\n','')
#     plate_chars.append(char)
#
print(plate_chars)






# result_chars = ''
# has_digit = False
# for c in chars:
#     if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
#         if c.isdigit():
#             has_digit = True
#         result_chars += c

# print(result_chars)
# plate_chars.append(result_chars)

# if has_digit and len(result_chars) > longest_text:
#     longest_idx = i

# plt.subplot(len(plate_imgs), 1, i + 1)
# plt.imshow(img_result, cmap='gray')
plt.show()