import argparse
import cv2
import numpy as np
import pytesseract
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--location', help='Image Location:')
    args = parser.parse_args()

    image_location = args.location

    if not image_location:
        print('Image location is not provided!')
        return

    print(f'Image location is: {image_location}')

    captcha_image = cv2.imread(image_location, cv2.IMREAD_GRAYSCALE)
    captcha_image = cv2.medianBlur(captcha_image, 5)

    threshold = cv2.adaptiveThreshold(captcha_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)
    kernel_erosion = np.ones((3, 3), np.uint8)
    kernel_dilation = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(threshold, kernel_erosion, iterations=1)
    dilation = cv2.dilate(erosion, kernel_dilation, iterations=1)

    new_image = 'apply-effect.png'

    cv2.imwrite(new_image, dilation)

    custom_config = ('--oem 3 --psm 11 -c '
                     'tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    img = cv2.imread(new_image)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    text = pytesseract.image_to_string(thr, config=custom_config)

    file_name = os.path.splitext(os.path.basename(image_location))[0]
    txt_file_name = f'{file_name}.txt'
    with open(txt_file_name, 'w') as f:
        f.write(text)

    os.remove(new_image)


if __name__ == "__main__":
    main()
