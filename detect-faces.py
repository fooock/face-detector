''' Script used to detect images with human faces '''
import sys
import os
import cv2

from timeit import default_timer as timer


BANNER = """   
   __                     _      _            _             
  / _|                   | |    | |          | |            
 | |_ __ _  ___ ___    __| | ___| |_ ___  ___| |_ ___  _ __ 
 |  _/ _` |/ __/ _ \  / _` |/ _ \ __/ _ \/ __| __/ _ \| '__|
 | || (_| | (_|  __/ | (_| |  __/ ||  __/ (__| || (_) | |   
 |_| \__,_|\___\___|  \__,_|\___|\__\___|\___|\__\___/|_|   
                                                            
                                                            """

RESULT_FILE = "detected_faces.txt"

def to_grayscale(img):
    '''
    Convert the given image to grayscale. This is necessary because many operations in
    OpenCV are done in grayscale for performance reasons
    '''
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def create_classifier():
    ''' Create the cascade classifier to detect faces in images '''
    return cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

def detect_face(gray_img, classifier):
    '''
    Try to detect a face for the given gray image. If a face is found this method return
    true, otherwise return false
    '''
    faces = classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
    return len(faces) > 0

def read_files_from(path):
    ''' Read the files in the given path to find faces '''
    print("[+] Search images in {}".format(path))
    images = []
    for (dirname, _, filenames) in os.walk(path):
        for file in filenames:
            images.append(os.path.join(dirname, file))
    return images

def write_report(analyzed_images):
    ''' Write to one file the absolute path of the images that contains faces '''
    with open(RESULT_FILE, 'w') as f:
        for face in analyzed_images:
            f.write(face + "\n")
    f.close()
    print("\n[+] Write report to {}".format(RESULT_FILE))


if __name__ == "__main__":
    print(BANNER)
    print("[+] OpenCV version: {}".format(cv2.__version__))
    # Check program arguments
    NUM_ARGS = len(sys.argv[1])
    if NUM_ARGS == 0:
        print("First argument need to be the directory where the pictures are")
        sys.exit()

    # check if the given directory exists
    PATH = sys.argv[1]
    if not os.path.exists(PATH):
        print("[+] The given directory does not exists")
        sys.exit()

    # Check if exists images in the given directory
    IMGS = read_files_from(PATH)
    NUM_IMGS = len(IMGS)
    if NUM_IMGS == 0:
        print("[+] No images found!")
        sys.exit()

    CL = create_classifier()

    # start now
    START = timer()
    # Store here the absolute path of the images that contains faces
    FOUND_FACES = []
    ANALYZED_IMGS = 0
    for image in IMGS:
        ANALYZED_IMGS += 1
        original_img = cv2.imread(image)
        gray_img = to_grayscale(original_img)
        # Check image
        if detect_face(gray_img, CL):
            FOUND_FACES.append(image)
        print("[+] Analyzed {} images of {} | Found {} faces"
              .format(ANALYZED_IMGS, len(IMGS), len(FOUND_FACES)), end="\r")

    # write result to file
    write_report(FOUND_FACES)

    # show time elapsed
    ELAPSED = timer() - START
    print("[+] Finished in {} seconds".format(round(ELAPSED)))
