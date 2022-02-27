import tensorflow as tf
import cv2
import numpy as np
import os
import sys

sys.setrecursionlimit(1000)

class ImageRegconition:
    def __init__(self, model_path='digits_model.h5'):
        assert os.path.isfile(model_path), "Model does not exist"
        assert os.path.split(model_path)[-1].endswith('.h5'), "The file is not in .h5 type"
        self.model = tf.keras.models.load_model(model_path)
    
    def detect_board(self, content):
        self.image = cv2.imdecode(content, cv2.IMREAD_UNCHANGED)
        assert self.image is not None, "Image does not exist"
        base_image = cv2.resize(self.image, (450, 450))
        threshold = self.__preprocess(base_image)

        # Finding the outline of the sudoku puzzle in the image
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(base_image, contours, -1, (0, 255, 0), 3)

        biggest_contour, max_area = self.__main_outline(contours)
        warped = self.__four_points_transfrom(base_image, biggest_contour)

        warped = cv2.resize(warped, (450, 450))
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        cells = np.array(self.__crop_cells(self.__split_cells(warped)))

        cells = cells.reshape(cells.shape + (1, ))
        cells = tf.image.resize(cells, (28, 28))

        predictions = self.model.predict(cells)
        result = np.argmax(predictions, axis=1)
        
        #self.board = result
        return np.reshape(result, (9, 9))
        
    def __crop_cells(self, cells):
        cells_cropped = []
        for image in cells:
            image = image[5:45, 5:45]
            cells_cropped.append(image)
        return cells_cropped

    def __preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3, 3), 6)
        threshold = cv2.adaptiveThreshold(blur, 255, 1, 1, 11, 2)
        return threshold
    
    def __main_outline(self, contours):
        biggest_contour = np.array([])
        max_area = 0
        
        for i in contours:
            area = cv2.contourArea(i)
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area:
                biggest_contour = approx
                max_area = area
        return biggest_contour, max_area
    
    def __order_points(self, pts):
        rect = np.zeros((4,2 ), dtype="float32")

        pts = pts.sum(axis=1)
        s = pts.sum(axis=1)

        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis = 1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect
    
    def __four_points_transfrom(self, image, pts):
        rect = self.__order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth-1, 0],
            [maxWidth-1, maxHeight-1],
            [0, maxHeight-1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped
    
    def __split_cells(self, image):
        rows = np.vsplit(image, 9)
        boxes = []
        for r in rows:
            cols = np.hsplit(r, 9)
            for box in cols:
                boxes.append(box)
        return boxes

class Solver:
    def __init__(self, solution_limit=1):
        self.solution_limit = solution_limit
    
    def solve(self, board):
        mark_3x3 = np.zeros(shape=(3,3,9), dtype='bool')
        mark_row = np.zeros(shape=(9, 9), dtype='bool')
        mark_col = np.zeros(shape=(9, 9), dtype='bool')

        count = 0
        STOP_RECURSION = False

        # Init mark array
        for i in range(9):
            for j in range(9):
                cell_value = board[i,j]
                if cell_value != 0:
                    if mark_row[i, cell_value - 1] or mark_col[j, cell_value-1] or mark_3x3[i//3,j//3,cell_value-1]:
                        raise Exception('Invalid Sudoku board, please re-fill the board!')
                    else:
                        mark_row[i,cell_value - 1] = True
                        mark_col[j,cell_value - 1] = True
                        mark_3x3[i//3,j//3,cell_value-1] = True

    
        def solve_recurion(i, j):
            nonlocal count, STOP_RECURSION

            if i < 9 and j < 9:
                if board[i, j] == 0:
                    for z in range(1, 10):
                        if (not mark_3x3[i//3,j//3,z-1]) and (not mark_row[i, z-1]) and (not mark_col[j, z-1]):
                            mark_3x3[i//3, j//3, z-1] = True
                            mark_row[i, z-1] = True
                            mark_col[j, z-1] = True
                            board[i, j] = z

                            if not STOP_RECURSION:
                                yield from solve_recurion(i, j + 1)
                            
                            mark_3x3[i//3, j//3, z-1] = False
                            mark_row[i, z-1] = False
                            mark_col[j, z-1] = False
                            board[i, j] = 0
                else:
                    yield from solve_recurion(i, j+1)
            elif i < 9 and j >= 9:
                yield from solve_recurion(i+1, 0)
            else:
                if count + 1 < self.solution_limit:
                    count += 1
                    yield board
                elif count + 1 == self.solution_limit:
                    count += 1
                    STOP_RECURSION = True
                    yield board

        return solve_recurion(0, 0)
