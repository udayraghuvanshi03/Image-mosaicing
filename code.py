import os
import numpy as np
import cv2
import sys

np.set_printoptions(threshold=sys.maxsize)


class ImageMosaicing:
    def __init__(self, array_of_images: list, type_of_derivative_filter: str, hc_window_size: tuple,
                 ncc_threshold: float, ncc_window: tuple, ransac_iterations: int):
        self.array_of_images = array_of_images
        self.type_of_derivative_filter = type_of_derivative_filter
        self.window_size = hc_window_size
        self.threshold = ncc_threshold
        self.ncc_window = ncc_window
        self.iterations = ransac_iterations

    def derivative(self):
        i_x = []
        i_y = []
        for image in self.array_of_images:
            image = cv2.boxFilter(image, -1, (3, 3))
            if self.type_of_derivative_filter.lower() == "sobel":
                sobel_mask_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                sobel_mask_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                i_x.append(cv2.filter2D(image, ddepth=-1, kernel=sobel_mask_x))
                i_y.append(cv2.filter2D(image, ddepth=-1, kernel=sobel_mask_y))
            elif self.type_of_derivative_filter.lower() == "prewitt":
                prewitt_mask_x = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
                prewitt_mask_y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
                i_x.append(cv2.filter2D(image, ddepth=-1, kernel=prewitt_mask_x))
                i_y.append(cv2.filter2D(image, ddepth=-1, kernel=prewitt_mask_y))
            else:
                print("Incorrect input")
                return
        return i_x, i_y

    def harris_corner_detector(self, k=0.06):
        i_x, i_y = self.derivative()
        harris_r_array = []
        for ind in range(len(i_x)):
            fx = i_x[ind]
            fy = i_y[ind]
            harris_r = np.zeros(np.shape(fx))
            rows, columns = np.shape(fx)[0], np.shape(fx)[1]
            for row in range(rows):
                for column in range(columns):
                    i = row - self.window_size[0] // 2
                    j = column - self.window_size[0] // 2
                    i_x_2 = 0
                    i_y_2 = 0
                    i_x_y = 0
                    while i <= row + self.window_size[0] // 2:
                        while j <= column + self.window_size[0] // 2:
                            if 0 < i < rows and 0 < j < columns:
                                i_x_2 += np.square(fx[i][j])
                                i_y_2 += np.square(fy[i][j])
                                i_x_y += fx[i][j] * fy[i][j]
                            j += 1
                        i += 1
                    m = np.array([[i_x_2, i_x_y], [i_x_y, i_y_2]])
                    r_score = np.linalg.det(m) - k * (np.square(m[0][0] + m[1][1]))
                    if self.threshold < r_score:
                        harris_r[row][column] = r_score
            harris_r_array.append(harris_r)
        return harris_r_array

    def non_max_supression(self):
        harris_r_array = self.harris_corner_detector()
        nms_harris_arr = []
        pix_dist = 5
        h, w = np.shape(harris_r_array[0])
        for ind in harris_r_array:
            nms_harris = np.zeros((h, w))
            for i in range(0, h, 64):
                for j in range(0, w, 64):
                    r_array = []
                    for m in range(i + pix_dist, i + 64 - pix_dist):
                        for n in range(j + pix_dist, j + 64 - pix_dist):
                            if m < h and n < w:
                                if ind[m, n] > 0 and ind[m, n] == np.max(
                                        ind[m - pix_dist:m + pix_dist + 1, n - pix_dist:n + pix_dist + 1]):
                                    r_array.append((ind[m, n], m, n))
                    if len(r_array) < 10:
                        for p in range(len(r_array)):
                            nms_harris[r_array[p][1], r_array[p][2]] = r_array[p][0]
                    else:
                        r_array = sorted(r_array, reverse=True)
                        for p in range(10):
                            nms_harris[r_array[p][1], r_array[p][2]] = r_array[p][0]

            nms_harris_arr.append(nms_harris)
        return nms_harris_arr

    def find_correspondences(self):
        corners_array = self.non_max_supression()
        corners_0 = corners_array[0]
        corners_1 = corners_array[1]
        rows, columns = np.shape(corners_0)[0], np.shape(corners_0)[1]
        correspondences_picture0 = []
        correspondences_picture1 = []
        for row in range(rows):
            for column in range(columns):
                if corners_0[row][column] > 0:
                    a = row - (self.ncc_window[0] // 2)
                    b = row + (self.ncc_window[0] // 2) + 1
                    c = column - (self.ncc_window[0] // 2)
                    d = column + (self.ncc_window[0] // 2) + 1
                    if 0 < a < rows - self.ncc_window[0] and 0 < c < columns - self.ncc_window[0]:
                        template = np.array(self.array_of_images[0][a:b, c:d])
                        max_ncc = 0
                        max_row = None
                        max_column = None
                        for row2 in range(rows):
                            for column2 in range(columns):
                                if corners_1[row2][column2] > 0:
                                    a2 = row2 - (self.ncc_window[0] // 2)
                                    b2 = row2 + (self.ncc_window[0] // 2) + 1
                                    c2 = column2 - (self.ncc_window[0] // 2)
                                    d2 = column2 + (self.ncc_window[0] // 2) + 1
                                    if 0 < a2 < rows - self.ncc_window[0] and 0 < c2 < columns - self.ncc_window[0]:
                                        match_to = np.array(self.array_of_images[1][a2:b2, c2:d2])
                                        f = (match_to - match_to.mean()) / (match_to.std() * np.sqrt(match_to.size))
                                        g = (template - template.mean()) / (template.std() * np.sqrt(template.size))
                                        product = f * g
                                        stds = np.sum(product)
                                        if stds > max_ncc:
                                            max_ncc = stds
                                            max_row = row2
                                            max_column = column2
                        if max_row is not None:
                            if max_ncc > 0.90:
                                correspondences_picture0.append([row, column])
                                correspondences_picture1.append([max_row, max_column])
        return correspondences_picture0, correspondences_picture1

    def feature_matching(self, img_l, img_r, matching_type: str):
        corres_0, corres_1 = self.find_correspondences()
        img_l = np.copy(img_l)
        img_r = np.copy(img_r)
        shift = np.shape(img_r)[1]
        full_img = np.concatenate((img_l, img_r), axis=1)
        if matching_type == 'before ransac':
            for i in range(len(corres_0)):
                color = list(np.random.random_sample(size=3) * 256)
                cv2.line(full_img, (corres_0[i][1], corres_0[i][0]), (corres_1[i][1] + shift, corres_1[i][0]), color,
                         thickness=1)

        elif matching_type == 'after ransac':
            inliers_0, inliers_1 = self.ransac()[1], self.ransac()[2]
            for i in range(len(corres_0)):
                if corres_0[i] in inliers_0:
                    cv2.line(full_img, (corres_0[i][1], corres_0[i][0]), (corres_1[i][1] + shift, corres_1[i][0]),
                             (0, 255, 0), thickness=1)
                else:
                    cv2.line(full_img, (corres_0[i][1], corres_0[i][0]), (corres_1[i][1] + shift, corres_1[i][0]),
                             (0, 0, 255), thickness=1)
        return full_img

    @staticmethod
    def find_homography(pts_src, pts_dst):
        pts_src = np.array(pts_src)
        pts_dst = np.array(pts_dst)

        # forming A matrix
        a_matrix = np.zeros((2 * len(pts_src), 9))
        for i in range(len(pts_src)):
            a_matrix[2 * i, 0] = pts_src[i][0]
            a_matrix[2 * i, 1] = pts_src[i][1]
            a_matrix[2 * i, 2] = 1
            a_matrix[2 * i, 6] = -pts_src[i][0] * pts_dst[i][0]
            a_matrix[2 * i, 7] = -pts_src[i][1] * pts_dst[i][0]
            a_matrix[2 * i, 8] = -pts_dst[i][0]

            a_matrix[2 * i + 1, 3] = pts_src[i][0]
            a_matrix[2 * i + 1, 4] = pts_src[i][1]
            a_matrix[2 * i + 1, 5] = 1
            a_matrix[2 * i + 1, 6] = -pts_src[i][0] * pts_dst[i][1]
            a_matrix[2 * i + 1, 7] = -pts_src[i][1] * pts_dst[i][1]
            a_matrix[2 * i + 1, 8] = -pts_dst[i][1]

        # Transposing A
        a_transpose = np.transpose(a_matrix)
        final_a_matrix = np.dot(a_transpose, a_matrix)
        u, d, u_transpose = np.linalg.svd(final_a_matrix, full_matrices=True)

        # h is the column of U associated with smallest eigen in D, which is the last value in D
        h = (u[:, 8])
        h = np.reshape(h, (3, 3))
        return h

    def ransac(self):
        corres_0, corres_1 = self.find_correspondences()
        max_inliers = 0
        max_inliers_0=[]
        max_inliers_1=[]
        homography = None
        for it in range(self.iterations):
            random_sampled_indx = []
            pts_src = []
            pts_dst = []
            i = 0
            while i < 4:
                random_sample = int(np.random.randint(0, high=len(corres_0) - 1, size=1, dtype=int))
                if random_sample not in random_sampled_indx:
                    pts_src.append(corres_0[random_sample])
                    pts_dst.append(corres_1[random_sample])
                    random_sampled_indx.append(random_sample)
                    i += 1

            h = self.find_homography(pts_src, pts_dst)
            j = 0
            inliers = 0
            inliers_0 = []
            inliers_1 = []
            while j < len(corres_0):
                src = np.transpose(np.array([[corres_0[j][0], corres_0[j][1], 1]]))
                est = np.matmul(h, src)
                if est[2][0] == 0:
                    j += 1
                    continue
                est = est / est[2][0]
                dist = np.sqrt(np.power(est[0] - corres_1[j][0], 2) + np.power(est[1] - corres_1[j][1], 2))
                if dist < 5:
                    inliers += 1
                    inliers_0.append(corres_0[j])
                    inliers_1.append(corres_1[j])
                j += 1

            if inliers > max_inliers:
                max_inliers = inliers
                max_inliers_0 = inliers_0
                max_inliers_1 = inliers_1
                if max_inliers > 3:
                    homography = self.find_homography(max_inliers_0, max_inliers_1)
        return homography,max_inliers_0,max_inliers_1

    def disp_img(self, array_of_rgb_images):
        harris_arr = self.harris_corner_detector()
        corner_imgs = []
        i = 0

    def image_stitching(self, img_1, img_2):

        h = np.linalg.inv(self.ransac()[0])
        y, x = img_2.shape[:2]
        p_1 = [[0], [0], [1]]
        p_2 = [[0], [x - 1], [1]]
        p_3 = [[y - 1], [0], [1]]
        p_4 = [[y - 1], [x - 1], [1]]

        p_1 = np.matmul(h, p_1)
        p_2 = np.matmul(h, p_2)
        p_3 = np.matmul(h, p_3)
        p_4 = np.matmul(h, p_4)

        p_1 = p_1 / p_1[2][0]
        p_2 = p_2 / p_2[2][0]
        p_3 = p_3 / p_3[2][0]
        p_4 = p_4 / p_4[2][0]

        y_min = round(min(p_1[0], p_2[0], p_3[0], p_4[0])[0])
        x_min = round(min(p_1[1], p_2[1], p_3[1], p_4[1])[0])
        y_max = round(max(p_1[0], p_2[0], p_3[0], p_4[0])[0])
        x_max = round(max(p_1[1], p_2[1], p_3[1], p_4[1])[0])

        pano = np.zeros((y_max - y_min + 10, x_max - x_min + round(np.shape(img_1)[1] * 0.3), 3), dtype=np.uint8)
        pano[abs(y_min):np.shape(img_1)[0] + abs(y_min), :np.shape(img_1)[1], :] = img_1

        for row in range(pano.shape[0]):
            for column in range(pano.shape[1]):
                est = np.matmul(np.linalg.inv(h), np.array([[row], [column], [1]]))
                est = est / est[2][0]
                est_y = round(est[0][0])
                est_x = round(est[1][0])
                if 0 <= est_x < img_2.shape[1] and 0 <= est_y + y_min < img_2.shape[0]:
                    if pano[row][column][:].any() > 0:
                        w_1 = min(row, column, img_1.shape[0] - row - y_min, img_1.shape[1] - column)
                        w_2 = min(est_y + y_min, est_x, img_2.shape[0] - (est_y + y_min),
                                  img_2.shape[1] - est_x)
                        if w_1 + w_2 != 0:
                            feather_1 = np.array(pano[row][column][:]).astype(np.uint64) * w_1 / (w_1 + w_2)
                            feather_2 = np.array(img_2[est_y + y_min][est_x][:]).astype(np.uint64) * w_2 / (w_1 + w_2)
                            pano[row][column][:] = (feather_1 + feather_2).astype(np.uint8)

                        else:
                            pano[row][column][:] = 0.5 * np.array(pano[row][column][:]) \
                                                   + 0.5 * np.array(img_2[abs(est_y + y_min)][est_x][:])
                    else:
                        pano[row][column][:] = img_2[abs(est_y + y_min)][est_x][:]

        return pano


if __name__ == "__main__":
    path_to_images = r'C:\Users\udayr\PycharmProjects\CVfiles\project2\DanaHallWay1\DanaHallWay1'
    imgs_arr = []
    imgs_arr2 = []
    for img_name in os.listdir(path_to_images):
        img = cv2.imread(path_to_images + '\\' + img_name, 0)  # Read and convert the image to grayscale
        imgs_arr.append(np.asarray(img).astype(float))  # Create an array of all the images
    for img_name2 in os.listdir(path_to_images):
        img = cv2.imread(path_to_images + '\\' + img_name2)  # Read the image
        imgs_arr2.append(np.asarray(img))
    image_to_analyze = [0, 1]
    rgb_1 = imgs_arr2[image_to_analyze[0]]
    rgb_2 = imgs_arr2[image_to_analyze[1]]
    g_1 = imgs_arr[image_to_analyze[0]]
    g_2 = imgs_arr[image_to_analyze[1]]
    inst = ImageMosaicing([g_1, g_2], "prewitt", (7, 7), 10000, (7, 7), 1000).image_stitching(rgb_1,rgb_2)
    # canvas = inst.image_stitching(rgb_1, rgb_2)
    cv2.imshow('Mosaic', inst)
    cv2.waitKey(0)
