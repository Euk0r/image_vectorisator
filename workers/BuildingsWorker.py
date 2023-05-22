import math
from xmlrpc.client import MAXINT
import cv2
import os
import numpy as np
import pickle
from PIL import Image
import shapefile #pyshp
from skimage.measure import find_contours, approximate_polygon

FN_RECTS='rects.dat'


class BuildingsWorker:
    
    def vectorise(self, full_name_img, dir_image, dir_to_save, fn_rects=FN_RECTS, fn_foo='graph-foo.png'):
        type_vec = 0
        name_img = os.path.splitext(full_name_img)[0]

        temp_img = cv2.imread(dir_image, cv2.IMREAD_COLOR) #for testing
        temp_img = cv2.GaussianBlur(temp_img, (0, 0), 2.5) #for testing
        temp_img = cv2.addWeighted(temp_img, 1.5, temp_img, -0.1, 0) #for testing
        edges = cv2.Canny(temp_img,100,200) #for testing

        path_to_rects = os.path.join(dir_to_save, name_img)
        if not os.path.exists(path_to_rects):
            os.makedirs(path_to_rects)
        path_to_foo = os.path.join(dir_to_save, name_img)
        if not os.path.exists(path_to_foo):
            os.makedirs(path_to_foo)    

        return_images = []
        new_name = "edges_" + full_name_img
        new_path = os.path.join(dir_to_save, name_img, new_name)
        im = Image.fromarray(edges)
        im.save(new_path)
        return_images.append((new_name, new_path))
        prethreshold = cv2.imread(dir_image, cv2.IMREAD_GRAYSCALE)
        _,threshold = cv2.threshold(prethreshold, 105, 255, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(path_to_foo, 'threshold.png'), threshold) #for testing
        contours,_=cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        temp_img2 = np.copy(temp_img)
        for cnt in contours:
            cv2.drawContours(temp_img2,[cnt],0,(0,0,255),2)
        cv2.imwrite(os.path.join(path_to_foo, 'contours.png'), temp_img2) #for testing
        list_rects = []
        # Поиск в каждом регионе
        for cnt in contours :
            area = cv2.contourArea(cnt)
            # Составление списка регионов на основе их области.
            if area > 50: 
                approx = cv2.approxPolyDP(cnt, 0.0075 * cv2.arcLength(cnt, True), True) #для непрямоугольных зданий
                approx2 = cv2.approxPolyDP(cnt, 0.032 * cv2.arcLength(cnt, True), True) #для проверки прямоугольных зданий
                if type_vec == 0:
                    if(len(approx2) in range(0, 5)):
                        rect = cv2.minAreaRect(approx) # пытаемся вписать прямоугольник
                        box = cv2.boxPoints(rect) # поиск четырех вершин прямоугольника
                        box = np.int0(box) # округление координат
                        #angle = get_angle_from_box(box)
                        #list_cnt.append(approx)
                        #list_ang.append(angle)
                        #center = (int(rect[0][0]),int(rect[0][1]))
                        #area = int(rect[1][0]*rect[1][1]) # вычисление площади
                        cv2.drawContours(temp_img,[box],0,(0,0,255),2) # рисуем прямоугольник
                        list_rects.append(box)
                        #cv2.circle(temp_img, center, 5, (255,255,0), 2) # рисуем маленький кружок в центре прямоугольника
                        # выводим в кадр величину угла наклона
                        #cv2.putText(temp_img, "%d" % int(angle), (center[0]+10, center[1]-10), 
                        #        cv2.FONT_HERSHEY_SIMPLEX, 1, (150,150,0), 2)
                    else:
                        cv2.drawContours(temp_img,[approx],0,(0,0,255),2) # рисуем прямоугольник
                        list_rects.append(approx.reshape(int(approx.size/2),2))
                elif type_vec == 1:
                    gray = np.float32(threshold)
                    mask = np.zeros(gray.shape, dtype="uint8")
                    cv2.fillPoly(mask, [approx], (255,255,255))
                    dst = cv2.cornerHarris(mask,10,3,0.028)
                    ret, dst = cv2.threshold(dst,0.1*dst.max(),255,0)
                    dst = np.uint8(dst)
                    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
                    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
                    corners = cv2.cornerSubPix(gray,np.float32(centroids),(5,5),(-1,-1),criteria)                    
                    corners = np.delete(corners, 0, 0)
                    if len(corners) >= 4:
                        #for corner in corners:
                        #    x,y = corner.ravel()
                        #    cv2.circle(temp_img,(int(x),int(y)),5,(0,255,0),3)
                        leftmost_point, *_, rightmost_point = sorted(corners, key=lambda lst: lst[0])
                        x1, y1 = leftmost_point
                        x2, y2 = rightmost_point

                        border_slope = (y2 - y1) / (x2 - x1)

                        points_below_border = [(x1, y1)]
                        points_above_border = [(x2, y2)]

                        for x, y in corners:
                            border = border_slope * (x - x1) + y1
                            if y < border:
                                points_below_border.append((x, y))
                            elif y > border:
                                points_above_border.append((x, y))

                        points_below_border.sort()
                        points_above_border.sort(reverse=True)
                        points_above_border.append((x1, y1))
                        merged_points = points_below_border + points_above_border                    
                        contour = np.array(merged_points, dtype=np.int32)
                        cnt_approx = cv2.approxPolyDP(contour, 0.0075 * cv2.arcLength(cnt, True), True)
                        cv2.drawContours(temp_img,[cnt_approx],0,(0,0,255),2)
                        list_rects.append(cnt_approx)                        
        with open(os.path.join(path_to_rects, fn_rects), 'wb') as f:
            pickle.dump(list_rects,f)

        cv2.imwrite(os.path.join(path_to_foo, fn_foo), temp_img)

        return_images.append((name_img + '_' + fn_foo, os.path.join(path_to_foo, fn_foo)))
        return return_images

    def save_shp(self, path_to_rects, path_to_save, offset=None, filename='shpfile.shp', fn_rects=FN_RECTS):
        with open(os.path.join(path_to_rects, fn_rects), 'rb') as fp:
            rects = pickle.load(fp)
        w = shapefile.Writer(path_to_save, shapeType=shapefile.POLYGON)
        w.field('id','N', 18)
        for id, cnt in enumerate(rects):
            poly_list = np.array(cnt).tolist()
            for i in range (0, len(poly_list)):
                poly_list[:][i][1] = -poly_list[:][i][1]
                if offset:
                    poly_list[:][i][0] += offset[0]
                    poly_list[:][i][1] += offset[1]
            w.poly([poly_list])
            w.record(id)
        w.close()
        return path_to_save

def min_distance(contour, contourOther):
    distanceMin = MAXINT
    for xA, yA in contour[0]:
        for xB, yB in contourOther[0]:
            distance = ((xB-xA)**2+(yB-yA)**2)**(1/2) # формула расстояния
            if (distance < distanceMin):
                distanceMin = distance
    return distanceMin

def find_closest_and_rotate(list_cnt, list_ang): #for testing
    amount = len(list_cnt)
    list_cnt_new = []
    if amount > 1:
        for i in range(0, amount - 1):
            index_min = -1
            min = MAXINT
            for j in range(i + 1, amount):
                temp = min_distance(list_cnt[i], list_cnt[j])
                if (temp < min): 
                    min = temp
                    index_min = j
            avg_ang = (list_ang[i] + list_ang[index_min]) / 2
            if abs(avg_ang > 5): list_cnt_new.append(list_cnt[i])
            else: 
                list_cnt_new.append(rotate_contour(list_cnt[i], avg_ang - list_ang[i]))
    return list_cnt_new

def cart2pol(x, y):
    theta = np.arctan2(y, x)
    rho = np.hypot(x, y)
    return theta, rho
def pol2cart(theta, rho):
    x = rho * np.cos(theta)
    y = rho * np.sin(theta)
    return x, y

def rotate_contour(cnt, angle):
    M = cv2.moments(cnt)
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])

    cnt_norm = cnt - [cx, cy]
    
    coordinates = cnt_norm[:, 0, :]
    xs, ys = coordinates[:, 0], coordinates[:, 1]
    thetas, rhos = cart2pol(xs, ys)
    
    thetas = np.rad2deg(thetas)
    thetas = (thetas + angle) % 360
    thetas = np.deg2rad(thetas)
    
    xs, ys = pol2cart(thetas, rhos)
    
    cnt_norm[:, 0, 0] = xs
    cnt_norm[:, 0, 1] = ys

    cnt_rotated = cnt_norm + [cx, cy]
    cnt_rotated = cnt_rotated.astype(np.int32)

    return cnt_rotated

def get_angle_from_box(box, other_edges=False):
    'вычисление координат двух векторов, являющихся сторонами прямоугольника'
    if not other_edges:
        edge1 = np.int0((box[1][0] - box[0][0],box[1][1] - box[0][1]))
        edge2 = np.int0((box[2][0] - box[1][0],box[2][1] - box[1][1]))
    else:
        edge1 = np.int0((box[2][0] - box[0][0],box[1][1] - box[1][1]))
        edge2 = np.int0((box[1][0] - box[1][0],box[2][1] - box[0][1]))
    # выясняем какой вектор больше
    usedEdge = edge1
    if cv2.norm(edge2) > cv2.norm(edge1):
        usedEdge = edge2
    reference = (1,0) # горизонтальный вектор, задающий горизонт

    # вычисляем угол между самой длинной стороной прямоугольника и горизонтом
    angle = 180.0/math.pi * math.acos((reference[0]*usedEdge[0] + reference[1]*usedEdge[1]) / (cv2.norm(reference) *cv2.norm(usedEdge)))
    if angle > 45 and not other_edges:
        angle = get_angle_from_box(box, True)
    return angle

def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]
def ang(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360

    if ang_deg-180>=0:
        # As in if statement
        return 360 - ang_deg
    else: 
        return ang_deg

def rotate(origin, point, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy

def vectorize_regions(dir_image, threshold: float = 0.5):
    """
    Vectorizes lines from a binarized array.

    Args:
        im (np.ndarray): Array of shape (H, W) with the first dimension
                         being a probability distribution over the region.
        threshold (float): Threshold for binarization

    Returns:
        [[x0, y0, ... xn, yn], [xm, ym, ..., xk, yk], ... ]
        A list of lists containing the region polygons.
    """
    temp_img = cv2.imread(dir_image, cv2.IMREAD_COLOR)
    prethreshold = cv2.imread(dir_image, cv2.IMREAD_GRAYSCALE)
    _,im = cv2.threshold(prethreshold, 105, 255, cv2.THRESH_BINARY)
    bin = im > threshold
    contours = find_contours(bin, 0.5, fully_connected='high', positive_orientation='high')
    if len(contours) == 0:
        return contours
    approx_contours = []
    for contour in contours:
        approx_contours.append(approximate_polygon(contour[:,[1,0]], 1).astype('uint').tolist())
    for cnt in approx_contours:
        cv2.drawContours(temp_img,[np.array(cnt)],0,(0,0,255),2)
            
    cv2.imshow('vec_buildings', temp_img) 
    
    # Выход из окна, если на клавиатуре нажата клавиша «q».
    if cv2.waitKey(0) & 0xFF == ord('q'): 
        cv2.destroyAllWindows()
    return approx_contours

if __name__ == "__main__":
    bw = BuildingsWorker()