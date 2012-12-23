import cv2
import cv
import numpy as np
from math import *
import sys
import gc
from scipy.interpolate import griddata
import configs
from threading import Thread, Lock
from board import Board
import traceback

def POST(name, img):
    cv2.namedWindow(name)
    cv2.imshow(name, img)

def intersect(line1, line2):

    x_1, y_1, x_2, y_2 = line1
    x_3, y_3, x_4, y_4 = line2

    denom = float((x_1 - x_2) * (y_3 - y_4) - (y_1 - y_2) * (x_3 -
    x_4))
    if denom == 0:
        return None
    x = ((x_1 * y_2 - y_1 * x_2) * (x_3 - x_4) - (x_1 - x_2) *
    (x_3 * y_4 - y_3 * x_4)) / denom
    y = ((x_1 * y_2 - y_1 * x_2) * (y_3 - y_4) - (y_1 - y_2) *
    (x_3 * y_4 - y_3 * x_4)) / denom

    if x < -50 or x > 1000 or y < -50 or y > 1000:
        return None

    return int(x), int(y)

def distance((x1,y1), (x2,y2)):
    return sqrt((x1-x2)**2 + (y1-y2)**2)


def get_sub_image(image, x, y):
    dx = float(configs.SIZE-configs.LSTEP-configs.RSTEP) / 15
    dy = float(configs.SIZE-configs.BSTEP-configs.TSTEP) / 15
    xp = float(configs.LSTEP) + dx * x
    yp = float(configs.TSTEP) + dy * y

    return cv2.getRectSubPix(image, (int(dx + configs.PATCH_EXPAND), int(dy + configs.PATCH_EXPAND)), (int(xp + dx/2), int(yp + dy/2))) 



responses = None
samples = None
model = None

board_ar = [[] for x in range(0,15**2)]

def acc(x,y):
    return board_ar[y*15 +x]

def new_info(x,y,c):
    a = acc(x,y)
    a.insert(0,c)
    while len(a) > configs.CHAR_BUFFER_SIZE:
        a.pop()

def lookup_char(x,y):
    a = acc(x,y)
    d = {}
    for l in a:
        if l not in d:
            d[l] = 1
        else:
            d[l] = d[l] + 1
    dd = zip(d.values(), d.keys())
    dd.sort(reverse=True)

    if len(dd) == 0:
        return None
  

    if dd[0][1] == None and len(dd) >= 2:
        nc = dd[0][0]
        ncf = float(nc) / len(a)
        if ncf != 1:
            if configs.DEBUG and x == configs.COORD_X and y == configs.COORD_Y:
                print "nc IS %.2f" % ncf
        if ncf > configs.BLANK_REQ_PERCENT:
            return None
        else:
            dd.remove(dd[0])

    return dd[0][1]
    


if configs.TRAIN:
    print "Training mode!"
    if not configs.RELOAD:
        global samples, responses
        responses = []
        samples = np.empty((0,configs.TRAIN_SIZE**2))
    else:
        global samples, responses
        samples = np.loadtxt('generalsamples.data',np.float32)
        responses = np.loadtxt('generalresponses.data',np.float32)
        responses = responses.reshape((responses.size,1))
        responses = map(lambda x: x[0], list(responses))
else:
    global samples, responses, model

    print "Loading trained data"
    
    samples = np.loadtxt('generalsamples.data',np.float32)
    responses = np.loadtxt('generalresponses.data',np.float32)
    responses = responses.reshape((responses.size,1))

    print "Training model"

    model = cv2.KNearest()
    model.train(samples,responses)

    print "Model trained"



def classify_letter(image, x, y, draw=False, blank_board=None):
    image = cv2.resize(image, (128,128))
    if draw:
        POST("letter start", image)

    luv = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))
    l_chan = luv[2]

    #-----
    shift = configs.BLANK_PATCH_BL_SHIFT

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray =  cv2.getRectSubPix(gray, (configs.BLANK_DETECT_SIZE,configs.BLANK_DETECT_SIZE), (64-shift,64+shift)) 
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    mean, stddev = cv2.meanStdDev(gray)
    norm_mean = stddev / mean * 100
    
    if draw:
        POST("OC", gray)
        print "Mean is %.2f and stddev is %.2f; experimental norm mean is %.2f" % (mean, stddev, norm_mean)

    if norm_mean < configs.STD_DEV_THRESH:
        #square is blank!
        if draw:
            print "Dropped due to blank"
        if blank_board is not None:
            blank_board.set(x, y, mean)
        return None


    #-----

    if draw:
        POST("letter L", l_chan)

    blur = cv2.GaussianBlur(l_chan, (configs.LETTER_BLUR,configs.LETTER_BLUR), 0)

    thresh = cv2.adaptiveThreshold(blur, 255, 0, 1, configs.LETTER_THRESH, configs.LETTER_BLOCK)
    
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (7,7))
    thresh = cv2.dilate(thresh, element)

    if draw:
        POST("letter thresh", thresh)
    othresh = thresh.copy()

    contours,hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    im = image.copy()

    #Find large contour closest to center of image
    minc = None
    mindst = float("inf")
    for cnt in contours:
        sz = cv2.contourArea(cnt)
        if sz>820:
            [x,y,w,h] = cv2.boundingRect(cnt)
            d = abs(cv2.pointPolygonTest(cnt, (64,64), measureDist=True))
            cv2.circle(im, (64,64), 2, (0,255,0), thickness=3)

            if d < mindst:
                mindst = d
                minc = cnt

    if mindst > 50:
        if draw:
            print "Dropped due to contour distance"
        return None

    if minc is None:
        if draw:
            print "Dropped due to no contours"
        return None
            
    [x,y,w,h] = cv2.boundingRect(minc)
    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),1)
    if draw:
        POST("letter contour", im)
    
    #Detect triple word stuffs
    if w > h*configs.TEXT_RATIO:
        if draw:
            print "Dropped due to insufficient ratio"
        return None


    if w*h >= 128**2 * configs.MAX_FILL:
        if draw:
            print "Too much fill"
        return None


    #TODO: I is weird
    if float(h)/float(w) > 2.0:
        nw = int(h*0.7)
    else:
        nw = w

    trimmed = cv2.getRectSubPix(othresh, (nw,h), (int(x + float(w)/2), int(y + float(h)/2))) 

    trimmed = cv2.resize(trimmed, (configs.TRAIN_SIZE, configs.TRAIN_SIZE))
    
    if draw:
        POST("Trimmed letter", trimmed)
    
    sample = trimmed.reshape((1,configs.TRAIN_SIZE**2))

    if configs.TRAIN:
        global samples, responses
        print "What letter is this? (enter to stop, esc to skip)"
        o = cv2.waitKey(0)
        if o == 10:
            responses = np.array(responses,np.float32)
            responses = responses.reshape((responses.size,1))
            print "training complete"

            np.savetxt('generalsamples.data',samples)
            np.savetxt('generalresponses.data',responses)
            sys.exit(0)
        elif o == 27:
            print "Skipping, here's another..."
        else:
            x = chr(o).lower()
            print "You said it's a %s" % str(x)
            responses.append(ord(x)-96)
            samples = np.append(samples, sample, 0)
            print "Added to sample set"
    else:

        #classify!
        sample = np.float32(sample)
        retval, results, neigh_resp, dists = model.find_nearest(sample, k = 1)
        retchar = chr(int((results[0][0])) + 96)
        if retchar == '0':
            #Star character!
            if draw:
                print "Dropped due to star"
            return None
        return retchar

class IterSkip(Exception): #Using exceptions for loop control... so hacky...
    def __init__(self):
        pass


class ScrabbleVision(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.board = Board()
        self.l = Lock()
        self.started = False
        self.killed = False

    def get_current_board(self):
        with self.l:
            return self.board.copy()

    def kill(self):
        self.killed = True

    def run(self):

            vc = cv2.VideoCapture(-1)

            if vc.isOpened(): # try to get the first frame
                rval, frame_raw = vc.read()
            else:
                rval = False

            while rval:

                if self.killed:
                    print "Vision terminating"
                    return

                reload(configs)
                element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (configs.ERODE_RAD,configs.ERODE_RAD))
                element2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (configs.DILATE_RAD,configs.DILATE_RAD))

                #BEGIN PROCESSING
                try:

                    
                    frame = cv2.flip(frame_raw, flipCode=-1)
                    POST("RAW", frame)

                    luv = cv2.split(cv2.cvtColor(frame, cv2.COLOR_RGB2LUV))

                    v_chan = luv[2]

                    #POST("V", v_chan)

                    blur = cv2.GaussianBlur(v_chan, (configs.BLUR_RAD,configs.BLUR_RAD), 0)

                    #POST("blur", blur)
                    thresh = cv2.adaptiveThreshold(blur, 255, 0, 1, configs.BOARD_THRESH_PARAM, configs.BOARD_BLOCK_SIZE)

                    #POST("thresh", thresh)

                    erode = cv2.erode(thresh, element)
                    erode = cv2.dilate(erode, element2)
                    
                    POST("erode", erode)

                    erode_draw = frame.copy()
                    
                    contours,hierarchy = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    possible_corners = []

                    #Find large contour closest to center of image
                    for cnt in contours:
                        sz = cv2.contourArea(cnt)
                        if sz>75 and sz < 650:
                            ellipse = cv2.fitEllipse(cnt)
                            ((x,y), (w,h), r) = ellipse
                            ar = w / h if w > h else h / w
                            if ar > 1.8:
                                continue
                            pf = (w * h * 0.75) / sz
                            if pf > 1.5:
                                continue
                            cv2.ellipse(erode_draw,ellipse,(0,255,0),2)
                            possible_corners.append((x,y))


                    """


                    lines = cv2.HoughLinesP(erode, 1, 3.14/180, 300, minLineLength=200, maxLineGap=100)[0]
                    m,n = erode.shape

                    goodlines = lines[:40]

                    #Calculate intersection points
                    points = set()
                    CLUSTER_RAD = 50
                    clusters = []
                    for x in goodlines:
                        for y in goodlines:
                            i = intersect(x,y)
                            if i is not None:
                                added = False
                                for (ctr, st) in clusters:
                                    if distance(i, ctr) < CLUSTER_RAD:
                                        st.add(i)
                                        added = True
                                        break
                                if not added:
                                    clusters.append((i,set()))
                    clustered_points = []
                    for (_,c) in clusters:
                        x = 0
                        y = 0
                        for p in c:
                            x += p[0]
                            y += p[1]
                        clustered_points.append((len(c), (x / len(c), y / len(c))))
                    clustered_points.sort(reverse=True)

                    draw = frame.copy()
                    for (x0,y0,x1,y1) in goodlines:
                        cv2.line(draw, (x0,y0), (x1,y1), (255,0,0), 1) 

                    if len(clustered_points) < 4:
                        print "Corner points of board not detected"
                        raise Exception

                    #Draw corner points
                    corners = []
                    for (_,c) in clustered_points[:4]:
                        cv2.circle(draw, c, 10, (0,255,0), thickness=3)
                        corners.append(c)
                       
                    POST("draw", draw)
                    
                    img_corners = [(0,0), (640,0), (640,480), (0,480)]
                    corners_sorted = [0,0,0,0]

                    """


                    def get_closest_corner(point):
                        dst = float("inf")
                        crnr = None
                        for pc in possible_corners:
                            d = distance(point, pc)
                            if d < dst:
                                dst = d
                                crnr = pc
                        #print "Closest to %s is %d" % (str(point), dst)
                        if crnr is None:
                            if configs.DEBUG:
                                print "Unable to find any corners"
                            raise IterSkip()
                        return crnr

                    """
                    for c in corners:
                        dst = float("inf")
                        cr = 0
                        for i in range(0,4):
                            d = distance(img_corners[i], c)
                            if d < dst:
                                dst = d
                                cr = i
                        corners_sorted[cr] = list(c)
                    """
                    tl = get_closest_corner((0,0))
                    br = get_closest_corner((configs.IMAGE_W, configs.IMAGE_H))
                    tl = (tl[0] + configs.TL_X, tl[1] + configs.TL_Y)
                    br = (br[0] + configs.BR_X, br[1] + configs.BR_Y)
                    tr = get_closest_corner((configs.IMAGE_W, 0))
                    bl = get_closest_corner((0, configs.IMAGE_H))

                    #Check lengths to ensure valid board layout
                    top_len = distance(tl, tr)
                    left_len = distance(tl, bl)
                    bottom_len = distance(bl, br)
                    right_len = distance(tr, br)
                    sides = np.array([top_len, left_len, bottom_len, right_len])

                    side_dev = float(sides.std()) / sides.mean()
                    if side_dev > configs.SIDE_DEV_THRESH:
                        if configs.DEBUG:
                            print "Invalid board corners detected! (std of %.2f)" % side_dev
                        raise IterSkip()

                    corners_sorted = [tl, tr, br, bl]

                    for cr in corners_sorted:
                        cv2.circle(erode_draw, (int(cr[0]), int(cr[1])), 15, (0,0,255), thickness=3)
                    POST("erode_draw", erode_draw)

                    #sort corners top left, top right, bottom right, bottom left
                    src = np.array(corners_sorted, np.float32)
                    dst = np.array([[0,0],[configs.SIZE,0],[configs.SIZE,configs.SIZE],[0,configs.SIZE]], np.float32)
                    
                    M = cv2.getPerspectiveTransform(src, dst)

                    norm = cv2.warpPerspective(frame, M, (configs.SIZE,configs.SIZE))

                    line_color = (0,0,255)


                    #start norm draw

                    norm_draw = norm.copy()
                    #Draw bounding lines
                    cv2.line(norm_draw, (configs.LSTEP,0), (configs.LSTEP,configs.SIZE), line_color)
                    cv2.line(norm_draw, (configs.SIZE-configs.RSTEP,0), (configs.SIZE-configs.RSTEP,configs.SIZE), line_color)
                    cv2.line(norm_draw, (0,configs.TSTEP), (configs.SIZE,configs.TSTEP), line_color)
                    cv2.line(norm_draw, (0,configs.SIZE-configs.BSTEP), (configs.SIZE,configs.SIZE-configs.BSTEP), line_color)

                    #Draw appropriate gridlines on the board
                    x = configs.LSTEP
                    for i in range(0,14):
                        x += float(configs.SIZE-configs.LSTEP-configs.RSTEP) / 15
                        cv2.line(norm_draw, (int(x),configs.TSTEP), (int(x),configs.SIZE-configs.BSTEP), line_color)

                    y = configs.TSTEP
                    for i in range(0,14):
                        y += float(configs.SIZE-configs.TSTEP-configs.BSTEP) / 15
                        cv2.line(norm_draw, (configs.LSTEP,int(y)), (configs.SIZE-configs.RSTEP,int(y)), line_color)

                    #POST("remapped", norm_draw)

                    #end norm draw
                   
                    if configs.TRAIN:
                        img = get_sub_image(norm, configs.COORD_X, configs.COORD_Y)
                        classify_letter(img, configs.COORD_X, configs.COORD_Y, draw=True)
                    else:

                        blank_b = Board()

                        letter_draw = norm_draw.copy()
                        y = configs.TSTEP
                        #Draw crazy grid thing
                        for j in range(0,15):
                            x = configs.LSTEP
                            for i in range(0,15):
                                img = get_sub_image(norm, i,j)
                                r = classify_letter(img, i, j, draw=(configs.DEBUG and i == configs.COORD_X and j == configs.COORD_Y), blank_board=blank_b)
                                in_blank = (blank_b.get(i,j) is not None)
                                if not in_blank:
                                    new_info(i,j,r)
                                if r is not None:
                                    cv2.putText(letter_draw, str(r.upper()), (int(x)+7,int(y)+22), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255))
                                x += float(configs.SIZE-configs.LSTEP-configs.RSTEP) / 15
                            y += float(configs.SIZE-configs.TSTEP-configs.BSTEP) / 15


                        #Analyze blank board to determine which are blanks and which are empty spaces
                        #TOOD: analyze blank_b
                        for i in range(0,15):
                            for j in range(0,15):
                                r = blank_b.get(i,j)
                                if r is not None:
                                    nearest = np.array(blank_b.get_nearest_not_none(i,j, configs.BLANK_NEIGHBORS))
                                    mean = np.mean(nearest)
                                    std = np.std(nearest)
                                    z = abs(r - mean) / std

                                    if configs.DEBUG and i == configs.COORD_X and j == configs.COORD_Y:
                                        print "Color is %d; mean of nearest %d neighbors is %.2f, std is %.2f, z is %.2f" % (r, configs.BLANK_NEIGHBORS, mean, std, z) 
                                    if z > configs.BLANK_Z_THRESH:
                                        #This is a blank!
                                        new_info(i,j,'-')
                                    else:
                                        new_info(i,j,None) #not a blank

                                    
                        POST("letter draw", letter_draw)


                       
                        #Conduct averaging of the given letters
                        avg_draw = norm_draw.copy()

                        with self.l:
                            y = configs.TSTEP
                            #Draw crazy grid thing
                            for j in range(0,15):
                                x = configs.LSTEP
                                for i in range(0,15):
                                    r = lookup_char(i,j) 
                                    self.board.set(i,j,r)
                                    if r is not None:
                                        cv2.putText(avg_draw, str(r.upper()), (int(x)+7,int(y)+22), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255,255,255))
                                    x += float(configs.SIZE-configs.LSTEP-configs.RSTEP) / 15
                                y += float(configs.SIZE-configs.TSTEP-configs.BSTEP) / 15

                        POST("AVG letter draw", avg_draw)

                except IterSkip as e:
                    pass
                except Exception as e:
                    print "Exception occured: %s" % str(e)
                    print "--------"
                    print traceback.format_exc()
                    print "--------"

                #END PROCESSING

                #next itr
                self.started = True
                key = cv2.waitKey(50)
                rval, frame_raw = vc.read()

            print "Terminating..."

