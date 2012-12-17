import cv2
import cv
import numpy as np
from math import *
import sys
import gc
from scipy.interpolate import griddata
import configs

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
        print "Reloaded responses:"
        print str(responses)
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

print "Responses contains: %s" % str(responses)


def classify_letter(image, draw=False):
    image = cv2.resize(image, (128,128))
    if draw:
        POST("letter start", image)

    luv = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))

    rgb = cv2.split(image)
    l_chan = luv[2]
    
    chan = rgb[0]
    POST("OC", chan)
    print "average value is %s" % str(cv2.mean(chan))

    if draw:
        POST("letter L", l_chan)

    blur = cv2.GaussianBlur(l_chan, (configs.LETTER_BLUR,configs.LETTER_BLUR), 0)

    thresh = cv2.adaptiveThreshold(blur, 255, 0, 1, configs.LETTER_THRESH, configs.LETTER_BLOCK)
    
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5,5))
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
        if sz>750:
            [x,y,w,h] = cv2.boundingRect(cnt)
            d = abs(cv2.pointPolygonTest(cnt, (128.0/2,128.0/2), measureDist=True))
            if d == 0:
                continue
            if d < mindst:
                mindst = d
                minc = cnt

    if mindst > 30:
        return None

    if minc is None:
        return None
            
    [x,y,w,h] = cv2.boundingRect(minc)
    cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),1)
    if draw:
        POST("letter contour", im)

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
        return chr(int((results[0][0])) + 96)


vc = cv2.VideoCapture(-1)

if vc.isOpened(): # try to get the first frame
    rval, frame = vc.read()
else:
    rval = False


while rval:

    reload(configs)
    element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (configs.ERODE_RAD,configs.ERODE_RAD))
    element2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (configs.DILATE_RAD,configs.DILATE_RAD))

    #BEGIN PROCESSING
    try:

        #POST("RAW", frame)

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

        for c in corners:
            dst = float("inf")
            cr = 0
            for i in range(0,4):
                d = distance(img_corners[i], c)
                if d < dst:
                    dst = d
                    cr = i
            corners_sorted[cr] = list(c)
    
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

        POST("remapped", norm_draw)

        #end norm draw
       
        if configs.TRAIN:
            img = get_sub_image(norm, configs.COORD_X, configs.COORD_Y)
            classify_letter(img, draw=True)
        else:

            letter_draw = norm_draw.copy()
            y = configs.TSTEP
            #Draw crazy grid thing
            for i in range(0,15):
                x = configs.LSTEP
                for j in range(0,15):
                    img = get_sub_image(norm, j,i)
                    r = classify_letter(img)
                    if r is not None:
                        cv2.putText(letter_draw, str(r.upper()), (int(x)+8,int(y)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255))
                    x += float(configs.SIZE-configs.LSTEP-configs.RSTEP) / 15
                y += float(configs.SIZE-configs.TSTEP-configs.BSTEP) / 15

            POST("letter draw", letter_draw)


    except Exception as e:
        print "Exception occured: %s" % str(e)

    #END PROCESSING

    #next itr
    key = cv2.waitKey(50)
    rval, frame = vc.read()

print "Terminating..."

