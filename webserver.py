import tornado.httpserver
import tornado.ioloop
import tornado.web
import webbrowser
from threading import Thread
from scoring import *
import cgi
import twl
import os

PORT = 80

#This global state is bad practice; TODO: Fix
global global_scoreboard
global global_board
global handle_count
handle_count = 0

class MainHandler(tornado.web.RequestHandler):  

    def post(self, val):
        params = dict(cgi.parse_qsl(self.request.body))
        addr = self.request.remote_ip
        global handle_count
        global global_scoreboard, global_board #TODO: Fix
        self.set_header("Content-Type", "text/html")
        if params["m"] == "check":
            self.write(str(handle_count))
        elif params["m"] == "scores":
            #Generate scoreboard dynamically
            scores = global_scoreboard.get_scores()
            sb = "<table>"
            for (x,y) in scores:
                sb += "<tr>"
                sb += "<td align=right><b>%s:</b></td><td>%s</td>" % (str(x), str(y))
                sb += "</tr>"
            sb += "</table>"
            self.write(sb)
        elif params["m"] == "board":
            sg =  "<table width=\"418px\" height=\"451px\" style=\"background-image: url('scrabble.jpg')\" cellpadding=0 cellspacing=0>"
            for j in range(0,15):
                sg += "<tr height=\"28\" valign=\"middle\">"
                for i in range(0,15):
                    p = global_board.get(i,j)
                    at = " bgcolor=\"#FFCC66\"" if p is not None else ""
                    ltr_txt = ("<b><font color=%s>%s</font></b>" % ("black" if not isinstance(p, Blank) else "blue", str(p).upper()) if p is not None else "")
                    pts = str(get_letter_points(p)) if p is not None else ""
                    sg += "<td width=\"29\" valign=\"middle\" align=\"center\"><table cellpadding=0 cellspacing=0 height=23 width=23%s><tr valign=middle><td align=center valign=middle>%s</td><td valign=bottom><font size=1>%s</font></td></tr></tr></table></td>" % (at, ltr_txt, pts)
                sg += "</tr>"
            sg += "</table>"
            self.write(sg)
        elif params["m"] == "dict":
            word = params["d"]
            if twl.check(word.strip().lower()):
                os.system("beep -f 150 -l 10 &>/dev/null &") #dictionary lookup beep
                self.write("The word %s is in the dictionary." % word)
            else:
                os.system("beep -f 100 -l 10 &>/dev/null &") #dictionary lookup beep
                self.write("The word %s IS NOT the dictionary." % word)

    def get(self, pth):
        addr = self.request.remote_ip

        if pth.endswith(".html") or pth.endswith(".htm") or pth.endswith(".txt") or pth=="":
            self.set_header("Content-Type", "text/html")
        if pth.endswith(".png"):
            self.set_header("Content-Type", "image/png")
        if pth.endswith(".jpg"):
            self.set_header("Content-Type", "image/jpg")

        if "index" in pth or pth == "":
            ipage = open("web/index.html", "rb")
            ihtml = ipage.read()
            self.write(ihtml)
        else:
            try:
                #Grab the path from the web dir
                page = open("web/%s" % pth, "rb")
                self.write(page.read())
            except IOError:
                self.set_header("Content-Type", "text/html")
                self.set_status(200)
                self.write("404: Not found")


class ScrabbleServer(Thread):
    def __init__(self, game_board, scoreboard):
        Thread.__init__(self)
        self.io_loop = None
        global global_scoreboard, global_board #TODO: Fix
        global_scoreboard = scoreboard
        global_board = game_board

    def kill(self):
        self.io_loop.stop()

    def refresh(self):
        global handle_count
        handle_count += 1

    def run(self):

        print 'WEB SERVER DISABLED'
        return

        application = tornado.web.Application([
            (r"/(.*)", MainHandler),

        ])

        #start server
        print "Starting Scrabble Server"
        http_server = tornado.httpserver.HTTPServer(application)
        http_server.listen(PORT)
        self.io_loop = tornado.ioloop.IOLoop.instance()

        self.io_loop.start()
        print "Scrabble server stopped"
       
