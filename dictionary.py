import mechanize, cookielib
from threading import Thread

#Hackish parser to look up defintiions from the scrabble website

URL = "http://www.hasbro.com/scrabble/en_US/search.cfm"

class DictLookup(Thread):
    def __init__(self, word):
        Thread.__init__(self)
        self.word = word
        self.start()

    def run(self):

        try:

            # Browser
            br = mechanize.Browser(factory=mechanize.RobustFactory())

            # Cookie Jar
            cj = cookielib.LWPCookieJar()
            br.set_cookiejar(cj)

            # Browser options
            br.set_handle_equiv(True)
            br.set_handle_redirect(True)
            br.set_handle_referer(True)
            br.set_handle_robots(False)

            # Follows refresh 0 but not hangs on refresh > 0
            br.set_handle_refresh(mechanize._http.HTTPRefreshProcessor(), max_time=1)

            # User-Agent (fake agent to google-chrome linux x86_64)
            br.addheaders = [('User-agent','Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.56 Safari/535.11'),
                             ('Accept', 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'),
                             ('Accept-Encoding', 'deflate,sdch'),
                             ('Accept-Language', 'en-US,en;q=0.8'),
                             ('Accept-Charset', 'ISO-8859-1,utf-8;q=0.7,*;q=0.3')]

            print "Opening URL..."

            br.open(URL)

            print "Url Opened..."

            nf = 0
            i = 0
            for f in br.forms():
                if "frmDict1" in str(f):
                    nf = i
                    break
                i += 1

            br.select_form(nr=nf)

            print "Submitting Word..."

            br.form["dictWord"] = self.word
            br.find_control("exact").items[0].selected=True
            br.submit()

            print "Word submitted..."

            pageresp = br.response().read()


            lst = pageresp.split("<div id=\"dictionary\">")[1].split("</div>")[0].split("<br />")
            wrd = lst[2]
            dfn = wrd.split("</p>")[0]

            print "Definition of %s: %s" % (self.word, dfn)
        except:
            print "Failed to resolve word %s." % self.word



