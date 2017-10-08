from controllers.modules import *

__UPLOADS__ = "uploads/"

class IndexHandler(RequestHandler):
    """
    Class to upload the file and generate report
    """

    def get(self):
        self.render("upload.html")
