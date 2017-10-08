from controllers.modules import *

__UPLOADS__ = "uploads/"

class ReportHandler(RequestHandler):
    """
    Class to upload the file and generate report
    """

    def get(self):
        # upload audio file in server
        fle = self.get_argument("file")
        data = {
                "fname" : __UPLOADS__ + fle
        }
        self.render("report.html", data = data)
