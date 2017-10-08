from controllers.modules import *

__UPLOADS__ = "uploads/"

class UploadHandler(RequestHandler):
    """
    Class to upload the file and generate report
    """

    def get(self):
        self.render("upload.html")

    def post(self):

        # upload audio file in server
        voice = self.request.files["filearg"][0]
        extn = os.path.splitext(voice['filename'])[1]
        cname = str(uuid.uuid4()) + extn
        fh = open(__UPLOADS__ + voice['filename'], 'wb')
        fh.write(voice['body'])
        fh.close()

        # self.redirect("/report?file=" + /cname)
