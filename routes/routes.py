from controllers import *

routes = [
    (
        r"/",
        index.IndexHandler
    ),
    (
        r"/upload",
        upload.UploadHandler
    ),
    (
        r"/report",
        report.ReportHandler
    )
]
