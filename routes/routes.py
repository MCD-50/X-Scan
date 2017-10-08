from controllers import *

routes = [
    (
        r"/",
        upload.UploadHandler
    ),
    (
        r"/report",
        report.ReportHandler
    )
]
