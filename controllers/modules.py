"""
Includes all the required modules to include
"""

# tornado modules
from tornado.ioloop import IOLoop
from tornado.escape import json_encode
from tornado.web import RequestHandler, Application
from tornado.httpserver import HTTPServer
from tornado.gen import coroutine
from tornado.options import define, options

# external modules
import os
from os.path import join, dirname
import uuid
from collections import defaultdict
