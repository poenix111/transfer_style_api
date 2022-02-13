from urllib import response
from flask import Flask, request, make_response, jsonify
import os
import sys
import json
import requests
import logging.handlers
import datetime
import re
import base64
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image
import time
from transfer import Transfer

app = Flask(__name__)
transfer = Transfer()

@app.route('/transfer', methods=['POST'])
def transfer_image():
    if request.method == 'POST':
        data = request.get_json()
        content_image = data['content_image']
        style_image = data['style_image']
        iterations = data['iterations'] or 10
        best_img, loss = transfer.run_style_transfer(content_image, style_image, num_iterations=iterations)
        response = {
            'best_img': best_img,
        }
        return make_response(jsonify(response), 200)
# Hello World