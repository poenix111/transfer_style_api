from urllib import response
from flask import Flask, request, make_response, jsonify
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from PIL import Image
from image_style_transfer.image_transfer import ImageTransfer


app = Flask(__name__)
image_transfer = ImageTransfer()

@app.route('/image-transfer', methods=['POST'])
def transfer_image():
    if request.method == 'POST':
        data = request.get_json()
        content_image = data['content_image']
        style_image = data['style_image']
        iterations = data['iterations'] or 10
        best_img, loss = image_transfer.run_style_transfer(content_image, style_image, num_iterations=iterations)
        response = {
            'best_img': best_img,
        }
        return make_response(jsonify(response), 200)