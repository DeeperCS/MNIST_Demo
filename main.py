from flask import Flask
from flask import Flask, request, url_for, render_template

import json
from flask import jsonify
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import pickle

app = Flask(__name__)
# import pdb; pdb.set_trace()
with open('weights-10.pkl', 'rb') as f:
    weights = pickle.load(f)

def conv(data_input, conv_filter_weight, conv_filter_bias):
    batch_size, channel_input, height_input, width_input = data_input.shape
    num_kernel, num_channel, height_kernel, width_kernel= conv_filter_weight.shape
    assert(channel_input==num_channel)
    output = np.zeros((batch_size, num_kernel, height_input-height_kernel+1, width_input-width_kernel+1))
    for i_batch in range(batch_size):
        for i_kernel in range(num_kernel):
            for i in range(output.shape[2]):
                for j in range(output.shape[3]):
                    output[i_batch, i_kernel, i, j] = \
                    np.sum(conv_filter_weight[i_kernel, ...] * data_input[i_batch, :, i:i+height_kernel, j:j+width_kernel]) + conv_filter_bias[i_kernel]
    return output

def max_pooling(data_input, pooling_height=2, pooling_width=2):
    batch_size, channel_input, height_input, width_input = data_input.shape
    output = np.zeros((batch_size, channel_input, height_input//2, width_input//2))
    for i_batch in range(output.shape[0]):
        for i_channel in range(channel_input):
            for i in range(output.shape[2]):
                for j in range(output.shape[3]):
                    output[i_batch, i_channel, i, j] = np.max(data_input[i_batch, i_channel, i*pooling_height:(i+1)*pooling_height, j*pooling_width:(j+1)*pooling_width])
    return output

def fc(input_data, weight_fc, weight_fc_bias):
    batch_size, input_dim = input_data.shape
    output = np.dot(input_data, weight_fc.T) + weight_fc_bias[np.newaxis, :]
    return output

def relu(data_input):
    return np.maximum(data_input, 0)

def feed_forward(data_input, weights):
    conv_filter1 = weights['conv1.weight']
    conv_filter1_bias = weights['conv1.bias']
    conv_filter2 = weights['conv2.weight']
    conv_filter2_bias = weights['conv2.bias']
    weight_fc1 = weights['fc1.weight']
    weight_fc1_bias = weights['fc1.bias']
    weight_fc2 = weights['fc2.weight']
    weight_fc2_bias = weights['fc2.bias']

    output1 = conv(data_input, conv_filter1, conv_filter1_bias)
    output2 = relu(max_pooling(output1))
    output3 = conv(output2, conv_filter2, conv_filter2_bias)
    output4 = relu(max_pooling(output3))
    view = output4.reshape(output4.shape[0], -1)
    output5 = relu(fc(view, weight_fc1, weight_fc1_bias))
    output6 = fc(output5, weight_fc2, weight_fc2_bias)
    return output6

def predict(test_img, weights):
	output = feed_forward(test_img, weights)
	# import pdb; pdb.set_trace()
	output_exp = np.exp(output)
	output_prob = output_exp / np.sum(output_exp)
	# pred = np.argsort(output)
	return output_prob[0] # pred[0][::-1]


res = {"result": 0,
       "data": [], 
       "error": ''}

@app.route('/', methods=['GET', 'POST'])
def main_page():
	if request.method == 'POST':
		# 

		file = request.form['data']

		starter = file.find(',')
		image_data = file[starter+1:]
		image_data = bytes(image_data, encoding="ascii")
		im = Image.open(BytesIO(base64.decodestring(image_data)))
		# import pdb; pdb.set_trace()
		imresize = im.resize((28,28))
		im_arr = np.array(imresize)[...,0]
		im_arr = (255 - im_arr) / 255.
		im_arr = im_arr[np.newaxis,np.newaxis]
		preds = predict(im_arr, weights)
		print(preds)
		res['result'] = 1
		res['data'] = [float(num) for num in preds]

		# msg = {'result':1, 'data':[float(num) for num in preds] }
		return jsonify(res)


	return render_template('index.html')

if __name__=="__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=5000)