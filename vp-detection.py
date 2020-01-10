import urllib

import flask
from flask import request, jsonify

from Single_image_test import image_VP

app = flask.Flask(__name__)
app.config["DEBUG"] = True

image64 = [{
    'image_base64' : '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'
}]
#send as encoded base64 string


@app.route('/', methods=['GET'])
def home():
    return "<h1>Vanishing Point API</h1><p>This site is a prototype API.</p>"

@app.route('/api/coordinates', methods=['GET'])
def api_all():
    if 'fileb' in request.args:
        fileb = request.args['fileb']
    else:
        return "Error: No string provided. Please specify a string."

    results = image_VP(fileb)
    return jsonify(results)


if __name__ == '__main__':
    app.run()