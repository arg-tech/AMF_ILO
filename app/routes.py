from flask import redirect, request, render_template, jsonify
from . import application
import json
from app.ilo import illocution_identification


@application.route('/', methods=['GET', 'POST'])
def amf_illocutions():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        ff = open(f.filename, 'r')
        content = json.load(ff)
        response = illocution_identification(content)
        print(response)
        return jsonify(response)
    elif request.method == 'GET':
        return render_template('docs.html')
 
 
