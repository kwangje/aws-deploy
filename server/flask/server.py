import random
import os
import wave
from flask import Flask, request, jsonify, redirect, url_for, render_template, abort
from voice_type_service import Voice_Type_Detect_Service
from werkzeug.utils import secure_filename


ALLOWED_EXTENSIONS = {"wav"}
# UPLOAD_FOLDER = (".../.../...")

app = Flask(__name__)
# app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        audio_file = request.files["file"]
        file_name_2 = secure_filename(audio_file.filename)
        audio_file.save(file_name_2)

        service = Voice_Type_Detect_Service()
        predicted_type = service.age_clf_single(file_name_2)
        result = {
            "filename_2": file_name_2,
            "voice_type": predicted_type,
        }
        os.remove(file_name_2)

        return jsonify(result)

    else:
        return jsonify({"error": "error during prediction"})


if __name__ == "__main__":
    app.run(debug=False)
