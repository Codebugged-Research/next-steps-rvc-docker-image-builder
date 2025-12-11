from flask import Flask, request, jsonify, send_file
import os
import subprocess
import tempfile
import uuid
import math
import shutil

os.environ['HOME'] = '/tmp'
os.environ['TMPDIR'] = '/tmp'
os.makedirs('/tmp/rvc_config', exist_ok=True)

app = Flask(__name__)
RVC_DIR = "/app/rvc"

def extract_audio(video_path, audio_path):
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "44100", "-ac", "2",
        audio_path, "-y"
    ], check=True, capture_output=True)
    return audio_path

def get_audio_duration(audio_path):
    result = subprocess.run([
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "csv=p=0", audio_path
    ], capture_output=True, text=True, check=True)
    return float(result.stdout.strip())

def split_audio_chunks(audio_path, chunk_duration=180):
    duration = get_audio_duration(audio_path)
    num_chunks = math.ceil(duration / chunk_duration)
    temp_dir = tempfile.mkdtemp()
    chunk_files = []
    
    for i in range(num_chunks):
        start_time = i * chunk_duration
        current_duration = min(chunk_duration, duration - start_time)
        chunk_file = os.path.join(temp_dir, f"chunk_{i:03d}.wav")
        
        subprocess.run([
            "ffmpeg", "-i", audio_path,
            "-ss", str(start_time),
            "-t", str(current_duration),
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            chunk_file, "-y"
        ], check=True, capture_output=True)
        chunk_files.append(chunk_file)
    
    return chunk_files, temp_dir

def process_audio_chunk(chunk_path, gender):
    original_dir = os.getcwd()
    os.chdir(RVC_DIR)
    
    unique_id = str(uuid.uuid4())[:8]
    output_path = f"/tmp/output_{unique_id}.wav"
    
    if gender.lower() == "female":
        model = "NikkiDorkDiaries.pth"
        idx = "added_IVF110_Flat_nprobe_1_NikkiDorkDiaries_v2.index"
    else:
        model = "AndyField_350e_5950s.pth"
        idx = "AndyField.index"
    
    try:
        subprocess.run([
            "python3", "tools/infer_cli.py",
            "--input_path", chunk_path,
            "--index_path", f"assets/weights/{idx}",
            "--f0method", "harvest",
            "--opt_path", output_path,
            "--model_name", model,
            "--index_rate", "0.7",
            "--device", "cuda:0",
            "--is_half", "True",
            "--filter_radius", "3",
            "--resample_sr", "0",
            "--rms_mix_rate", "0.25",
            "--protect", "0.33",
            "--f0up_key", "0"
        ], check=True, capture_output=True, text=True, env={**os.environ, 'HOME': '/tmp'})
        
        os.chdir(original_dir)
        return output_path
    except subprocess.CalledProcessError as e:
        os.chdir(original_dir)
        raise Exception(f"RVC processing failed: {e.stderr}")

def combine_audio_chunks(processed_chunks, output_path):
    if len(processed_chunks) == 1:
        subprocess.run([
            "ffmpeg", "-i", processed_chunks[0],
            "-c", "copy", output_path, "-y"
        ], check=True, capture_output=True)
    else:
        inputs = []
        filter_parts = []
        for i, chunk in enumerate(processed_chunks):
            inputs.extend(["-i", chunk])
            filter_parts.append(f"[{i}:0]")
        
        filter_complex = "".join(filter_parts) + f"concat=n={len(processed_chunks)}:v=0:a=1[out]"
        cmd = ["ffmpeg"] + inputs + [
            "-filter_complex", filter_complex,
            "-map", "[out]", output_path, "-y"
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    
    return output_path

def replace_video_audio(video_path, audio_path, output_path):
    subprocess.run([
        "ffmpeg", "-i", video_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", output_path, "-y"
    ], check=True, capture_output=True)
    return output_path

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"}), 200

@app.route('/process', methods=['POST'])
def process_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400
    
    video_file = request.files['video']
    gender = request.form.get('gender', 'Female')
    
    temp_dir = tempfile.mkdtemp()
    chunks_dir = None
    
    try:
        video_path = os.path.join(temp_dir, "input.mp4")
        video_file.save(video_path)
        
        audio_path = os.path.join(temp_dir, "audio.wav")
        extract_audio(video_path, audio_path)
        
        chunk_files, chunks_dir = split_audio_chunks(audio_path)
        
        processed_chunks = []
        for i, chunk_file in enumerate(chunk_files):
            print(f"Processing chunk {i+1}/{len(chunk_files)}")
            processed_chunk = process_audio_chunk(chunk_file, gender)
            processed_chunks.append(processed_chunk)
        
        final_audio = os.path.join(temp_dir, "final_audio.wav")
        combine_audio_chunks(processed_chunks, final_audio)
        
        output_video = os.path.join(temp_dir, "output.mp4")
        replace_video_audio(video_path, final_audio, output_video)
        
        return send_file(output_video, mimetype='video/mp4', as_attachment=True, download_name='converted.mp4')
    
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if temp_dir:
                shutil.rmtree(temp_dir, ignore_errors=True)
            if chunks_dir:
                shutil.rmtree(chunks_dir, ignore_errors=True)
        except:
            pass

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)