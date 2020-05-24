from flask import Flask,render_template, Response, request
import sys
# Tornado web server
from tornado.wsgi import WSGIContainer
from tornado.httpserver import HTTPServer
import tornado.ioloop
from tornado.ioloop import IOLoop
from text2speech import T2S
import os

t2s = T2S()
speakers = [x for x in list(t2s.tt_sp_name_lookup.keys()) if "(Music)" not in x]
infer_dir = "server_infer"

# default config
sample_background_text = "Enter text." # this is the faded out text when nothing has been entered
sample_speaker = ["(Show) My Little Pony_Twilight",]
sample_style_mode = "torchmoji_hidden"
sample_textseg_mode="segment_by_line"
sample_batch_mode="nochange"
sample_max_attempts=128
sample_max_duration_s=12
sample_batch_size=128
sample_dyna_max_duration_s = 0.3
# Initialize Flask.
app = Flask(__name__)

@app.route('/tts', methods=['GET', 'POST'])
def texttospeech():
    if request.method == 'POST':
        result = request.form
        speaker = request.form.getlist('input_speaker')
        text = result['input_text']
        style_mode = result['input_style_mode']
        textseg_mode = result['input_textseg_mode']
        batch_mode = result['input_batch_mode']
        max_attempts = result['input_max_attempts']
        max_duration_s = result['input_max_duration_s']
        batch_size = result['input_batch_size']
        dyna_max_duration_s = result['dyna_max_duration_s']
        print(f"REQUEST RECIEVED\nText: '{text}'\nSpeaker: '{speaker}'\nStyle Mode: '{style_mode}'\nBatch Size: {result['input_batch_size']}")
        
        filename = t2s.infer(text, speaker, style_mode)
        print(f"GENERATED {filename}")
        return render_template('main.html',
                                sp_len=len(speakers),
                                speakers_available_short=[sp.split("_")[-1] for sp in speakers],
                                speakers_available=speakers,
                                current_text=text,
                                voice=filename,
                                sample_text=sample_background_text,
                                speaker=speaker,
                                style_mode=style_mode,
                                textseg_mode=textseg_mode,
                                batch_mode=batch_mode,
                                max_attempts=max_attempts,
                                max_duration_s=max_duration_s,
                                batch_size=batch_size,
                                dyna_max_duration_s=dyna_max_duration_s,)

#Route to render GUI
@app.route('/')
def show_entries():
    return render_template('main.html',
                            sp_len=len(speakers),
                            speakers_available_short=[sp.split("_")[-1] for sp in speakers],
                            speakers_available=speakers,
                            current_text='',
                            sample_text=sample_background_text,
                            voice=None,
                            speaker=sample_speaker,
                            style_mode=sample_style_mode,
                            textseg_mode=sample_textseg_mode,
                            batch_mode=sample_batch_mode,
                            max_attempts=sample_max_attempts,
                            max_duration_s=sample_max_duration_s,
                            batch_size=sample_batch_size,
                            dyna_max_duration_s=sample_dyna_max_duration_s,)

#Route to stream music
@app.route('/<voice>', methods=['GET'])
def streammp3(voice):
    def generate():
        with open(os.path.join(infer_dir, voice), "rb") as fwav:# open audio_path
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)
    
    return Response(generate(), mimetype="audio/mp3")

#launch a Tornado server with HTTPServer.
if __name__ == "__main__":
    port = 5000
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(port)
    io_loop = tornado.ioloop.IOLoop.current()
    io_loop.start()