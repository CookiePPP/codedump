from flask import Flask,render_template, Response, request, send_from_directory
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
tacotron_conf = [[name,details] if os.path.exists(details['modelpath']) else [f"[MISSING]{name}",details] for name, details in list(t2s.conf['tacotron']['models'].items())]
waveglow_conf = [[name,details] if os.path.exists(details['modelpath']) else [f"[MISSING]{name}",details] for name, details in list(t2s.conf['waveglow']['models'].items())]
infer_dir = "server_infer"

# default html config
sample_tacotron = t2s.conf['tacotron']['default_model']
sample_waveglow = t2s.conf['waveglow']['default_model']
sample_current_text = "" # default text entered
sample_background_text = "Enter text." # this is the faded out text when nothing has been entered
sample_speaker = ["(Show) My Little Pony_Twilight",]
sample_style_mode = "torchmoji_hidden"
sample_textseg_mode="segment_by_line"
sample_batch_mode="nochange"
sample_max_attempts=256
sample_max_duration_s=20
sample_batch_size=256
sample_dyna_max_duration_s = 0.10
sample_use_arpabet = "on"
sample_target_score = 0.75
sample_multispeaker_mode = "random"
sample_cat_silence_s = 0.1

use_localhost = t2s.conf['localhost']

# Initialize Flask.
app = Flask(__name__)

@app.route('/tts', methods=['GET', 'POST'])
def texttospeech():
    if request.method == 'POST':
        print("REQUEST RECIEVED")
        # grab all the form inputs
        result = request.form
        
        assert result.get('input_text'), "No input_text found in request form!"
        
        speaker = result.getlist('input_speaker')
        text = result.get('input_text')
        style_mode = result.get('input_style_mode')
        textseg_mode = result.get('input_textseg_mode')
        batch_mode = result.get('input_batch_mode')
        max_attempts = int(result.get('input_max_attempts')) if result.get('input_max_attempts') else 256
        max_duration_s = float(result.get('input_max_duration_s'))
        batch_size = int(result.get('input_batch_size'))
        dyna_max_duration_s = float(result.get('input_dyna_max_duration_s'))
        use_arpabet = True if result.get('input_use_arpabet') == "on" else False
        target_score = float(result.get('input_target_score'))
        multispeaker_mode = result.get('input_multispeaker_mode')
        cat_silence_s = float(result.get('input_cat_silence_s'))
        wg_current = result.get('input_wg_current')
        tt_current = result.get('input_tt_current')
        print(result)
        
        # update tacotron if needed
        if t2s.tt_current != tt_current:
            t2s.update_tt(tt_current)
        
        # update waveglow if needed
        if t2s.wg_current != wg_current:
            t2s.update_wg(wg_current)
        
        # CRLF to LF
        text = text.replace('\r\n','\n')
        
        # generate an audio file from the inputs
        filename, gen_time, gen_dur, total_specs, n_passes = t2s.infer(text, speaker, style_mode, textseg_mode, batch_mode, max_attempts, max_duration_s, batch_size, dyna_max_duration_s, use_arpabet, target_score, multispeaker_mode, cat_silence_s)
        print(f"GENERATED {filename}\n\n")
        
        # send updated webpage back to client along with page to the file
        return render_template('main.html',
                                use_localhost=use_localhost,
                                tacotron_conf=tacotron_conf,
                                tt_current=tt_current,
                                tt_len=len(tacotron_conf),
                                waveglow_conf=waveglow_conf,
                                wg_current=wg_current,
                                wg_len=len(waveglow_conf),
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
                                dyna_max_duration_s=dyna_max_duration_s,
                                use_arpabet=result.get('input_use_arpabet'),
                                target_score=target_score,
                                gen_time=round(gen_time,2),
                                gen_dur=round(gen_dur,2),
                                total_specs=total_specs,
                                n_passes=n_passes,
                                multispeaker_mode=multispeaker_mode,
                                cat_silence_s=cat_silence_s,)

#Route to render GUI
@app.route('/')
def show_entries():
    return render_template('main.html',
                            use_localhost=use_localhost,
                            tacotron_conf=tacotron_conf,
                            tt_current=sample_tacotron,
                            tt_len=len(tacotron_conf),
                            waveglow_conf=waveglow_conf,
                            wg_current=sample_waveglow,
                            wg_len=len(waveglow_conf),
                            sp_len=len(speakers),
                            speakers_available_short=[sp.split("_")[-1] for sp in speakers],
                            speakers_available=speakers,
                            current_text=sample_current_text,
                            sample_text=sample_background_text,
                            voice=None,
                            speaker=sample_speaker,
                            style_mode=sample_style_mode,
                            textseg_mode=sample_textseg_mode,
                            batch_mode=sample_batch_mode,
                            max_attempts=sample_max_attempts,
                            max_duration_s=sample_max_duration_s,
                            batch_size=sample_batch_size,
                            dyna_max_duration_s=sample_dyna_max_duration_s,
                            use_arpabet=sample_use_arpabet,
                            target_score=sample_target_score,
                            gen_time="",
                            gen_dur="",
                            total_specs="",
                            n_passes="",
                            multispeaker_mode=sample_multispeaker_mode,
                            cat_silence_s=sample_cat_silence_s,)

#Route to stream music
@app.route('/<voice>', methods=['GET'])
def streammp3(voice):
    print("AUDIO_REQUEST: ", request)
    def generate():
        with open(os.path.join(infer_dir, voice), "rb") as fwav:# open audio_path
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)
    
    stream_audio = False
    if stream_audio: # don't have seeking working atm
        return Response(generate(), mimetype="audio/wav")
    else:
        return send_from_directory(infer_dir, voice)

#launch a Tornado server with HTTPServer.
if __name__ == "__main__":
    port = 5000
    http_server = HTTPServer(WSGIContainer(app))
    http_server.listen(port)
    io_loop = tornado.ioloop.IOLoop.current()
    io_loop.start()