import threading


buffer = []
outputs = []


def worker():
    global buffer, outputs

    import os
    import json
    import time
    import shared
    import random
    import modules.default_pipeline as pipeline
    import modules.path
    import modules.patch

    from PIL import Image
    from PIL.PngImagePlugin import PngInfo
    from modules.sdxl_styles import apply_style, aspect_ratios
    from modules.util import generate_temp_filename

    try:
        async_gradio_app = shared.gradio_root
        flag = f'''App started successful. Use the app with {str(async_gradio_app.local_url)} or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}'''
        if async_gradio_app.share:
            flag += f''' or {async_gradio_app.share_url}'''
        print(flag)
    except Exception as e:
        print(e)

    def handler(task):
        prompt, negative_prompt, style_selction, performance_selction, \
        aspect_ratios_selction, image_number, image_seed, sharpness, save_metadata, sampler_name, \
        sampler_steps_speed, switch_step_speed, sampler_steps_quality, switch_step_quality, cfg, \
        base_model_name, refiner_model_name, l1, w1, l2, w2, l3, w3, l4, w4, l5, w5 = task

        loras = [(l1, w1), (l2, w2), (l3, w3), (l4, w4), (l5, w5)]

        modules.patch.sharpness = sharpness

        pipeline.refresh_base_model(base_model_name)
        pipeline.refresh_refiner_model(refiner_model_name)
        pipeline.refresh_loras(loras)
        pipeline.clean_prompt_cond_caches()

        p_txt, n_txt = apply_style(style_selction, prompt, negative_prompt)

        if performance_selction == 'Speed':
            steps = sampler_steps_speed
            switch = round(sampler_steps_speed * switch_step_speed)
        else:
            steps = sampler_steps_quality
            switch = round(sampler_steps_quality * switch_step_quality)

        width, height = aspect_ratios[aspect_ratios_selction]

        results = []
        seed = image_seed
        if not isinstance(seed, int) or seed < 0 or seed > 2**31 - 1:
            seed = random.randint(0, 2**31 - 1)

        all_steps = steps * image_number

        def callback(step, x0, x, total_steps, y):
            done_steps = i * steps + step
            outputs.append(['preview', (
                int(100.0 * float(done_steps) / float(all_steps)),
                f'Step {step}/{total_steps} in the {i}-th Sampling',
                y)])

        for i in range(image_number):
            imgs = pipeline.process(p_txt, n_txt, steps, switch, width, height, seed, sampler_name, cfg, callback=callback)

            pnginfo = None
            if save_metadata != 'Disabled':
                prompt = {
                    'p_txt': p_txt, 'n_txt': n_txt, 'steps': steps, 'switch': switch, 'cfg': cfg,
                    'width': width, 'height': height, 'seed': seed, 'sampler_name': sampler_name,
                    'base_model_name': base_model_name, 'refiner_model_name': refiner_model_name,
                    'l1': l1, 'w1': w1, 'l2': l2, 'w2': w2, 'l3': l3, 'w3': w3,
                    'l4': l4, 'w4': w4, 'l5': l5, 'w5': w5,
                    'sharpness': sharpness, 'software': 'Fooocus'
                }
                if save_metadata == 'PNG':
                    pnginfo = PngInfo()
                    pnginfo.add_text("prompt", json.dumps(prompt))

            for x in imgs:
                local_temp_filename = generate_temp_filename(folder=modules.path.temp_outputs_path, extension='png')
                os.makedirs(os.path.dirname(local_temp_filename), exist_ok=True)
                Image.fromarray(x).save(local_temp_filename, pnginfo=pnginfo)
                if save_metadata == 'JSON':
                    json_path = local_temp_filename.replace('.png', '.json')
                    with open(json_path, 'w') as jsonfile:
                        json.dump(prompt, jsonfile)
                        jsonfile.close()

            seed += 1
            results += imgs

        outputs.append(['results', results])
        return

    while True:
        time.sleep(0.01)
        if len(buffer) > 0:
            task = buffer.pop(0)
            handler(task)
    pass


threading.Thread(target=worker, daemon=True).start()
