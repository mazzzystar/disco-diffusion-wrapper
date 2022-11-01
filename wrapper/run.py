from mutils import *
from guided_diffusion.script_util import create_model_and_diffusion
import deepl

USE_TRANSLATE=False  # Set False if you don't need to translate the text_prompts.

if USE_TRANSLATE:
    auth_key = ""  # Replace with your free deepL key. See https://github.com/DeepLcom/deepl-python
    translator = deepl.Translator(auth_key)
torch.backends.cudnn.benchmark = True


def translate(text, source_lang="ZH", target_lang="EN-US"):
    # You can change the source language by your situation.
    res = translator.translate_text(text, source_lang="ZH", target_lang="EN-US")
    return res

def simple_prompts(text, image_path=None):
    text_prompts = {
        0: [f"{text}"],
    }
    image_prompts = None
    if image_path is not None and len(image_path) > 3:
        image_prompts = {
            0:[f'{image_path}:2',],
        }
    print('Text prompts only.') if image_prompts is None else print('Text & Image prompts both used.')
    return text_prompts, image_prompts

    

def load_diffusion_model(batch_name):
    model_config, clip_models, secondary_model, lpips_model = choose_diffusion_model(diffusion_model_name='512x512_diffusion_uncond_finetune_008100', use_secondary_model=True, diffusion_sampling_mode='ddim')
    set_parameters(model_config=model_config, batch_name=batch_name)
    print('Prepping model...')
#     print(model_config)
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(torch.load(f'{model_path}/{diffusion_model}.pt', map_location='cpu'))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if 'qkv' in name or 'norm' in name or 'proj' in name:
            param.requires_grad_()
    if model_config['use_fp16']:
        model.convert_to_fp16()
    print('Finished of loading model.')
    return model, diffusion, clip_models, secondary_model, lpips_model
    

def diffuse(text_prompt, batch_name, image_prompts={}, init_image=None, steps=241, batch_size=1, display_rate=40):
    model, diffusion, clip_models, secondary_model, lpips_model = load_diffusion_model(batch_name)
    global args
    batchNum = args["batchNum"]
    start_frame = args["start_frame"]
    
    print(f'Starting Run: {batch_name}({batchNum}) at frame {start_frame}')
    args['prompts_series'] = split_prompts(text_prompts) if text_prompts else None
    args['image_prompts_series'] = split_prompts(image_prompts) if image_prompts else None
    args['text_prompts'] = text_prompts
    args['image_prompts'] = image_prompts
    args['batch_name'] = batch_name
    args['steps'] = steps
    args['batch_size'] = batch_size
    args['display_rate'] = display_rate
    args['init_image'] = init_image
    
    args = SimpleNamespace(**args)

    gc.collect()
    torch.cuda.empty_cache()
    try:
        do_run(model, diffusion, clip_models, secondary_model, lpips_model)
    except Exception as e:
        print(e)
        print("""If CUDA out of memory, you can decrease the number of CLIP model by setting it to False.
                    But make sure at least 1 CLIP model is setting to True.
                    See https://github.com/mazzzystar/disco-diffusion-wrapper/blob/148262c34ea45f094a9d4ef1536a80f1c1201602/wrapper/mutils.py#L1264-L1271""")
        pass
    finally:
        print('Seed used:', args.seed)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    print(str(sys.argv[1]))
    input_text = str(sys.argv[1])
    input_img_path = None # If do not need input img，then set None or ""
    if USE_TRANSLATE:
        input_text = translate(input_text)
    if translator is not None:
        del translator
    text_prompts, image_prompts = simple_prompts(input_text, input_img_path)
    print(text_prompts)
    outdirName = "my-test"
    diffuse(text_prompts, outdirName, steps=200, image_prompts=image_prompts, init_image=input_img_path, display_rate=40)
    
